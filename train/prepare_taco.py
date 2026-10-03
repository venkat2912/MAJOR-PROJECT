"""Downloads the TACO litter dataset and converts it to YOLO detection format.

TACO's 60 fine-grained categories are folded into 6 material classes. The
mapping is printed so it can be checked before training.

Each image is also cut into the same overlapping tiles the engine uses, so the
model is trained on objects at the scale it sees them at inference.

The image host throttles bulk downloads. Images already on disk are skipped, so
running this again resumes where the last run stopped.
"""
import argparse
import io
import json
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image, ImageOps

ANN_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
CLASSES = ["plastic", "paper_cardboard", "metal", "glass", "organic", "other"]

PAPER = ("paper", "carton", "cardboard", "tissue", "magazine", "pizza box", "toilet tube", "corrugated")
METAL = ("metal", "aluminium", "pop tab", "foil", "aerosol", "scrap")
PLASTIC = ("plastic", "styrofoam", "polypropylene", "foam", "straw", "wrapper", "crisp packet",
           "six pack rings", "squeezable tube", "tupperware", "spread tub", "garbage bag",
           "bottle", "lid", "cup", "container", "film", "glove", "utensils", "blister")


def material(name, supercategory):
    """Order matters: 'Paper cup' is paper and 'Metal lid' is metal before the
    generic plastic keywords get a chance."""
    s = f"{name} {supercategory}".lower()
    if "glass" in s:
        return "glass"
    if any(k in s for k in PAPER):
        return "paper_cardboard"
    if any(k in s for k in METAL) or re.search(r"\bcan\b", s):
        return "metal"
    if "food waste" in s:
        return "organic"
    if any(k in s for k in PLASTIC):
        return "plastic"
    return "other"


class Throttle:
    """Shared by all workers: when the host answers 429 every worker waits, and
    the wait doubles until requests go through again. Gives up once the total
    time spent waiting passes the budget, so a blocked address fails fast."""

    def __init__(self, budget):
        self.lock = threading.Lock()
        self.resume_at = 0.0
        self.delay = 5.0
        self.waited = 0.0
        self.budget = budget
        self.gave_up = False

    def wait(self):
        while True:
            with self.lock:
                left = self.resume_at - time.monotonic()
            if left <= 0:
                return
            time.sleep(left)

    def hit(self, retry_after):
        with self.lock:
            now = time.monotonic()
            if now < self.resume_at:  # another worker already backed off
                return
            pause = retry_after or self.delay
            self.resume_at = now + pause
            self.waited += pause
            self.delay = min(self.delay * 2, 120.0)
            if self.waited > self.budget:
                self.gave_up = True
            print(f"  host is throttling, pausing {pause:.0f}s", flush=True)

    def ok(self):
        with self.lock:
            self.delay = 5.0


def download(session, throttle, url, attempts=5):
    """Returns (bytes, None) or (None, reason)."""
    reason = "no response"
    for attempt in range(attempts):
        if throttle.gave_up:
            return None, "http 429"
        throttle.wait()
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                throttle.ok()
                return r.content, None
            reason = f"http {r.status_code}"
            if r.status_code == 429:
                after = r.headers.get("Retry-After", "")
                throttle.hit(float(after) if after.isdigit() else None)
                continue
            if r.status_code not in (500, 502, 503, 504):
                break
        except requests.RequestException as e:
            reason = type(e).__name__
        time.sleep(2.0 * (attempt + 1))
    return None, reason


def tile_starts(length, size, step):
    """Mirrors tile_starts() in python/baseline.py and engine/src/main.cpp."""
    if length <= size:
        return [0]
    starts = list(range(0, length - size, step))
    starts.append(length - size)
    return starts


def write_tiles(image_path, label_dir, boxes, size, overlap, background, rng):
    """Cuts one saved image into tiles and writes a label file for each tile
    that is kept. `boxes` are (cls, x1, y1, x2, y2) as fractions of the image.
    A box goes to a tile when at least 40% of it is inside; tiles holding only
    smaller fragments are dropped, and a share of the empty ones is kept as
    background. Returns the number of tiles kept."""
    kept = 0
    with Image.open(image_path) as im:
        W, H = im.size
        if W <= size and H <= size:
            return 0
        step = max(1, int(size * (1.0 - overlap)))
        tw, th = min(size, W), min(size, H)
        px = [(c, x1 * W, y1 * H, x2 * W, y2 * H) for c, x1, y1, x2, y2 in boxes]
        index = 0
        for ty in tile_starts(H, size, step):
            for tx in tile_starts(W, size, step):
                stem = f"{image_path.stem}_t{index:02d}"
                index += 1
                lines, fragment = [], False
                for c, x1, y1, x2, y2 in px:
                    ix1, iy1 = max(x1, tx), max(y1, ty)
                    ix2, iy2 = min(x2, tx + tw), min(y2, ty + th)
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    visible = (ix2 - ix1) * (iy2 - iy1) / ((x2 - x1) * (y2 - y1))
                    if visible < 0.4 or ix2 - ix1 < 4 or iy2 - iy1 < 4:
                        fragment = True
                        continue
                    lines.append(f"{c} {((ix1 + ix2) / 2 - tx) / tw:.6f} {((iy1 + iy2) / 2 - ty) / th:.6f} "
                                 f"{(ix2 - ix1) / tw:.6f} {(iy2 - iy1) / th:.6f}")
                # Drawn for every tile so the choice is the same on a resumed run.
                keep_empty = rng.random() < background
                if not lines and (fragment or not keep_empty):
                    continue
                tile_path = image_path.with_name(stem + ".jpg")
                if not tile_path.exists():
                    im.crop((tx, ty, tx + tw, ty + th)).save(tile_path, "JPEG", quality=90)
                (label_dir / f"{stem}.txt").write_text("\n".join(lines))
                kept += 1
    return kept


def fetch(session, throttle, img, out_path, max_side):
    """Saves one image with its pixels in the orientation the labels assume.
    Returns None on success, otherwise a short reason."""
    if out_path.exists():
        return None
    url = img.get("flickr_url") or img.get("flickr_640_url")
    if not url:
        return "no url"
    data, reason = download(session, throttle, url)
    if data is None:
        return reason
    try:
        raw = Image.open(io.BytesIO(data))
        want = img["width"] / img["height"]
        chosen = None
        # Some photos carry an EXIF rotation; keep whichever version has the
        # aspect ratio the annotations were drawn on.
        for cand in (raw, ImageOps.exif_transpose(raw)):
            if abs(cand.width / cand.height - want) < 0.02:
                chosen = cand
                break
        if chosen is None:
            return "aspect ratio does not match the annotations"
        chosen = chosen.convert("RGB")
        if max(chosen.size) > max_side:
            chosen.thumbnail((max_side, max_side), Image.BILINEAR)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        chosen.save(out_path, "JPEG", quality=90)
        return None
    except Exception as e:
        return f"bad image ({type(e).__name__})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/taco")
    p.add_argument("--max-side", type=int, default=1920)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tile", type=int, default=640, help="tile size, 0 trains on whole images only")
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--background", type=float, default=0.05,
                   help="share of empty tiles kept as background")
    p.add_argument("--max-wait", type=float, default=1800,
                   help="seconds to spend waiting on a throttling host before giving up")
    p.add_argument("--min-fraction", type=float, default=0.9,
                   help="fail when fewer than this share of the images was downloaded")
    args = p.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ann_path = out / "annotations.json"
    if not ann_path.exists():
        ann_path.write_bytes(requests.get(ANN_URL, timeout=120).content)
    coco = json.loads(ann_path.read_text())

    cat_to_cls = {}
    print("category mapping:")
    for c in coco["categories"]:
        m = material(c["name"], c.get("supercategory", ""))
        cat_to_cls[c["id"]] = CLASSES.index(m)
        print(f"  {c['name']:<32} -> {m}")

    by_image = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)

    images = sorted(coco["images"], key=lambda im: im["id"])
    random.Random(args.seed).shuffle(images)
    n_val = int(len(images) * args.val_fraction)
    split = {im["id"]: ("val" if i < n_val else "train") for i, im in enumerate(images)}

    session = requests.Session()
    session.headers["User-Agent"] = f"taco-dataset-download python-requests/{requests.__version__}"
    throttle = Throttle(args.max_wait)

    def image_path(im):
        return out / "images" / split[im["id"]] / f"{im['id']:06d}.jpg"

    def job(im):
        return im, fetch(session, throttle, im, image_path(im), args.max_side)

    print(f"downloading {len(images)} images ...")
    kept = {"train": 0, "val": 0}
    tiles = {"train": 0, "val": 0}
    counts = [0] * len(CLASSES)
    failures = Counter()
    with ThreadPoolExecutor(args.workers) as pool:
        for done, (im, reason) in enumerate(pool.map(job, images), 1):
            if done % 100 == 0:
                print(f"  {done}/{len(images)}")
            if reason is not None:
                failures[reason] += 1
                continue
            W, H = im["width"], im["height"]
            lines = []
            boxes = []
            for a in by_image.get(im["id"], []):
                x, y, w, h = a["bbox"]
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(W), x + w), min(float(H), y + h)
                if x2 - x1 <= 1 or y2 - y1 <= 1:
                    continue
                cls = cat_to_cls[a["category_id"]]
                counts[cls] += 1
                boxes.append((cls, x1 / W, y1 / H, x2 / W, y2 / H))
                lines.append(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                             f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            label = out / "labels" / split[im["id"]] / f"{im['id']:06d}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(lines))
            kept[split[im["id"]]] += 1
            if args.tile > 0:
                tiles[split[im["id"]]] += write_tiles(
                    image_path(im), label.parent, boxes, args.tile, args.overlap,
                    args.background, random.Random(args.seed * 1000003 + im["id"]))

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")

    print(f"images: {kept['train']} train, {kept['val']} val, "
          f"{len(images) - kept['train'] - kept['val']} failed to download")
    for reason, n in failures.most_common():
        print(f"  failed: {n:<5}{reason}")
    print("boxes per class:")
    for n, c in zip(CLASSES, counts):
        print(f"  {n:<18}{c}")
    if args.tile > 0:
        print(f"tiles: {tiles['train']} train, {tiles['val']} val")
    print(f"wrote {out / 'data.yaml'}")

    got = kept["train"] + kept["val"]
    if got < args.min_fraction * len(images):
        sys.exit(f"only {got} of {len(images)} images were downloaded. Run this again to "
                 f"fetch the rest (finished images are skipped), or pass --min-fraction 0 "
                 f"to train on what is there.")


if __name__ == "__main__":
    main()
