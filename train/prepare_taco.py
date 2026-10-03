"""Downloads the TACO litter dataset and converts it to YOLO detection format.

TACO's 60 fine-grained categories are folded into 6 material classes. The
mapping is printed so it can be checked before training.
"""
import argparse
import io
import json
import random
import re
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


def fetch(img, out_path, max_side):
    """Saves one image with its pixels in the orientation the labels assume."""
    if out_path.exists():
        return True
    url = img.get("flickr_url") or img.get("flickr_640_url")
    if not url:
        return False
    try:
        data = requests.get(url, timeout=60).content
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
            return False
        chosen = chosen.convert("RGB")
        if max(chosen.size) > max_side:
            chosen.thumbnail((max_side, max_side), Image.BILINEAR)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        chosen.save(out_path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/taco")
    p.add_argument("--max-side", type=int, default=1920)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
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

    def job(im):
        dest = out / "images" / split[im["id"]] / f"{im['id']:06d}.jpg"
        return im, fetch(im, dest, args.max_side)

    print(f"downloading {len(images)} images ...")
    kept = {"train": 0, "val": 0}
    counts = [0] * len(CLASSES)
    with ThreadPoolExecutor(args.workers) as pool:
        for done, (im, ok) in enumerate(pool.map(job, images), 1):
            if done % 100 == 0:
                print(f"  {done}/{len(images)}")
            if not ok:
                continue
            W, H = im["width"], im["height"]
            lines = []
            for a in by_image.get(im["id"], []):
                x, y, w, h = a["bbox"]
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(W), x + w), min(float(H), y + h)
                if x2 - x1 <= 1 or y2 - y1 <= 1:
                    continue
                cls = cat_to_cls[a["category_id"]]
                counts[cls] += 1
                lines.append(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                             f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            label = out / "labels" / split[im["id"]] / f"{im['id']:06d}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(lines))
            kept[split[im["id"]]] += 1

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")

    print(f"images: {kept['train']} train, {kept['val']} val, "
          f"{len(images) - kept['train'] - kept['val']} failed to download")
    print("boxes per class:")
    for n, c in zip(CLASSES, counts):
        print(f"  {n:<18}{c}")
    print(f"wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
