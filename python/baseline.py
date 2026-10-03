"""Plain Python version of the tiled pipeline, used as the benchmark baseline.

Tiling, preprocessing and box merging run on the CPU with NumPy/OpenCV, the
model and output decoding run on the GPU through PyTorch. It loads the same TorchScript file as the C++
engine, so detections should match and only the speed should differ.
"""
import argparse
import json
import math
import time

import cv2
import numpy as np
import torch

from merge import resolve_overlaps

NMS_LIMIT = 4096


def tile_starts(length, size, step):
    if length <= size:
        return [0]
    starts = list(range(0, length - size, step))
    starts.append(length - size)
    return starts


def make_tiles(W, H, S, overlap, add_full):
    """Mirrors make_tiles() in engine/src/main.cpp."""
    step = max(1, int(S * (1.0 - overlap)))
    tiles = []
    for y in tile_starts(H, S, step):
        for x in tile_starts(W, S, step):
            tiles.append(dict(src_x=x, src_y=y, scale=1.0, pad_x=0, pad_y=0,
                              cw=min(S, W), ch=min(S, H)))
    if add_full and (W > S or H > S):
        scale = max(W, H) / S
        cw, ch = W / scale, H / scale
        tiles.append(dict(src_x=0, src_y=0, scale=scale,
                          pad_x=math.floor((S - cw) / 2), pad_y=math.floor((S - ch) / 2),
                          cw=min(S, math.ceil(cw)), ch=min(S, math.ceil(ch))))
    return tiles


def preprocess(frame, tiles, S, batch):
    out = np.full((batch, S, S, 3), 114, dtype=np.uint8)
    for i, t in enumerate(tiles):
        if t["scale"] == 1.0:
            crop = frame[t["src_y"]:t["src_y"] + t["ch"], t["src_x"]:t["src_x"] + t["cw"]]
        else:
            crop = cv2.resize(frame, (t["cw"], t["ch"]), interpolation=cv2.INTER_LINEAR)
        out[i, t["pad_y"]:t["pad_y"] + t["ch"], t["pad_x"]:t["pad_x"] + t["cw"]] = crop
    chw = np.ascontiguousarray(out[..., ::-1].transpose(0, 3, 1, 2))
    return torch.from_numpy(chw).cuda().float().div_(255.0)


def decode(pred, tiles, conf, W, H):
    n = len(tiles)
    pred = pred[:n].float()
    scores, cls = pred[:, 4:, :].max(dim=1)
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]

    def col(key):
        return torch.tensor([t[key] for t in tiles], dtype=torch.float32, device=pred.device)[:, None]

    scale, pad_x, pad_y, src_x, src_y = (col(k) for k in ("scale", "pad_x", "pad_y", "src_x", "src_y"))
    x1 = ((cx - w / 2 - pad_x) * scale + src_x).clamp(0, W)
    y1 = ((cy - h / 2 - pad_y) * scale + src_y).clamp(0, H)
    x2 = ((cx + w / 2 - pad_x) * scale + src_x).clamp(0, W)
    y2 = ((cy + h / 2 - pad_y) * scale + src_y).clamp(0, H)
    keep = (scores >= conf) & (x2 - x1 >= 1) & (y2 - y1 >= 1)
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)[keep]
    return boxes, scores[keep], cls[keep]


def run(frame, model, args):
    H, W = frame.shape[:2]
    S, B = args.size, args.batch
    t = dict(preprocess=0.0, infer=0.0, postprocess=0.0)
    tiles = make_tiles(W, H, S, args.overlap, not args.no_full)

    boxes, scores, classes = [], [], []
    for first in range(0, len(tiles), B):
        chunk = tiles[first:first + B]
        t0 = time.perf_counter()
        batch = preprocess(frame, chunk, S, B)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        pred = model(batch)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        b, s, c = decode(pred, chunk, args.conf, W, H)
        boxes.append(b)
        scores.append(s)
        classes.append(c)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        t["preprocess"] += (t1 - t0) * 1e3
        t["infer"] += (t2 - t1) * 1e3
        t["postprocess"] += (t3 - t2) * 1e3

    t0 = time.perf_counter()
    boxes, scores, classes = torch.cat(boxes), torch.cat(scores), torch.cat(classes)
    if scores.numel() > NMS_LIMIT:
        top = scores.topk(NMS_LIMIT).indices
        boxes, scores, classes = boxes[top], scores[top], classes[top]
    boxes, scores, classes = resolve_overlaps(
        boxes.cpu().numpy(), scores.cpu().numpy(), classes.cpu().numpy(),
        threshold=args.match_thr, use_ios=args.metric == "ios", merge=not args.no_merge)
    t["postprocess"] += (time.perf_counter() - t0) * 1e3

    dets = [dict(cls=int(c), score=float(s), box=[float(v) for v in b])
            for b, s, c in zip(boxes, scores, classes)]
    return dets, t, len(tiles)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--match-thr", type=float, default=0.5)
    p.add_argument("--metric", choices=("ios", "iou"), default="ios")
    p.add_argument("--no-merge", action="store_true")
    p.add_argument("--no-full", action="store_true")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--json")
    args = p.parse_args()

    frame = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"cannot read image {args.input}")
    model = torch.jit.load(args.model, map_location="cuda").eval()

    warmup = 2 if args.repeat > 3 else 0
    total = dict(preprocess=0.0, infer=0.0, postprocess=0.0)
    with torch.no_grad():
        for i in range(args.repeat):
            dets, t, tiles = run(frame, model, args)
            if i >= warmup:
                for k in total:
                    total[k] += t[k]
    measured = args.repeat - warmup
    total = {k: v / measured for k, v in total.items()}

    all_ms = sum(total.values())
    print(f"{args.input}: {frame.shape[1]}x{frame.shape[0]}, {tiles} tiles, {len(dets)} objects")
    print(f"mean over {measured} run(s), ms per frame")
    for k, v in total.items():
        print(f"  {k:<12}{v:8.3f}")
    print(f"  {'total':<12}{all_ms:8.3f}  ({1000.0 / all_ms:.1f} fps)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(image=args.input, width=frame.shape[1], height=frame.shape[0],
                           tiles=tiles, timing_ms=total, detections=dets), f, indent=2)


if __name__ == "__main__":
    main()
