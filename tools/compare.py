"""Compares the C++ engine's JSON output with the Python baseline's.

Reports how many detections agree (same class, IoU >= 0.9) and the per-stage
speedup. Coverage is left out of the totals because the baseline skips it.
"""
import argparse
import json


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("engine_json")
    p.add_argument("baseline_json")
    p.add_argument("--iou", type=float, default=0.9)
    args = p.parse_args()

    with open(args.engine_json) as f:
        eng = json.load(f)
    with open(args.baseline_json) as f:
        base = json.load(f)

    unmatched = list(base["detections"])
    matched = 0
    for d in eng["detections"]:
        best, best_iou = None, args.iou
        for i, o in enumerate(unmatched):
            if o["cls"] != d["cls"]:
                continue
            v = iou(d["box"], o["box"])
            if v >= best_iou:
                best, best_iou = i, v
        if best is not None:
            unmatched.pop(best)
            matched += 1

    n_eng, n_base = len(eng["detections"]), len(base["detections"])
    print(f"detections: engine {n_eng}, baseline {n_base}, matching {matched}")

    et, bt = eng["timing_ms"], base["timing_ms"]
    rows = [
        ("preprocess", et["upload"] + et["preprocess"], bt["preprocess"]),
        ("infer", et["infer"], bt["infer"]),
        ("postprocess", et["decode"] + et["nms"], bt["postprocess"]),
    ]
    rows.append(("total", sum(r[1] for r in rows), sum(r[2] for r in rows)))
    print(f"{'stage':<12}{'engine ms':>12}{'baseline ms':>14}{'speedup':>10}")
    for name, e, b in rows:
        speedup = f"{b / e:.2f}x" if e > 0 else "-"
        print(f"{name:<12}{e:>12.3f}{b:>14.3f}{speedup:>10}")


if __name__ == "__main__":
    main()
