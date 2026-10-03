"""Resolves overlapping detections from different tiles.

Mirrors GpuPipeline::nms() in engine/src/gpu_pipeline.cu: boxes are taken in
score order, and each kept box absorbs every lower-ranked box of the same class
that overlaps it past the threshold. Overlap is measured on the original boxes,
not on the grown one.
"""
import numpy as np


def resolve_overlaps(boxes, scores, classes, threshold=0.5, use_ios=True, merge=True,
                     class_agnostic=False):
    """boxes is [n, 4] as x1, y1, x2, y2. Returns (boxes, scores, classes) of
    the kept detections, highest score first."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32)
    classes = np.asarray(classes)
    n = len(scores)
    if n == 0:
        return boxes, scores, classes

    # Same tie-break as the engine: score, then x1, then y1.
    order = np.lexsort((boxes[:, 1], boxes[:, 0], -scores))
    boxes, scores, classes = boxes[order], scores[order], classes[order]

    iw = np.minimum(boxes[:, None, 2], boxes[None, :, 2]) - np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    ih = np.minimum(boxes[:, None, 3], boxes[None, :, 3]) - np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if use_ios:
        denom = np.minimum(area[:, None], area[None, :])
    else:
        denom = area[:, None] + area[None, :] - inter
    match = inter / np.maximum(denom, 1e-9) > threshold
    if not class_agnostic:
        match &= classes[:, None] == classes[None, :]

    removed = np.zeros(n, dtype=bool)
    keep, kept_boxes = [], []
    for i in range(n):
        if removed[i]:
            continue
        box = boxes[i].copy()
        later = match[i, i + 1:]
        if merge:
            fresh = np.nonzero(later & ~removed[i + 1:])[0] + i + 1
            if len(fresh):
                box[:2] = np.minimum(box[:2], boxes[fresh, :2].min(axis=0))
                box[2:] = np.maximum(box[2:], boxes[fresh, 2:].max(axis=0))
        removed[i + 1:] |= later
        keep.append(i)
        kept_boxes.append(box)
    return np.stack(kept_boxes), scores[keep], classes[keep]
