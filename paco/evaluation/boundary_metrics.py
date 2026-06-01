# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Boundary-sensitive part-segmentation metrics that the PACO/LVIS evaluator does
not provide: per-instance mask IoU and the standard boundary F-measure
(2px tolerance), plus greedy pred<->GT matching.

These are pure functions over boolean HxW numpy masks so they can be unit tested
without any model or dataset.
"""
from typing import Dict, List

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean masks."""
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0  # both empty
    return float(inter) / float(union)


def _boundary(mask: np.ndarray) -> np.ndarray:
    """One-pixel inner boundary of a boolean mask (mask XOR erosion(mask))."""
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask)
    eroded = binary_erosion(mask, border_value=0)
    return mask & ~eroded


def boundary_f_measure(pred: np.ndarray, gt: np.ndarray, tol: float = 2.0) -> float:
    """
    Standard boundary F-measure (Csurka et al. / DAVIS), with a pixel tolerance.

    precision = fraction of predicted-boundary pixels within `tol` of a GT
    boundary pixel; recall = the symmetric quantity; F = harmonic mean.
    """
    pb = _boundary(pred)
    gb = _boundary(gt)
    pb_n = int(pb.sum())
    gb_n = int(gb.sum())

    if pb_n == 0 and gb_n == 0:
        return 1.0  # both masks empty -> perfect agreement
    if pb_n == 0 or gb_n == 0:
        return 0.0

    # distance_transform_edt gives distance to nearest zero; we want distance to
    # nearest boundary pixel, so transform the complement of the boundary map.
    dt_gt = distance_transform_edt(~gb)
    dt_pred = distance_transform_edt(~pb)

    precision = float((dt_gt[pb] <= tol).mean())
    recall = float((dt_pred[gb] <= tol).mean())
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def match_and_score(
    preds: List[Dict],
    gts: List[Dict],
    iou_thresh: float = 0.5,
    tol: float = 2.0,
) -> List[Dict]:
    """
    Greedy pred<->GT matching within a single (image, category) group.

    Each pred/gt dict must have a boolean "mask" (HxW) and "category_id". Preds
    should also carry "score". Returns one record per matched pair with iou + bf,
    plus the counts needed to compute recall@thresh.

    Matching is greedy by descending pred score, taking the highest-IoU unused GT
    above `iou_thresh`.
    """
    results = []
    by_cat_gt: Dict[int, List[Dict]] = {}
    for g in gts:
        by_cat_gt.setdefault(g["category_id"], []).append(g)

    used = {cat: set() for cat in by_cat_gt}
    for p in sorted(preds, key=lambda x: x.get("score", 0.0), reverse=True):
        cat = p["category_id"]
        cands = by_cat_gt.get(cat, [])
        best_j, best_iou = -1, iou_thresh
        for j, g in enumerate(cands):
            if j in used[cat]:
                continue
            iou = mask_iou(p["mask"], g["mask"])
            if iou >= best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            used[cat].add(best_j)
            g = cands[best_j]
            results.append(
                {
                    "category_id": cat,
                    "iou": best_iou,
                    "bf": boundary_f_measure(p["mask"], g["mask"], tol=tol),
                    "matched": True,
                }
            )
    return results
