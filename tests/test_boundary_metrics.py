# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Unit tests for boundary-sensitive metrics (run with pytest)."""
import numpy as np

from paco.evaluation.boundary_metrics import (
    boundary_f_measure,
    mask_iou,
    match_and_score,
)


def _square(h=64, w=64, top=10, left=10, size=40):
    m = np.zeros((h, w), dtype=bool)
    m[top : top + size, left : left + size] = True
    return m


def test_iou_identical_and_disjoint():
    a = _square()
    assert mask_iou(a, a) == 1.0
    b = _square(left=10 + 50)  # shifted fully off the original (disjoint)
    # disjoint within a wider canvas
    big_a = np.zeros((64, 200), dtype=bool)
    big_a[10:50, 10:50] = True
    big_b = np.zeros((64, 200), dtype=bool)
    big_b[10:50, 120:160] = True
    assert mask_iou(big_a, big_b) == 0.0


def test_iou_both_empty_is_one():
    z = np.zeros((32, 32), dtype=bool)
    assert mask_iou(z, z) == 1.0


def test_bf_identical_is_one():
    a = _square()
    assert boundary_f_measure(a, a, tol=2.0) == 1.0


def test_bf_small_shift_high_large_shift_low():
    a = _square()
    near = _square(top=11, left=11)  # 1px shift -> within 2px tolerance
    far = _square(top=10, left=10 + 12)  # 12px shift -> outside tolerance
    f_near = boundary_f_measure(a, near, tol=2.0)
    f_far = boundary_f_measure(a, far, tol=2.0)
    assert f_near > 0.9, f_near
    assert f_far < f_near
    assert f_far < 0.5, f_far


def test_bf_empty_vs_nonempty_is_zero():
    a = _square()
    z = np.zeros_like(a)
    assert boundary_f_measure(a, z, tol=2.0) == 0.0


def test_match_and_score_greedy():
    a = _square(left=10)
    a2 = _square(left=10)  # identical GT for a
    pred = [
        {"mask": a, "category_id": 5, "score": 0.9},
        {"mask": _square(left=120, w=200), "category_id": 5, "score": 0.5},
    ]
    # widen canvas for the second pred/gt
    wide = np.zeros((64, 200), dtype=bool)
    wide[10:50, 10:50] = True
    pred[0]["mask"] = wide
    pred[1]["mask"] = np.zeros((64, 200), dtype=bool)
    pred[1]["mask"][10:50, 120:160] = True
    gt = [
        {"mask": pred[0]["mask"].copy(), "category_id": 5},
        {"mask": pred[1]["mask"].copy(), "category_id": 5},
    ]
    res = match_and_score(pred, gt, iou_thresh=0.5, tol=2.0)
    assert len(res) == 2
    assert all(r["iou"] == 1.0 for r in res)
    assert all(r["bf"] == 1.0 for r in res)


def test_match_below_threshold_not_matched():
    wide_p = np.zeros((64, 200), dtype=bool)
    wide_p[10:50, 10:50] = True
    wide_g = np.zeros((64, 200), dtype=bool)
    wide_g[10:50, 120:160] = True  # disjoint -> IoU 0
    res = match_and_score(
        [{"mask": wide_p, "category_id": 1, "score": 1.0}],
        [{"mask": wide_g, "category_id": 1}],
        iou_thresh=0.5,
    )
    assert res == []
