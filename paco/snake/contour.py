# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Contour utilities for Deep-Snake-style mask refinement.

The public functions operate in original image pixel coordinates with points in
``(x, y)`` order. Invalid masks return ``None`` rather than fabricating a contour;
callers should skip those instances for training/refinement.
"""
from typing import Optional, Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_util


def mask_to_largest_contour(
    mask: np.ndarray,
    min_points: int = 3,
    min_area: float = 1.0,
) -> Optional[np.ndarray]:
    """Return the largest external contour as an ``Nx2`` float32 ``(x, y)`` array."""
    mask_u8 = np.ascontiguousarray(mask.astype(np.uint8))
    if mask_u8.ndim != 2 or not mask_u8.any():
        return None

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None

    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < min_points:
        return None
    return pts


def resample_closed_contour(points_xy: np.ndarray, num_vertices: int) -> Optional[np.ndarray]:
    """Uniformly resample a closed contour to exactly ``num_vertices`` points."""
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or num_vertices < 3:
        return None

    closed = np.concatenate([pts, pts[:1]], axis=0)
    seg_len = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(seg_len.sum())
    if total <= 1e-6:
        return None

    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    samples = np.linspace(0.0, total, num_vertices, endpoint=False, dtype=np.float32)
    out = np.empty((num_vertices, 2), dtype=np.float32)

    seg_idx = np.searchsorted(cumulative, samples, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(seg_len) - 1)
    denom = np.maximum(seg_len[seg_idx], 1e-6)
    alpha = ((samples - cumulative[seg_idx]) / denom).reshape(-1, 1)
    out[:] = closed[seg_idx] * (1.0 - alpha) + closed[seg_idx + 1] * alpha
    return out


def mask_to_resampled_contour(mask: np.ndarray, num_vertices: int) -> Optional[np.ndarray]:
    """Convert a mask to its largest ordered contour and resample it to fixed K."""
    contour = mask_to_largest_contour(mask)
    if contour is None:
        return None
    return resample_closed_contour(contour, num_vertices)


def align_contour_to_reference(
    contour_xy: np.ndarray,
    reference_xy: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Cyclically shift/reverse ``contour_xy`` to best align with ``reference_xy``.

    This gives a stable supervised target for per-vertex offset regression.
    """
    contour = np.asarray(contour_xy, dtype=np.float32)
    ref = np.asarray(reference_xy, dtype=np.float32)
    if contour.shape != ref.shape or contour.ndim != 2 or contour.shape[1] != 2:
        return None

    best = None
    best_err = float("inf")
    for candidate in (contour, contour[::-1].copy()):
        for shift in range(len(candidate)):
            shifted = np.roll(candidate, shift, axis=0)
            err = float(np.mean(np.sum((shifted - ref) ** 2, axis=1)))
            if err < best_err:
                best_err = err
                best = shifted
    return best.astype(np.float32)


def contour_to_mask(
    points_xy: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize a closed contour into a boolean ``height x width`` mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return mask.astype(bool)

    pts_i = np.rint(pts).astype(np.int32)
    pts_i[:, 0] = np.clip(pts_i[:, 0], 0, width - 1)
    pts_i[:, 1] = np.clip(pts_i[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [pts_i.reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def encode_binary_mask(mask: np.ndarray) -> dict:
    """Encode a boolean mask as JSON-serializable COCO RLE."""
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def normalize_points(points_xy: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Normalize pixel points to roughly ``[-1, 1]`` using ``(height, width)``."""
    h, w = image_size
    pts = np.asarray(points_xy, dtype=np.float32).copy()
    pts[:, 0] = (pts[:, 0] / max(w - 1, 1)) * 2.0 - 1.0
    pts[:, 1] = (pts[:, 1] / max(h - 1, 1)) * 2.0 - 1.0
    return pts


def clamp_points(points_xy: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Clamp pixel points to image bounds using ``(height, width)``."""
    h, w = image_size
    pts = np.asarray(points_xy, dtype=np.float32).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, max(w - 1, 0))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(h - 1, 0))
    return pts
