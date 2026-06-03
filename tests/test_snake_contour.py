import numpy as np

from paco.snake.contour import (
    align_contour_to_reference,
    contour_to_mask,
    mask_to_largest_contour,
    mask_to_resampled_contour,
    resample_closed_contour,
)


def test_mask_to_contour_and_resample_rectangle():
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 8:22] = True

    contour = mask_to_largest_contour(mask)
    assert contour is not None
    assert contour.shape[1] == 2

    sampled = resample_closed_contour(contour, 16)
    assert sampled is not None
    assert sampled.shape == (16, 2)
    assert sampled[:, 0].min() >= 7
    assert sampled[:, 0].max() <= 22


def test_empty_and_tiny_masks_are_invalid():
    assert mask_to_largest_contour(np.zeros((8, 8), dtype=bool)) is None

    tiny = np.zeros((8, 8), dtype=bool)
    tiny[3, 3] = True
    assert mask_to_resampled_contour(tiny, 8) is None


def test_contour_to_mask_round_trip():
    points = np.array([[5, 5], [14, 5], [14, 14], [5, 14]], dtype=np.float32)
    mask = contour_to_mask(points, 20, 20)

    assert mask.dtype == bool
    assert mask.sum() > 0

    sampled = mask_to_resampled_contour(mask, 12)
    assert sampled is not None
    mask2 = contour_to_mask(sampled, 20, 20)
    assert mask2.sum() > 0
    assert np.logical_and(mask, mask2).sum() > 0


def test_align_contour_handles_cyclic_shift_and_reverse():
    ref = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    shifted_reversed = np.array([[10, 10], [10, 0], [0, 0], [0, 10]], dtype=np.float32)

    aligned = align_contour_to_reference(shifted_reversed, ref)
    assert aligned is not None
    np.testing.assert_allclose(aligned, ref)
