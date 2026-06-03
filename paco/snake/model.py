# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Lightweight circular-convolution Snake refinement network."""
from typing import Callable, Iterable, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class CircularConv1d(nn.Module):
    """1D convolution over a closed contour ring."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd for symmetric circular padding")
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, self.pad), mode="circular"))


class SnakeRefiner(nn.Module):
    """
    Predict per-vertex 2D offsets from contour coordinates and sampled features.

    Inputs use shape ``B x K x 2`` for normalized coordinates and ``B x K x C``
    for sampled features. Output shape is ``B x K x 2`` in pixel-offset units.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        max_offset: float = 16.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.feature_dim = feature_dim
        self.max_offset = float(max_offset)

        layers = [CircularConv1d(feature_dim + 2, hidden_dim, kernel_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [CircularConv1d(hidden_dim, hidden_dim, kernel_size), nn.ReLU()]
        self.body = nn.Sequential(*layers)
        self.head = CircularConv1d(hidden_dim, 2, kernel_size)

    def forward(self, vertices_norm: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if vertices_norm.dim() == 2:
            vertices_norm = vertices_norm.unsqueeze(0)
        if features.dim() == 2:
            features = features.unsqueeze(0)
        if vertices_norm.shape[:2] != features.shape[:2]:
            raise ValueError("vertices_norm and features must share B x K dimensions")
        if vertices_norm.shape[-1] != 2:
            raise ValueError("vertices_norm must have last dimension 2")
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.feature_dim}, got {features.shape[-1]}"
            )

        x = torch.cat([vertices_norm, features], dim=-1).permute(0, 2, 1)
        x = self.body(x)
        offsets = self.head(x).permute(0, 2, 1)
        return torch.tanh(offsets) * self.max_offset


def normalize_vertices_tensor(
    vertices_xy: torch.Tensor,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """Normalize pixel vertices to ``[-1, 1]`` using ``(height, width)``."""
    h, w = image_size
    out = vertices_xy.clone().float()
    out[..., 0] = (out[..., 0] / max(w - 1, 1)) * 2.0 - 1.0
    out[..., 1] = (out[..., 1] / max(h - 1, 1)) * 2.0 - 1.0
    return out


def refine_vertices(
    model: SnakeRefiner,
    vertices_xy: torch.Tensor,
    image_size: Tuple[int, int],
    feature_sampler: Callable[[torch.Tensor], torch.Tensor],
    steps: int = 1,
) -> torch.Tensor:
    """
    Iteratively refine pixel-space vertices.

    ``feature_sampler`` receives a ``K x 2`` pixel-coordinate tensor and returns
    sampled features as ``K x C``.
    """
    if vertices_xy.dim() != 2 or vertices_xy.shape[-1] != 2:
        raise ValueError("vertices_xy must be K x 2")
    h, w = image_size
    current = vertices_xy.float()
    for _ in range(steps):
        features = feature_sampler(current)
        vertices_norm = normalize_vertices_tensor(current, image_size)
        offsets = model(vertices_norm, features)[0]
        current = current + offsets
        current[:, 0].clamp_(0, max(w - 1, 0))
        current[:, 1].clamp_(0, max(h - 1, 0))
    return current


def feature_dim_for_levels(level_dims: Sequence[int]) -> int:
    """Return concatenated feature dimension for one or more sampled levels."""
    return int(sum(level_dims))
