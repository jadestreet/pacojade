# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Placeholder interface for future frozen-DINO feature sampling."""


class DINOFeatureExtractor:
    """
    Future interface: image -> DINO patch feature map -> sample(points_xy).

    DINO is intentionally not implemented in the FPN-first phase. Keeping this
    placeholder makes CLI/API branching explicit without adding dependencies.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DINO feature extraction is reserved for a later ablation. "
            "Use --feature-source fpn for the current implementation."
        )
