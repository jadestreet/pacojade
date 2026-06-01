# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
FPN feature + predicted-mask extraction interface for the Deep Snake refinement
module (Phase 4).

This wraps a *frozen* PACO detector and exposes, per image:
  * the predicted part/object ``Instances`` (boxes, classes, scores, masks) in
    the original image coordinate frame, and
  * ``sample(points_xy, level)`` — bilinearly sampled FPN features (P2..P5) at
    arbitrary 2D points given in original-image pixel coordinates.

This is the input contract the snake consumes: for each predicted part instance
we have its mask -> contour (contour conversion is future work), and a way to
read backbone features at any contour vertex location, with correct stride
scaling between the original image, the network-input frame, and each FPN level.

Coordinate frames
-----------------
read_image -> original (H0, W0)
  --ResizeShortestEdge(1024,1024)--> network frame (Hn, Wn), scale = (Wn/W0, Hn/H0)
  --pad to /32--> padded tensor, FPN level L has stride s_L, size ~ padded/s_L
Predicted instances are postprocessed back to the original frame; ``sample``
maps original -> network -> feature-grid internally.
"""
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate, LazyConfig
from detectron2.data.detection_utils import read_image
from detectron2.modeling.postprocessing import detector_postprocess


class FPNFeatureExtractor:
    FPN_STRIDES = {"p2": 4, "p3": 8, "p4": 16, "p5": 32}

    def __init__(
        self,
        config_file: str,
        checkpoint: str,
        device: str = "cuda",
        image_size: int = 1024,
    ):
        cfg = LazyConfig.load(config_file)
        self.model = instantiate(cfg.model)
        self.model.to(device).eval()
        DetectionCheckpointer(self.model).load(checkpoint)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.input_format = self.model.input_format  # e.g. "RGB"/"BGR"
        self.aug = T.ResizeShortestEdge(
            short_edge_length=image_size, max_size=image_size
        )

    @torch.no_grad()
    def run(self, image_path: str) -> Dict:
        """Run the frozen detector on one image; capture FPN features + instances."""
        img = read_image(image_path, format=self.input_format)  # H0 x W0 x C
        h0, w0 = img.shape[:2]

        aug_input = T.AugInput(img)
        self.aug(aug_input)
        image = aug_input.image  # resized (Hn x Wn x C)
        hn, wn = image.shape[:2]
        image_t = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        ).to(self.device)

        inputs = [{"image": image_t, "height": h0, "width": w0}]
        images = self.model.preprocess_image(inputs)
        features = self.model.backbone(images.tensor)
        proposals, _ = self.model.proposal_generator(images, features, None)
        results, _ = self.model.roi_heads(images, features, proposals, None)
        instances = detector_postprocess(results[0], h0, w0)  # original frame

        return {
            "features": features,  # dict level -> (1, C, Hf, Wf), network/padded frame
            "instances": instances.to("cpu"),  # original-image frame
            "scale": (wn / w0, hn / h0),  # original -> network
            "orig_size": (h0, w0),
            "net_size": (hn, wn),
            "padded_size": tuple(images.tensor.shape[-2:]),
            "image_resized": image,  # for visualization in network frame
        }

    def sample(
        self,
        result: Dict,
        points_xy: torch.Tensor,
        level: str = "p2",
    ) -> torch.Tensor:
        """
        Bilinearly sample FPN features at ``points_xy`` (N x 2, original-image
        pixel coords). Returns (N, C).
        """
        feat = result["features"][level]  # (1, C, Hf, Wf)
        _, _, hf, wf = feat.shape
        sx, sy = result["scale"]
        stride = self.FPN_STRIDES[level]

        pts = torch.as_tensor(points_xy, dtype=torch.float32, device=feat.device)
        # original -> network frame -> feature-grid index
        fx = (pts[:, 0] * sx) / stride
        fy = (pts[:, 1] * sy) / stride
        # normalize to [-1, 1] for grid_sample (align_corners=True)
        gx = fx / max(wf - 1, 1) * 2 - 1
        gy = fy / max(hf - 1, 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
        sampled = F.grid_sample(
            feat, grid, mode="bilinear", align_corners=True
        )  # (1, C, N, 1)
        return sampled[0, :, :, 0].t().contiguous()  # (N, C)

    def part_instances(
        self, result: Dict, part_class_ids: Optional[set] = None
    ) -> List[Dict]:
        """
        Convenience: return per-instance dicts {box, class_id, score, mask} in the
        original frame. If ``part_class_ids`` is given, keep only those classes.
        """
        inst = result["instances"]
        out = []
        for i in range(len(inst)):
            cid = int(inst.pred_classes[i])
            if part_class_ids is not None and cid not in part_class_ids:
                continue
            out.append(
                {
                    "box": inst.pred_boxes.tensor[i].numpy(),
                    "class_id": cid,
                    "score": float(inst.scores[i]),
                    "mask": inst.pred_masks[i].numpy().astype(bool),
                }
            )
        return out
