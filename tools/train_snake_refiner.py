#!/usr/bin/env python
"""
Train the FPN-based Snake contour refinement module on PACO-LVIS predictions.

This trains only the lightweight Snake network. The PACO detector/backbone are
loaded through FPNFeatureExtractor, set eval/frozen, and used only for feature
sampling at contour vertices.
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F

from paco.evaluation.boundary_metrics import mask_iou
from paco.snake.contour import (
    align_contour_to_reference,
    mask_to_resampled_contour,
    normalize_points,
)
from paco.snake.fpn_extractor import FPNFeatureExtractor
from paco.snake.model import SnakeRefiner, normalize_vertices_tensor


def decode_to_mask(segm, h, w):
    if isinstance(segm, list):
        rles = mask_util.frPyObjects(segm, h, w)
        rle = mask_util.merge(rles)
    elif isinstance(segm, dict):
        rle = segm
        if isinstance(rle.get("counts"), list):
            rle = mask_util.frPyObjects(rle, h, w)
    else:
        raise ValueError(f"unsupported segmentation type {type(segm)}")
    return mask_util.decode(rle).astype(bool)


def iter_json_array(path: str, chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    """Yield objects from a top-level JSON array without materializing it."""
    decoder = json.JSONDecoder()
    buf = ""
    pos = 0
    started = False

    with open(path, "r") as f:
        while True:
            chunk = f.read(chunk_size)
            if chunk:
                buf += chunk

            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1

                if not started:
                    if pos >= len(buf):
                        break
                    if buf[pos] != "[":
                        raise ValueError(f"{path} is not a JSON array")
                    started = True
                    pos += 1
                    continue

                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf) and buf[pos] == "]":
                    return
                if pos >= len(buf):
                    break

                try:
                    item, next_pos = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break
                yield item
                pos = next_pos

            if pos:
                buf = buf[pos:]
                pos = 0
            if not chunk:
                if buf.strip():
                    raise ValueError(f"trailing JSON data in {path}")
                return


def build_gt_index(gt_json: str):
    gt = json.load(open(gt_json))
    id2name = {c["id"]: c["name"] for c in gt["categories"]}
    part_cats = {cid for cid, name in id2name.items() if ":" in name}
    img_info = {im["id"]: im for im in gt["images"]}

    gt_by_img_cat = defaultdict(list)
    for ann in gt["annotations"]:
        cid = ann["category_id"]
        if cid not in part_cats:
            continue
        gt_by_img_cat[(ann["image_id"], cid)].append(ann)
    return gt, part_cats, img_info, gt_by_img_cat


def matched_samples_for_image(
    img_id: int,
    preds: List[dict],
    gt_by_img_cat: Dict[Tuple[int, int], List[dict]],
    img_info: Dict[int, dict],
    part_cats: set,
    used_gt: Dict[Tuple[int, int], set],
    coco_root: str,
    vertices: int,
    iou_thresh: float,
) -> Iterator[dict]:
    if img_id not in img_info:
        return
    im = img_info[img_id]
    pred_by_cat = defaultdict(list)
    for pred in preds:
        cid = pred["category_id"]
        if cid in part_cats:
            pred_by_cat[cid].append(pred)

    for cid, pred_group in pred_by_cat.items():
        key = (img_id, cid)
        gt_group = gt_by_img_cat.get(key, [])
        if not gt_group:
            continue
        gt_mask_cache = {}
        used = used_gt[key]
        for pred in sorted(pred_group, key=lambda x: x.get("score", 0.0), reverse=True):
            pred_mask = decode_to_mask(pred["segmentation"], im["height"], im["width"])
            best_j, best_iou = -1, iou_thresh
            for j, gt_ann in enumerate(gt_group):
                if j in used:
                    continue
                if j not in gt_mask_cache:
                    gt_mask_cache[j] = decode_to_mask(
                        gt_ann["segmentation"], im["height"], im["width"]
                    )
                iou = mask_iou(pred_mask, gt_mask_cache[j])
                if iou >= best_iou:
                    best_iou, best_j = iou, j
            if best_j < 0:
                continue
            used.add(best_j)

            pred_contour = mask_to_resampled_contour(pred_mask, vertices)
            gt_contour = mask_to_resampled_contour(gt_mask_cache[best_j], vertices)
            if pred_contour is None or gt_contour is None:
                continue
            gt_contour = align_contour_to_reference(gt_contour, pred_contour)
            if gt_contour is None:
                continue

            yield {
                "image_id": img_id,
                "image_path": os.path.join(coco_root, im["file_name"]),
                "height": im["height"],
                "width": im["width"],
                "category_id": cid,
                "score": pred.get("score", 0.0),
                "pred_contour": pred_contour.astype(np.float32),
                "gt_contour": gt_contour.astype(np.float32),
            }


def iter_training_matches(
    gt_index,
    pred_json: str,
    coco_root: str,
    vertices: int,
    iou_thresh: float,
    max_instances: int,
) -> Iterator[dict]:
    _gt, part_cats, img_info, gt_by_img_cat = gt_index
    used_gt = defaultdict(set)
    current_img_id = None
    current_preds = []
    emitted = 0

    for pred in iter_json_array(pred_json):
        img_id = pred.get("image_id")
        if current_img_id is None:
            current_img_id = img_id
        if img_id != current_img_id:
            for sample in matched_samples_for_image(
                current_img_id,
                current_preds,
                gt_by_img_cat,
                img_info,
                part_cats,
                used_gt,
                coco_root,
                vertices,
                iou_thresh,
            ):
                yield sample
                emitted += 1
                if max_instances and emitted >= max_instances:
                    return
            current_img_id = img_id
            current_preds = []
        current_preds.append(pred)

    if current_img_id is not None:
        for sample in matched_samples_for_image(
            current_img_id,
            current_preds,
            gt_by_img_cat,
            img_info,
            part_cats,
            used_gt,
            coco_root,
            vertices,
            iou_thresh,
        ):
            yield sample
            emitted += 1
            if max_instances and emitted >= max_instances:
                return


def load_training_matches(gt_json, pred_json, coco_root, vertices, iou_thresh, max_instances):
    gt = json.load(open(gt_json))
    preds = json.load(open(pred_json))
    id2name = {c["id"]: c["name"] for c in gt["categories"]}
    part_cats = {cid for cid, name in id2name.items() if ":" in name}
    img_info = {im["id"]: im for im in gt["images"]}

    gt_by_img_cat = defaultdict(list)
    for ann in gt["annotations"]:
        cid = ann["category_id"]
        if cid not in part_cats:
            continue
        im = img_info[ann["image_id"]]
        mask = decode_to_mask(ann["segmentation"], im["height"], im["width"])
        gt_by_img_cat[(ann["image_id"], cid)].append({"mask": mask})

    pred_by_img_cat = defaultdict(list)
    for pred in preds:
        img_id = pred["image_id"]
        cid = pred["category_id"]
        if img_id not in img_info or cid not in part_cats:
            continue
        im = img_info[img_id]
        mask = decode_to_mask(pred["segmentation"], im["height"], im["width"])
        pred_by_img_cat[(img_id, cid)].append(
            {"mask": mask, "score": pred.get("score", 0.0), "category_id": cid}
        )

    samples = []
    for key, pred_group in pred_by_img_cat.items():
        gt_group = gt_by_img_cat.get(key, [])
        used = set()
        for pred in sorted(pred_group, key=lambda x: x["score"], reverse=True):
            best_j, best_iou = -1, iou_thresh
            for j, gt_item in enumerate(gt_group):
                if j in used:
                    continue
                iou = mask_iou(pred["mask"], gt_item["mask"])
                if iou >= best_iou:
                    best_iou, best_j = iou, j
            if best_j < 0:
                continue
            used.add(best_j)

            pred_contour = mask_to_resampled_contour(pred["mask"], vertices)
            gt_contour = mask_to_resampled_contour(gt_group[best_j]["mask"], vertices)
            if pred_contour is None or gt_contour is None:
                continue
            gt_contour = align_contour_to_reference(gt_contour, pred_contour)
            if gt_contour is None:
                continue

            img_id, cid = key
            im = img_info[img_id]
            samples.append(
                {
                    "image_id": img_id,
                    "image_path": os.path.join(coco_root, im["file_name"]),
                    "height": im["height"],
                    "width": im["width"],
                    "category_id": cid,
                    "score": pred["score"],
                    "pred_contour": pred_contour.astype(np.float32),
                    "gt_contour": gt_contour.astype(np.float32),
                }
            )
            if max_instances and len(samples) >= max_instances:
                return samples
    return samples


def sample_levels(extractor, result, vertices_xy, levels):
    feats = [extractor.sample(result, vertices_xy, level=level) for level in levels]
    return torch.cat(feats, dim=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-json", default=os.path.join(os.environ.get("PACO_ANNOTATION_ROOT", ""), "paco_lvis_v1_val_mini.json"))
    ap.add_argument("--pred-json", default="/xuanwu-tank/north/jade/Paco/output/r50_mini/lvis_instances_results.json")
    ap.add_argument("--coco-root", default=os.environ.get("COCO_IMAGE_ROOT", "/xuanwu-tank/north/jade/Paco/coco"))
    ap.add_argument("--config", default="configs/mask_rcnn_configs/r50_attr_fpn_100_ep.py")
    ap.add_argument("--checkpoint", default="/xuanwu-tank/north/jade/Paco/models/r50_fpn_lvis.pth")
    ap.add_argument("--outdir", default="/xuanwu-tank/north/jade/Paco/output/snake_fpn")
    ap.add_argument("--feature-source", choices=["fpn", "dino", "fpn_dino"], default="fpn")
    ap.add_argument("--levels", nargs="+", default=["p2", "p3"])
    ap.add_argument("--vertices", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--max-offset", type=float, default=16.0)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--max-instances", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.feature_source != "fpn":
        raise NotImplementedError("Only --feature-source fpn is implemented in this phase")

    os.makedirs(args.outdir, exist_ok=True)
    extractor = FPNFeatureExtractor(args.config, args.checkpoint, device=args.device)
    gt_index = build_gt_index(args.gt_json)
    model = None
    opt = None
    feature_dim = None
    trained_samples = 0

    for epoch in range(args.epochs):
        total_loss, n = 0.0, 0
        last_image_path = None
        result = None
        for sample in iter_training_matches(
            gt_index,
            args.pred_json,
            args.coco_root,
            args.vertices,
            args.iou_thresh,
            args.max_instances,
        ):
            image_path = sample["image_path"]
            if image_path != last_image_path:
                result = extractor.run(image_path)
                last_image_path = image_path

            if model is None:
                feature_dim = sum(result["features"][level].shape[1] for level in args.levels)
                model = SnakeRefiner(
                    feature_dim=feature_dim,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    max_offset=args.max_offset,
                ).to(args.device)
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            model.train()
            image_size = (sample["height"], sample["width"])

            current = torch.as_tensor(sample["pred_contour"], dtype=torch.float32, device=args.device)
            target = torch.as_tensor(sample["gt_contour"], dtype=torch.float32, device=args.device)

            loss = torch.zeros((), device=args.device)
            for _ in range(args.steps):
                features = sample_levels(extractor, result, current, args.levels)
                vertices_norm = normalize_vertices_tensor(current, image_size)
                offsets = model(vertices_norm, features)[0]
                current = current + offsets
                loss = loss + F.smooth_l1_loss(current, target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())
            n += 1
            trained_samples += 1

        print(f"epoch={epoch + 1} loss={total_loss / max(n, 1):.6f} samples={n}")

    if model is None or opt is None or feature_dim is None or trained_samples == 0:
        raise SystemExit("No matched contour samples found; run baseline eval first")

    ckpt = {
        "model": model.state_dict(),
        "feature_dim": feature_dim,
        "levels": args.levels,
        "vertices": args.vertices,
        "steps": args.steps,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "max_offset": args.max_offset,
    }
    out_path = os.path.join(args.outdir, "snake_refiner.pth")
    torch.save(ckpt, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
