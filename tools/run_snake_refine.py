#!/usr/bin/env python
"""
Run FPN Snake refinement on baseline PACO prediction JSON.

The output keeps the same LVIS/PACO prediction JSON shape so it can be consumed
directly by tools/eval_part_boundary.py.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pycocotools.mask as mask_util
import torch

from paco.snake.contour import (
    contour_to_mask,
    encode_binary_mask,
    mask_to_resampled_contour,
)
from paco.snake.fpn_extractor import FPNFeatureExtractor
from paco.snake.model import SnakeRefiner, normalize_vertices_tensor


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json_atomic(path, data):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def append_jsonl(path, data):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


def progress_payload(args, status, start_time, **extra):
    elapsed = max(time.time() - start_time, 1e-9)
    payload = {
        "time": now_iso(),
        "status": status,
        "elapsed_seconds": elapsed,
        "output_path": os.path.join(args.outdir, "lvis_instances_results.json"),
    }
    payload.update(extra)
    processed_images = payload.get("processed_images", 0)
    refined_preds = payload.get("refined_preds", 0)
    skipped_preds = payload.get("skipped_preds", 0)
    payload["images_per_second"] = processed_images / elapsed if processed_images else 0.0
    payload["preds_per_second"] = (refined_preds + skipped_preds) / elapsed if (refined_preds + skipped_preds) else 0.0
    return payload


def install_failure_status_hook(status_json, start_time):
    original_hook = sys.excepthook

    def hook(exc_type, exc, tb):
        write_json_atomic(
            status_json,
            {
                "time": now_iso(),
                "status": "failed",
                "elapsed_seconds": time.time() - start_time,
                "error": repr(exc),
            },
        )
        original_hook(exc_type, exc, tb)

    sys.excepthook = hook


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
    ap.add_argument("--snake-checkpoint", default="/xuanwu-tank/north/jade/Paco/output/snake_fpn/snake_refiner.pth")
    ap.add_argument("--outdir", default="/xuanwu-tank/north/jade/Paco/output/snake_fpn_eval")
    ap.add_argument("--feature-source", choices=["fpn", "dino", "fpn_dino"], default="fpn")
    ap.add_argument("--max-images", type=int, default=0, help="optional smoke-test cap")
    ap.add_argument("--max-preds", type=int, default=0, help="optional smoke-test cap")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--progress-jsonl", default="", help="optional JSONL progress output")
    ap.add_argument("--status-json", default="", help="optional JSON status output")
    ap.add_argument("--progress-every-images", type=int, default=25)
    ap.add_argument("--progress-every-preds", type=int, default=5000)
    args = ap.parse_args()
    start_time = time.time()
    install_failure_status_hook(args.status_json, start_time)

    if args.feature_source != "fpn":
        raise NotImplementedError("Only --feature-source fpn is implemented in this phase")

    os.makedirs(args.outdir, exist_ok=True)
    gt = json.load(open(args.gt_json))
    preds = json.load(open(args.pred_json))
    img_info = {im["id"]: im for im in gt["images"]}
    preds_by_image = defaultdict(list)
    for pred in preds:
        if pred["image_id"] in img_info:
            preds_by_image[pred["image_id"]].append(pred)
    total_images = len(preds_by_image)
    total_preds = sum(len(group) for group in preds_by_image.values())
    initial_payload = progress_payload(
        args,
        "running",
        start_time,
        processed_images=0,
        total_images=total_images,
        refined_preds=0,
        skipped_preds=0,
        total_preds=total_preds,
        stage="loaded_inputs",
    )
    write_json_atomic(args.status_json, initial_payload)
    append_jsonl(args.progress_jsonl, initial_payload)

    ckpt = torch.load(args.snake_checkpoint, map_location=args.device)
    model = SnakeRefiner(
        feature_dim=ckpt["feature_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        max_offset=ckpt["max_offset"],
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    levels = ckpt["levels"]
    vertices = ckpt["vertices"]
    steps = ckpt["steps"]

    extractor = FPNFeatureExtractor(args.config, args.checkpoint, device=args.device)
    refined = []
    skipped = 0
    processed_images = 0
    last_logged_preds = 0
    with torch.no_grad():
        for img_id, group in preds_by_image.items():
            if args.max_images and processed_images >= args.max_images:
                break
            im = img_info[img_id]
            image_path = os.path.join(args.coco_root, im["file_name"])
            result = extractor.run(image_path)
            image_size = (im["height"], im["width"])
            processed_images += 1

            for pred in group:
                if args.max_preds and len(refined) >= args.max_preds:
                    break
                mask = decode_to_mask(pred["segmentation"], im["height"], im["width"])
                contour = mask_to_resampled_contour(mask, vertices)
                if contour is None:
                    refined.append(pred)
                    skipped += 1
                    continue

                current = torch.as_tensor(contour, dtype=torch.float32, device=args.device)
                for _ in range(steps):
                    features = sample_levels(extractor, result, current, levels)
                    vertices_norm = normalize_vertices_tensor(current, image_size)
                    offsets = model(vertices_norm, features)[0]
                    current = current + offsets
                    current[:, 0].clamp_(0, max(im["width"] - 1, 0))
                    current[:, 1].clamp_(0, max(im["height"] - 1, 0))

                new_mask = contour_to_mask(
                    current.detach().cpu().numpy(),
                    im["height"],
                    im["width"],
                )
                new_pred = dict(pred)
                new_pred["segmentation"] = encode_binary_mask(new_mask)
                refined.append(new_pred)
                total_done_preds = len(refined)
                if args.progress_every_preds and total_done_preds - last_logged_preds >= args.progress_every_preds:
                    payload = progress_payload(
                        args,
                        "running",
                        start_time,
                        processed_images=processed_images,
                        total_images=total_images,
                        refined_preds=len(refined) - skipped,
                        skipped_preds=skipped,
                        total_preds=total_preds,
                        stage="refining_predictions",
                    )
                    write_json_atomic(args.status_json, payload)
                    append_jsonl(args.progress_jsonl, payload)
                    print(
                        "REFINE_PROGRESS "
                        f"images={processed_images}/{total_images} "
                        f"preds={len(refined)}/{total_preds} "
                        f"elapsed={payload['elapsed_seconds']:.1f}s",
                        flush=True,
                    )
                    last_logged_preds = total_done_preds
            if args.max_preds and len(refined) >= args.max_preds:
                break
            if args.progress_every_images and processed_images % args.progress_every_images == 0:
                payload = progress_payload(
                    args,
                    "running",
                    start_time,
                    processed_images=processed_images,
                    total_images=total_images,
                    refined_preds=len(refined) - skipped,
                    skipped_preds=skipped,
                    total_preds=total_preds,
                    stage="refining_images",
                )
                write_json_atomic(args.status_json, payload)
                append_jsonl(args.progress_jsonl, payload)
                print(
                    "REFINE_PROGRESS "
                    f"images={processed_images}/{total_images} "
                    f"preds={len(refined)}/{total_preds} "
                    f"elapsed={payload['elapsed_seconds']:.1f}s",
                    flush=True,
                )

    out_path = os.path.join(args.outdir, "lvis_instances_results.json")
    with open(out_path, "w") as f:
        json.dump(refined, f)
    completed_payload = progress_payload(
        args,
        "completed",
        start_time,
        processed_images=processed_images,
        total_images=total_images,
        refined_preds=len(refined) - skipped,
        skipped_preds=skipped,
        total_preds=total_preds,
        stage="completed",
    )
    write_json_atomic(args.status_json, completed_payload)
    append_jsonl(args.progress_jsonl, completed_payload)
    print(
        f"Wrote {out_path}; images={processed_images} "
        f"refined={len(refined) - skipped} skipped={skipped}"
    )


if __name__ == "__main__":
    main()
