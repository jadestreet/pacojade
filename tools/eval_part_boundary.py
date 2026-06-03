#!/usr/bin/env python
"""
Boundary-sensitive part-segmentation eval: mean part-mask IoU + standard
boundary F-measure (2px) over matched predicted/GT part masks, reported overall,
per object category, and for a thin/elongated-parts slice.

Predictions come from the PACOEvaluator dump (lvis_instances_results.json), whose
``category_id`` is the original PACO dataset id (matches the GT json). Masks are
matched greedily per (image, part-category) at IoU >= 0.5.

Example:
    python tools/eval_part_boundary.py \
        --gt   $PACO_ANNOTATION_ROOT/paco_lvis_v1_val_mini.json \
        --pred output/r50_mini/lvis_instances_results.json \
        --out  results/phase1_baseline_metrics.md
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

from paco.evaluation.boundary_metrics import match_and_score

# Semantic part names (the token after "obj:") considered thin / elongated, where
# the proposal expects contour refinement to help most.
THIN_PARTS = {
    "leg", "arm", "handle", "stretcher", "spindle", "rail", "stile", "neck",
    "rim", "antenna", "wire", "cable", "strap", "stem", "spout", "rod", "shaft",
    "fork", "prong", "bar", "pipe", "wiper", "runningboard", "tube", "down_tube",
    "top_tube", "seat_tube", "seat_stay",
}


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
        "images_per_second": extra.get("processed_images", 0) / elapsed
        if extra.get("processed_images", 0)
        else 0.0,
    }
    payload.update(extra)
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
    """Decode COCO polygon/RLE segmentation to a boolean HxW mask."""
    if isinstance(segm, list):  # polygons
        rles = mask_util.frPyObjects(segm, h, w)
        rle = mask_util.merge(rles)
    elif isinstance(segm, dict):
        rle = segm
        if isinstance(rle.get("counts"), list):  # uncompressed RLE
            rle = mask_util.frPyObjects(rle, h, w)
    else:
        raise ValueError(f"unsupported segmentation type {type(segm)}")
    return mask_util.decode(rle).astype(bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True, help="(mini) PACO-LVIS json with GT")
    ap.add_argument("--pred", required=True, help="lvis_instances_results.json")
    ap.add_argument("--out", default=None, help="optional markdown report path")
    ap.add_argument(
        "--dump-per-instance",
        default=None,
        help="optional json path for per-matched-instance {category,iou,bf} records",
    )
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--progress-jsonl", default="", help="optional JSONL progress output")
    ap.add_argument("--status-json", default="", help="optional JSON status output")
    ap.add_argument("--progress-every-images", type=int, default=100)
    args = ap.parse_args()
    start_time = time.time()
    install_failure_status_hook(args.status_json, start_time)

    def record(status, **extra):
        payload = progress_payload(args, status, start_time, **extra)
        write_json_atomic(args.status_json, payload)
        append_jsonl(args.progress_jsonl, payload)
        print(
            "BOUNDARY_PROGRESS "
            f"stage={payload.get('stage')} "
            f"images={payload.get('processed_images', 0)}/{payload.get('total_images', 0)} "
            f"matches={payload.get('matches', 0)} "
            f"elapsed={payload['elapsed_seconds']:.1f}s",
            flush=True,
        )

    gt = json.load(open(args.gt))
    id2name = {c["id"]: c["name"] for c in gt["categories"]}
    part_cats = {cid for cid, n in id2name.items() if ":" in n}  # object-part cats
    img_hw = {im["id"]: (im["height"], im["width"]) for im in gt["images"]}
    total_images = len(img_hw)
    record("running", stage="loaded_gt_metadata", processed_images=0, total_images=total_images)

    def parent_obj(cid):
        return id2name[cid].split(":")[0]

    def semantic_part(cid):
        return id2name[cid].split(":")[-1]

    # GT part instances grouped by image
    gt_by_img = defaultdict(list)
    gt_part_masks = 0
    for a in gt["annotations"]:
        cid = a["category_id"]
        if cid not in part_cats:
            continue
        h, w = img_hw[a["image_id"]]
        gt_by_img[a["image_id"]].append(
            {"mask": decode_to_mask(a["segmentation"], h, w), "category_id": cid}
        )
        gt_part_masks += 1
    record(
        "running",
        stage="decoded_gt_parts",
        processed_images=0,
        total_images=total_images,
        gt_part_masks=gt_part_masks,
    )

    # Predicted part instances grouped by image
    preds = json.load(open(args.pred))
    pred_by_img = defaultdict(list)
    pred_part_masks = 0
    for p in preds:
        cid = p["category_id"]
        if cid not in part_cats or p["image_id"] not in img_hw:
            continue
        h, w = img_hw[p["image_id"]]
        pred_by_img[p["image_id"]].append(
            {
                "mask": decode_to_mask(p["segmentation"], h, w),
                "category_id": cid,
                "score": p.get("score", 0.0),
            }
        )
        pred_part_masks += 1
    record(
        "running",
        stage="decoded_pred_parts",
        processed_images=0,
        total_images=total_images,
        gt_part_masks=gt_part_masks,
        pred_part_masks=pred_part_masks,
    )

    # Match per image and accumulate
    per_cat = defaultdict(lambda: {"iou": [], "bf": [], "n_gt": 0})
    per_instance = []
    for processed_images, img_id in enumerate(img_hw, start=1):
        for g in gt_by_img.get(img_id, []):
            per_cat[g["category_id"]]["n_gt"] += 1
        matches = match_and_score(
            pred_by_img.get(img_id, []),
            gt_by_img.get(img_id, []),
            iou_thresh=args.iou_thresh,
            tol=args.tol,
        )
        for m in matches:
            per_cat[m["category_id"]]["iou"].append(m["iou"])
            per_cat[m["category_id"]]["bf"].append(m["bf"])
            per_instance.append(
                {
                    "category": id2name[m["category_id"]],
                    "semantic_part": semantic_part(m["category_id"]),
                    "object": parent_obj(m["category_id"]),
                    "iou": m["iou"],
                    "bf": m["bf"],
                }
            )
        if args.progress_every_images and processed_images % args.progress_every_images == 0:
            record(
                "running",
                stage="matching",
                processed_images=processed_images,
                total_images=total_images,
                gt_part_masks=gt_part_masks,
                pred_part_masks=pred_part_masks,
                matches=len(per_instance),
            )
    record(
        "running",
        stage="matched_all_images",
        processed_images=total_images,
        total_images=total_images,
        gt_part_masks=gt_part_masks,
        pred_part_masks=pred_part_masks,
        matches=len(per_instance),
    )

    if args.dump_per_instance:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_per_instance)), exist_ok=True)
        with open(args.dump_per_instance, "w") as f:
            json.dump(per_instance, f)

    def agg(cat_ids):
        ious, bfs, n_gt, n_match = [], [], 0, 0
        for cid in cat_ids:
            d = per_cat.get(cid)
            if not d:
                continue
            ious += d["iou"]
            bfs += d["bf"]
            n_gt += d["n_gt"]
            n_match += len(d["iou"])
        miou = float(np.mean(ious)) if ious else float("nan")
        mbf = float(np.mean(bfs)) if bfs else float("nan")
        recall = (n_match / n_gt) if n_gt else float("nan")
        return miou, mbf, n_match, n_gt, recall

    all_part_cats = sorted(per_cat.keys())
    by_obj = defaultdict(list)
    for cid in part_cats:
        by_obj[parent_obj(cid)].append(cid)
    thin_cats = [cid for cid in part_cats if semantic_part(cid) in THIN_PARTS]

    lines = []
    lines.append("# Phase 1 — Baseline boundary metrics (R50 FPN, PACO-LVIS val mini)\n")
    lines.append(f"- predictions: `{args.pred}`")
    lines.append(f"- matching: greedy per (image, part-category) at IoU >= {args.iou_thresh}")
    lines.append(f"- boundary-F tolerance: {args.tol}px\n")
    header = "| group | mean IoU | boundary-F | matched/GT | recall@%.1f |" % args.iou_thresh
    sep = "|---|---|---|---|---|"

    def row(name, cat_ids):
        miou, mbf, nm, ng, rec = agg(cat_ids)
        return (
            f"| {name} | {miou:.3f} | {mbf:.3f} | {nm}/{ng} | {rec:.3f} |"
        )

    lines += ["## Overall", header, sep, row("all parts", all_part_cats), ""]
    lines += ["## Per object category", header, sep]
    for obj in sorted(by_obj):
        lines.append(row(obj, by_obj[obj]))
    lines += ["", "## Thin / elongated parts", header, sep,
              row("thin parts", thin_cats), ""]

    report = "\n".join(lines)
    print(report)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(report + "\n")
        print(f"\nWrote {args.out}")
    record(
        "completed",
        stage="completed",
        processed_images=total_images,
        total_images=total_images,
        gt_part_masks=gt_part_masks,
        pred_part_masks=pred_part_masks,
        matches=len(per_instance),
        out=args.out,
        per_instance_out=args.dump_per_instance,
    )


if __name__ == "__main__":
    main()
