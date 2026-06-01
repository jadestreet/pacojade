#!/usr/bin/env python
"""
Phase 5 — report visualizations for the PACO baseline on the 5-category mini set.

Produces, into results/figures/:
  data_gt_<obj>.png        GT part masks overlaid (one per object category)
  pred_vs_gt_<obj>.png     GT vs baseline-predicted part masks, side by side
  boundary_<obj>_<part>.png  zoom on a thin part: pred contour vs GT contour
  fpn_<obj>_<part>.png     P2/P3 feature heatmap for the same crop
  hist_iou_bf.png          per-instance IoU / boundary-F histograms (if dump given)

Self-validating: asserts each core figure exists; exits non-zero otherwise.
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
import cv2
import torch

import paco.data.datasets  # noqa: registers datasets
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.detection_utils import read_image
from paco.snake.fpn_extractor import FPNFeatureExtractor

OBJS = ["chair", "bottle", "mug", "laptop_computer", "car_(automobile)"]
THIN = {"leg", "arm", "handle", "neck", "rim", "spout", "stretcher", "antenna",
        "wiper", "stem", "rod", "bar", "spindle", "rail"}


def decode_mask(segm, h, w):
    if isinstance(segm, list):
        rle = mask_util.merge(mask_util.frPyObjects(segm, h, w))
    else:
        rle = segm
        if isinstance(rle.get("counts"), list):
            rle = mask_util.frPyObjects(rle, h, w)
    return mask_util.decode(rle).astype(bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="paco_lvis_v1_val_mini")
    ap.add_argument("--config", default="configs/mask_rcnn_configs/r50_attr_fpn_100_ep.py")
    ap.add_argument("--checkpoint", default="models/r50_fpn_lvis.pth")
    ap.add_argument("--outdir", default="results/figures")
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--per-instance", default=None, help="json from eval_part_boundary --dump-per-instance")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    produced = []

    dicts = DatasetCatalog.get(args.dataset)
    meta = MetadataCatalog.get(args.dataset)
    names = meta.thing_classes
    part_ids = {i for i, n in enumerate(names) if ":" in n and n.split(":")[0] in OBJS}

    by_id = {d["image_id"]: d for d in dicts}

    # best image per object = the one with the LARGEST single part bbox of that
    # object (clear, big instances make readable report figures). The loaded
    # dataset_dicts drop the COCO "area" field, so use bbox area from the XYWH
    # box. Pure max-part-COUNT picks crowded scenes (e.g. a stadium full of
    # tiny "chair:*" seats) which are useless as figures.
    def bbox_area(a):
        b = a["bbox"]
        return float(b[2] * b[3])  # XYWH

    best = {}
    for d in dicts:
        big = defaultdict(float)
        for a in d["annotations"]:
            n = names[a["category_id"]]
            if ":" in n and n.split(":")[0] in OBJS:
                obj = n.split(":")[0]
                big[obj] = max(big[obj], bbox_area(a))
        for obj, score in big.items():
            if obj not in best or score > best[obj][1]:
                best[obj] = (d, score)

    extractor = FPNFeatureExtractor(args.config, args.checkpoint, device="cuda")

    def part_record(d):
        r = dict(d)
        r["annotations"] = [a for a in d["annotations"]
                            if ":" in names[a["category_id"]]
                            and names[a["category_id"]].split(":")[0] in OBJS]
        return r

    # ---- Figure 1 & 2: data GT, and GT vs pred ----
    for obj, (d, _c) in best.items():
        img = read_image(d["file_name"], format="RGB")
        # GT
        vis_gt = Visualizer(img, metadata=meta, scale=1.0)
        gt_out = vis_gt.draw_dataset_dict(part_record(d)).get_image()
        p1 = os.path.join(args.outdir, f"data_gt_{obj}.png")
        plt.imsave(p1, gt_out); produced.append(p1)

        # Pred
        res = extractor.run(d["file_name"])
        inst = res["instances"]
        keep = torch.tensor(
            [bool(int(c) in part_ids and float(s) >= args.score_thresh)
             for c, s in zip(inst.pred_classes, inst.scores)]
        ) if len(inst) else torch.zeros(0, dtype=torch.bool)
        inst_k = inst[keep] if len(inst) else inst
        vis_pr = Visualizer(img, metadata=meta, scale=1.0)
        pr_out = vis_pr.draw_instance_predictions(inst_k).get_image()

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(gt_out); ax[0].set_title(f"GT parts: {obj}"); ax[0].axis("off")
        ax[1].imshow(pr_out); ax[1].set_title(f"Pred parts (R50, s>={args.score_thresh})"); ax[1].axis("off")
        p2 = os.path.join(args.outdir, f"pred_vs_gt_{obj}.png")
        fig.tight_layout(); fig.savefig(p2, dpi=110); plt.close(fig); produced.append(p2)

    # ---- Figure 3 & 4: boundary overlay + FPN features for thin parts ----
    # Robust search: scan images that contain thin GT parts (most thin parts
    # first), run the extractor once per image, and match ANY thin GT part whose
    # class is present in THAT image's predictions (IoU > 0.2). A single
    # best-by-count image per object is too fragile -- the model's top-300
    # detections for one image may not include that specific part class.
    def n_thin(d):
        return sum(
            1 for a in d["annotations"]
            if ":" in names[a["category_id"]]
            and names[a["category_id"]].split(":")[-1] in THIN
            and names[a["category_id"]].split(":")[0] in OBJS
        )

    thin_imgs = sorted(
        [d for d in dicts if n_thin(d) > 0], key=n_thin, reverse=True
    )

    made_boundary = 0
    MAX_BOUNDARY_EXAMPLES = 3
    seen_parts = set()
    for d in thin_imgs:
        if made_boundary >= MAX_BOUNDARY_EXAMPLES:
            break
        h, w = d["height"], d["width"]
        res = extractor.run(d["file_name"])
        inst = res["instances"]
        if len(inst) == 0:
            continue
        pred_cids = [int(c) for c in inst.pred_classes]
        pred_cid_set = set(pred_cids)

        gt_thin = [
            a for a in d["annotations"]
            if ":" in names[a["category_id"]]
            and names[a["category_id"]].split(":")[-1] in THIN
            and names[a["category_id"]].split(":")[0] in OBJS
            and a["category_id"] in pred_cid_set
        ]
        # largest-area parts first, and prefer part classes not yet shown
        gt_thin.sort(key=lambda x: (names[x["category_id"]] in seen_parts,
                                    -x.get("area", 0)))

        for a in gt_thin:
            if made_boundary >= MAX_BOUNDARY_EXAMPLES:
                break
            part_name = names[a["category_id"]]
            cid = a["category_id"]
            gtm = decode_mask(a["segmentation"], h, w)

            best_iou, best_i = 0.0, -1
            for i in range(len(inst)):
                if pred_cids[i] != cid:
                    continue
                pm = inst.pred_masks[i].numpy().astype(bool)
                inter = (pm & gtm).sum(); union = (pm | gtm).sum()
                iou = inter / union if union else 0.0
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i < 0 or best_iou < 0.2:
                continue
            predm = inst.pred_masks[best_i].numpy().astype(bool)

            ys, xs = np.where(gtm | predm)
            pad = 15
            y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, h)
            x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, w)
            img = read_image(d["file_name"], format="RGB")
            crop = img[y0:y1, x0:x1].copy()

            def contours(m):
                c, _ = cv2.findContours(m[y0:y1, x0:x1].astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                return c
            ov = crop.copy()
            cv2.drawContours(ov, contours(gtm), -1, (0, 255, 0), 2)
            cv2.drawContours(ov, contours(predm), -1, (255, 0, 0), 2)
            tag = f"{part_name.replace(':','_')}_{d['image_id']}"
            p3 = os.path.join(args.outdir, f"boundary_{tag}.png")
            plt.imsave(p3, ov); produced.append(p3)

            # FPN feature heatmap (P2) over the same crop
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(crop); ax[0].set_title(f"{part_name} crop"); ax[0].axis("off")
            ax[1].imshow(ov)
            ax[1].set_title(f"GT(green) vs Pred(red) IoU={best_iou:.2f}"); ax[1].axis("off")
            lvl = "p2"
            feat = res["features"][lvl][0]  # (C,Hf,Wf)
            fmap = feat.norm(dim=0).cpu().numpy()  # (Hf,Wf)
            sx, sy = res["scale"]; stride = extractor.FPN_STRIDES[lvl]
            fy0, fy1 = int(y0 * sy / stride), int(y1 * sy / stride)
            fx0, fx1 = int(x0 * sx / stride), int(x1 * sx / stride)
            sub = fmap[max(fy0, 0):fy1, max(fx0, 0):fx1]
            ax[2].imshow(sub, cmap="viridis")
            ax[2].set_title(f"{lvl} feature norm"); ax[2].axis("off")
            p4 = os.path.join(args.outdir, f"fpn_{tag}.png")
            fig.tight_layout(); fig.savefig(p4, dpi=110); plt.close(fig)
            produced.append(p4)
            seen_parts.add(part_name)
            made_boundary += 1

    # ---- Figure 5: histograms ----
    if args.per_instance and os.path.exists(args.per_instance):
        recs = json.load(open(args.per_instance))
        if recs:
            ious = [r["iou"] for r in recs]; bfs = [r["bf"] for r in recs]
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].hist(ious, bins=20, color="steelblue"); ax[0].set_title("per-instance IoU"); ax[0].set_xlabel("IoU")
            ax[1].hist(bfs, bins=20, color="indianred"); ax[1].set_title("per-instance boundary-F"); ax[1].set_xlabel("BF")
            p5 = os.path.join(args.outdir, "hist_iou_bf.png")
            fig.tight_layout(); fig.savefig(p5, dpi=110); plt.close(fig); produced.append(p5)

    # validate
    summary = "\n".join(produced)
    with open(os.path.join(args.outdir, "_produced.txt"), "w") as f:
        f.write(summary + "\n")
    for p in produced:
        assert os.path.exists(p) and os.path.getsize(p) > 1000, f"bad figure {p}"
    assert len(best) == len(OBJS), f"missing object samples: {set(OBJS) - set(best)}"
    assert made_boundary > 0, "no boundary/FPN example produced"
    print(f"VISUALIZE_OK {len(produced)} figures, {made_boundary} boundary examples")


if __name__ == "__main__":
    main()
