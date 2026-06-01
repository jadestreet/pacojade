#!/usr/bin/env python
"""
Run PACO baseline eval on a dataset and DUMP the numeric results to JSON (so the
part/obj AP is available as a file, not just stdout). Also writes the
PACOEvaluator predictions dump (lvis_instances_results.json) into --outdir.

Example:
    python tools/run_eval.py \
        --config configs/mask_rcnn_configs/r50_attr_fpn_100_ep.py \
        --checkpoint models/r50_fpn_lvis.pth \
        --dataset paco_lvis_v1_val_mini \
        --outdir output/r50_mini
"""
import argparse
import json
import os

import numpy as np

import paco.data.datasets  # noqa: registers datasets
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate, LazyConfig
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = LazyConfig.load(args.config)
    cfg.dataloader.test.dataset.names = args.dataset
    cfg.dataloader.evaluator.output_dir = args.outdir

    model = instantiate(cfg.model)
    model.to("cuda")
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(args.checkpoint)

    ret = inference_on_dataset(
        model,
        instantiate(cfg.dataloader.test),
        instantiate(cfg.dataloader.evaluator),
    )
    out = jsonable(ret)
    with open(os.path.join(args.outdir, "phase2_ap.json"), "w") as f:
        json.dump(out, f, indent=2)

    # compact, parseable success line
    segm = out.get("segm", {})
    pp = segm.get("post-processed", {})
    print(
        "EVAL_OK "
        f"segm_AP={segm.get('AP')} "
        f"obj_AP={pp.get('obj-AP')} "
        f"part_AP={pp.get('obj-part-AP-heirarchical')}"
    )


if __name__ == "__main__":
    main()
