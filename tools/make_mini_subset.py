#!/usr/bin/env python
"""
Build a minimal PACO-LVIS subset restricted to a handful of object categories
(and their object-part categories), for fast local baseline experiments.

The output JSON keeps the FULL ``categories`` / ``part_categories`` /
``attributes`` blocks unchanged (so category ids stay consistent with the
pretrained model and the PACO evaluator simply assigns AP=-1 to GT-less classes
and filters them out of the mean). Only ``images`` and ``annotations`` are
filtered.

Example:
    python tools/make_mini_subset.py \
        --src $PACO_ANNOTATION_ROOT/paco_lvis_v1_val.json \
        --out $PACO_ANNOTATION_ROOT/paco_lvis_v1_val_mini.json
"""
import argparse
import json
import os

# Proposal's 5-category subset (exact PACO object names).
DEFAULT_OBJECTS = [
    "chair",
    "bottle",
    "mug",
    "laptop_computer",
    "car_(automobile)",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="source paco_lvis_v1_*.json")
    ap.add_argument("--out", required=True, help="output mini json path")
    ap.add_argument(
        "--objects",
        nargs="*",
        default=DEFAULT_OBJECTS,
        help="object category names to keep (parts of these are kept too)",
    )
    ap.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="optional global cap on #images (0 = no cap / full subset)",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="optional path to write image file_name/coco_url manifest (jsonl)",
    )
    args = ap.parse_args()

    with open(args.src) as f:
        data = json.load(f)

    objs = set(args.objects)

    # Category names look like "chair" (object) or "chair:leg" (object-part).
    id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    keep_cat_ids = set()
    for cid, name in id_to_name.items():
        base = name.split(":")[0]
        if name in objs or base in objs:
            keep_cat_ids.add(cid)

    missing = objs - {n.split(":")[0] for n in id_to_name.values()}
    if missing:
        raise SystemExit(f"ERROR: object names not found in categories: {missing}")

    n_obj = sum(1 for cid in keep_cat_ids if ":" not in id_to_name[cid])
    n_part = len(keep_cat_ids) - n_obj
    print(
        f"Keeping {len(keep_cat_ids)} category ids "
        f"({n_obj} object + {n_part} object-part) for {sorted(objs)}"
    )

    anns = [a for a in data["annotations"] if a["category_id"] in keep_cat_ids]
    img_ids_with_ann = {a["image_id"] for a in anns}
    images = sorted(
        (im for im in data["images"] if im["id"] in img_ids_with_ann),
        key=lambda im: im["id"],
    )

    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]
        kept = {im["id"] for im in images}
        anns = [a for a in anns if a["image_id"] in kept]

    out = dict(data)  # preserve categories/part_categories/attributes/etc.
    out["images"] = images
    out["annotations"] = anns

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)

    print(
        f"Wrote {args.out}: {len(images)} images, {len(anns)} annotations "
        f"({sum(1 for a in anns if ':' in id_to_name[a['category_id']])} part anns)"
    )

    if args.manifest:
        with open(args.manifest, "w") as f:
            for im in images:
                fn = im["file_name"]
                url = im.get("coco_url") or f"http://images.cocodataset.org/{fn}"
                f.write(json.dumps({"file_name": fn, "coco_url": url}) + "\n")
        print(f"Wrote manifest {args.manifest} ({len(images)} entries)")


if __name__ == "__main__":
    main()
