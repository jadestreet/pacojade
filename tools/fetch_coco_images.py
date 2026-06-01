#!/usr/bin/env python
"""
Download only the COCO images referenced by a (mini) PACO-LVIS JSON, into
``$COCO_IMAGE_ROOT`` using the same relative layout that
``paco/data/datasets/paco.py`` expects (``os.path.join(image_root, file_name)``).

Source URL is taken from each image's ``coco_url`` (which encodes the
train2017/val2017 subfolder); the destination path uses ``file_name`` so it
always matches the dataset loader regardless of whether file_name is bare or
already includes a subfolder.

Example:
    python tools/fetch_coco_images.py \
        --json $PACO_ANNOTATION_ROOT/paco_lvis_v1_val_mini.json
"""
import argparse
import concurrent.futures as cf
import json
import os
import urllib.request


def download_one(url_dest):
    url, dest = url_dest
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return ("skip", dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
        return ("ok", dest)
    except Exception as e:  # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        return ("fail", f"{dest} <- {url}: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True, help="(mini) PACO-LVIS json")
    ap.add_argument(
        "--image-root",
        default=os.environ.get("COCO_IMAGE_ROOT"),
        help="defaults to $COCO_IMAGE_ROOT",
    )
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not args.image_root:
        raise SystemExit("ERROR: set $COCO_IMAGE_ROOT or pass --image-root")

    with open(args.json) as f:
        data = json.load(f)

    tasks = {}
    for im in data["images"]:
        fn = im["file_name"]
        url = im.get("coco_url") or f"http://images.cocodataset.org/{fn}"
        dest = os.path.join(args.image_root, fn)
        tasks[dest] = (url, dest)
    tasks = list(tasks.values())

    print(f"{len(tasks)} unique images -> {args.image_root}")
    counts = {"ok": 0, "skip": 0, "fail": 0}
    fails = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (status, info) in enumerate(ex.map(download_one, tasks), 1):
            counts[status] += 1
            if status == "fail":
                fails.append(info)
            if i % 50 == 0 or i == len(tasks):
                print(
                    f"  [{i}/{len(tasks)}] ok={counts['ok']} "
                    f"skip={counts['skip']} fail={counts['fail']}",
                    flush=True,
                )

    print(f"DONE ok={counts['ok']} skip={counts['skip']} fail={counts['fail']}")
    if fails:
        print("FAILURES:")
        for x in fails[:20]:
            print("  " + x)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
