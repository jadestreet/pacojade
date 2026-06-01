# PACO Deep-Snake project — progress & environment notes

_Authoritative status doc. Phase 1 (baseline + FPN features) is COMPLETE & VERIFIED
(2026-05-31). Read this first before resuming — especially the env recipe below,
which is the single most time-consuming thing to rediscover._

---

## TL;DR for the next iteration

- **Env works.** `conda activate paco` then everything imports + runs on GPU.
  If something doesn't import, see "Environment recipe" — do NOT start from scratch.
- **Phase 1 done:** baseline part AP 16.13, mean part IoU 0.745 / boundary-F 0.838,
  thin parts IoU 0.692; 17 report figures; 7/7 unit tests pass.
- **Next:** Deep Snake contour module (roadmap in `plan.md` §11), unblocked by the
  FPN extractor (`paco/snake/fpn_extractor.py`).
- **Hard rule (user):** debug conda/pip envs with install/uninstall ONLY. Never edit
  package source or binaries — downgrade/pin instead.
- **Session gotcha:** this session had an intermittent bash-output rendering issue.
  Always write command output to a file and `Read` it; don't trust streamed stdout.
  (And don't run verification concurrently with a pip install — see env note #5.)

---

## Environment recipe (VERIFIED working, pip-only)

Machine: qilin, RTX A6000 (sm_86), system CUDA 12.6 (nvcc), gcc 11.4.
Env: `/home/gaorory/miniforge3/envs/paco`, python 3.9.23.

**Final working pins** (rebuild from these if the env is ever lost):
```
numpy==1.23.5   scipy==1.10.1   Pillow<10 (9.5.0)
torch==1.10.2+cu113   torchvision==0.11.3+cu113
detectron2 0.6  (built from PACO's pinned commit, CPU-ops build — see #3)
opencv-python 4.11  lvis 0.5.3  ego4d 1.7.3  submitit 1.5.4  pytest 8.4.2
```

Install order that works (always pin numpy in the same command so the resolver
can't bump it):
```bash
conda activate paco
pip install "numpy==1.23.5" "scipy==1.10.1" "Pillow<10" pytest
pip install "numpy==1.23.5" opencv-python lvis ego4d submitit
# detectron2: CPU-ops build from PACO's pinned commit (see #3 for why)
pip uninstall -y detectron2
CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0 pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/detectron2.git@0703e08a5f589f7503a3fbfce41309c80204eec8"
```
Data env vars (persisted in the conda env):
`PACO_ANNOTATION_ROOT=/home/gaorory/cs269/datasets/paco/annotations`,
`COCO_IMAGE_ROOT=/home/gaorory/cs269/datasets/coco`,
`PACO_IMAGE_ROOT=/home/gaorory/cs269/datasets/paco/images`.

Smoke test (run from a NON-repo cwd, e.g. /tmp, so the local `paco/` dir doesn't
shadow the installed package):
```python
import torch, detectron2, paco.data.datasets
from detectron2.data import DatasetCatalog
assert torch.cuda.is_available()
assert len(DatasetCatalog.get("paco_lvis_v1_val_mini")) == 423
from detectron2.data.detection_utils import get_fed_loss_cls_weights  # must import
```

---

## How the env was debugged (method — reuse this)

The env *looked* fine (activated, torch saw the GPU), but five distinct issues
were stacked; each fix only exposed the next. What made it converge:

1. **Root cause before any fix; one variable at a time.** Each fix was minimal
   and verified before the next, so a new error was confidently "the next layer,"
   not a regression.
2. **Read the traceback's bottom line + the failing library's source.** Every
   error named the exact file/line and pointed straight at the cause — no guessing.
3. **Trust files, not stdout.** Output rendering was flaky and led to early false
   "it works" claims. Fix: pipe every check to a file and `Read` it
   (`python -m pip list` is the authoritative installed-package source, not memory).
4. **Stay inside the user's pip/conda-only rule.** e.g. downgrading numpy fixes
   ALL of lvis's `np.float` uses at once — strictly better than editing lvis source.

### The five root causes (in discovery order)

1. **numpy — two-sided constraint → pin `numpy==1.23.5`.** numpy 2.x breaks torch
   1.10's C-API ABI (`"module compiled using NumPy 1.x cannot be run in NumPy
   2.0.2"`); but numpy ≥1.24 removed `np.float`, which lvis 0.5.3's `accumulate()`
   uses during AP computation (`AttributeError: module 'numpy' has no attribute
   'float'`). Only numpy 1.23.x satisfies both. Pair with `scipy==1.10.1`.
   - Diagnostic tell: inference produced predictions fine; only the post-inference
     AP-accumulation step crashed → the bug was in lvis, not the model.

2. **`requirements.txt` deps were never installed.** `pip install -e .` had run
   (editable `paco` present) but only ~17 packages existed. Tell: scipy+pytest
   installed in one command, yet pytest still failed to import → that contradiction
   prompted dumping the authoritative `pip list` instead of guessing.

3. **detectron2 — prebuilt 0.6 wheel too old → build from PACO's pinned commit,
   CPU-ops only.** The release wheel imports but lacks `get_fed_loss_cls_weights`
   (added after 0.6; PACO's config imports it; pinned commit 0703e08… dated
   2022-09-12 has it). A normal CUDA build would FAIL: system nvcc is 12.6 but
   torch is cu113. Inspected detectron2 source → R50-FPN routes ROIAlign/NMS
   through **torchvision**, not detectron2's `_C` CUDA ops; the only `_C` ops are
   rotated-box/deform-conv/COCOeval (unused here). So a CPU-ops build
   (`CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0`, which makes `torch.cuda.is_available()`
   False at build time) is sufficient and avoids the toolchain conflict. **GPU is
   still used at runtime** — only the op-compilation was CPU.

4. **Pillow 11.3 removed `Image.LINEAR`** used by detectron2 0.6
   (`transforms/transform.py:46`) → pin `Pillow<10` (9.5.0).

5. **(Transient — not a real fix, but a lesson.)** A one-off
   `"libopenblasp-...so: cannot open shared object file"` appeared because a
   verification ran *concurrently* with a pip uninstall, catching scipy's temp
   `~cipy.libs` rename mid-operation. It resolved once installs settled.
   **Lesson: never run verification while a pip install/uninstall is in flight.**

---

## Data (VERIFIED done)

- `paco_lvis_v1.zip` (129 MB, sha256 verified), extracted via Python `zipfile`
  (`unzip` binary is absent on this machine).
- Mini subset `paco_lvis_v1_val_mini.json`: **423 images, 7421 anns (5637 part
  anns)**, all 5 objects (chair, bottle, mug, laptop_computer, car_(automobile)).
- All 423 referenced COCO images downloaded to `$COCO_IMAGE_ROOT` (0 missing).
- Registered as `paco_lvis_v1_val_mini` in `paco/data/datasets/builtin.py`.
- NOTE: the loaded dataset_dicts drop the COCO `area` field; compute area from the
  XYWH bbox if needed (this bit the figure-selection code — see visualize.py).

---

## Phase 1 results (COMPLETE & VERIFIED, 2026-05-31)

Baseline = R50 FPN (LVIS), `models/r50_fpn_lvis.pth`, eval on the 423-img mini set.

**Detection AP** (`output/r50_mini/phase2_ap.json`):
- part AP (hierarchical) = **16.13**  ← matches paper full-set R50 = 15.8 ✓
- object AP = 44.3, segm AP = 18.2, bbox AP = 22.1

**Boundary metrics** (`results/phase1_baseline_metrics.md`, 2777 matched parts):
- all parts: mean IoU **0.745**, boundary-F **0.838**, recall@0.5 0.493
- thin/elongated parts: IoU **0.692**, BF 0.856  ← the gap Deep Snake targets
- per object IoU: bottle 0.755 / car 0.714 / chair 0.747 / laptop 0.797 / mug 0.781
- Reproducible: re-run on the final JSON gave byte-identical numbers (inference is
  deterministic; a feared eval/boundary file-overwrite race caused no corruption).

**Figures** (`results/figures/`, 17 PNGs, visually checked):
data_gt_<obj> ×5, pred_vs_gt_<obj> ×5, boundary_chair_{arm,leg,stretcher} ×3,
fpn_chair_{arm,leg,stretcher} ×3 (crop + GT/pred contour + P2 feature heatmap),
hist_iou_bf. (Figure image-selection uses largest-bbox part per object, not max
part-count — count picks crowded/unreadable scenes like a stadium of tiny seats.)

**Unit tests:** `tests/test_boundary_metrics.py` 7/7 pass.

---

## Code delivered this phase

- `tools/make_mini_subset.py`, `tools/fetch_coco_images.py` — data prep
- `tools/run_eval.py` — eval → AP json + predictions dump
- `paco/evaluation/boundary_metrics.py` + `tools/eval_part_boundary.py`
  + `tests/test_boundary_metrics.py` — IoU + standard boundary-F (2px)
- `paco/snake/fpn_extractor.py` — FPN feature + predicted-mask extraction interface
  (the Deep-Snake input contract: predicted masks + `sample(points, level)` over
  FPN P2–P5 with correct stride alignment)
- `tools/visualize.py` — Phase-5 report figures
- `paco/data/datasets/builtin.py` — registered `paco_lvis_v1_val_mini`

---

## Next actions (Phase 5+ — Deep Snake, see plan.md §11)

1. Contour module: predicted mask → ordered contour → resample to K vertices.
2. Circular-conv block predicting per-vertex 2D offsets; iterate N steps, sampling
   FPN P2/P3 via `FPNFeatureExtractor.sample()`.
3. Train ONLY the refinement module (detector frozen) on 1 GPU.
4. Re-evaluate with `eval_part_boundary.py`; the original-vs-refined delta on
   IoU/boundary-F (esp. thin parts) is the headline result.
5. (Optional) DINOv3 feature exploration on chair+mug.
