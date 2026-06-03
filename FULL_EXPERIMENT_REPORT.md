# PACO Full FPN Snake Experiment Report

## S1 Existing Experiment Report

### Repository State
- repo path: `/home/jade1st/pj/pacojade-rory`
- branch: `rory`
- setup commit: `362bae22478e4cc2e7261c7a9bb92f772b92705b`
- active runbook / detailed state tracker: `SETUP_REPORT_269contour.md`

### Environment
- conda env: `269contour`
- dataset root: `/xuanwu-tank/north/jade/Paco`
- annotation root: `/xuanwu-tank/north/jade/Paco/paco/annotations`
- COCO image root: `/xuanwu-tank/north/jade/Paco/coco`
- detector checkpoint: `/xuanwu-tank/north/jade/Paco/models/r50_fpn_lvis.pth`

### Official Baseline
- PACO model zoo `r50_fpn_lvis`: `APobj=34.7`, `APopart=15.8`.
- PACO CVPR 2023 Table 2 R50 FPN mask AP: `APobj=31.5 +/- 0.3`, `APopart=12.3 +/- 0.1`.
- Important comparison note: the official numbers are reference points from PACO's reported benchmark setting. Our strict before/after comparison is the local same-split R50-FPN prediction JSON versus the Snake-refined JSON.

### GPU Selection
- Main train/val run used GPUs `4,5` for sharded train prediction dump; later stages mostly used GPU `4`.
- Test refinement reused the trained Snake checkpoint, kept shard0 on GPU `4`, and split shard1 across GPUs `5,6,7` after explicit approval.

### Commands
- train shard 0: `results/experiment_logs/run_20260601_200407/command_train_part_shard_0.sh`
- train shard 1: `results/experiment_logs/run_20260601_200407/command_train_part_shard_1.sh`
- train merge: `results/experiment_logs/run_20260601_200407/command_train_part_merge.sh`
- val dump: `results/experiment_logs/run_20260601_200407/command_detector_val_full.sh`
- Snake training: `results/experiment_logs/run_20260601_200407/command_snake_training_full.sh`
- val refinement: `results/experiment_logs/run_20260601_200407/command_val_refinement_full.sh`
- refined AP eval: `results/experiment_logs/run_20260601_200407/command_refined_ap_eval.sh`
- boundary evals: `results/experiment_logs/run_20260601_200407/command_baseline_boundary_eval.sh`, `results/experiment_logs/run_20260601_200407/command_refined_boundary_eval.sh`
- test split continuation: `results/experiment_logs/test_20260602_144243/continue_after_split_refinement.sh`

### Outputs
- train prediction JSON: `/xuanwu-tank/north/jade/Paco/output/r50_full_train_20260601_200407/lvis_instances_results.json`
- val prediction JSON: `/xuanwu-tank/north/jade/Paco/output/r50_full_val_20260601_200407/lvis_instances_results.json`
- Snake checkpoint: `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407/train/snake_refiner.pth`
- refined val JSON: `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407/refined_val/lvis_instances_results.json`
- refined val AP JSON: `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407/refined_ap.json`
- refined test JSON: `/xuanwu-tank/north/jade/Paco/output/snake_fpn_test_20260602_144243/refined_test/lvis_instances_results.json`
- refined test AP JSON: `/xuanwu-tank/north/jade/Paco/output/snake_fpn_test_20260602_144243/refined_test_ap.json`

### Logs
- main run directory: `results/experiment_logs/run_20260601_200407`
- test run directory: `results/experiment_logs/test_20260602_144243`
- runtime summary: `results/experiment_logs/run_20260601_200407/runtime_summary.csv`
- detailed stage tables, recovery notes, and final test status: `SETUP_REPORT_269contour.md`

### Problems Encountered
- The first single-GPU train prediction dump was too slow and incomplete; it was stopped and replaced by sharded detector dumping.
- Full Snake training initially failed with exit `137` because the train prediction JSON was too large for the original loading path; the run was repaired with a memory-safer training path and resumed from `snake_training`.
- Test shard1 refinement hit a status temp-file collision, then was restarted; later shard1 was split into three deterministic subshards to reduce wall time.
- Test boundary evaluation decoded `1,820,545` predicted part masks and peaked around hundreds of GiB of RAM, so future boundary eval should use streaming/per-image decode.

## S2 Experiment And Result

### 1. Implementation Write-Up
The experiment freezes the official PACO R50-FPN detector and adds a lightweight contour-refinement stage on top of its predicted masks. The detector first produces PACO-LVIS object/part prediction JSONs. For each matched part prediction, the Snake path extracts local FPN features, resamples the predicted mask boundary into a fixed-length contour, and trains `SnakeRefiner` to move contour vertices toward the ground-truth part contour.

The trained Snake checkpoint is then applied at inference time: each detector mask is converted to a contour, refined using FPN features, converted back into a binary mask, and written back into an LVIS/PACO-compatible prediction JSON. The main experiment uses PACO-LVIS train for Snake training and PACO-LVIS val for local before/after evaluation. A final optional test split run reuses the same trained Snake checkpoint; it does not retrain or tune on test.

For slides, describe the pipeline as:

```text
PACO-LVIS images -> frozen R50-FPN detector -> prediction JSON
prediction masks + GT masks -> train FPN Snake contour refiner
val/test detector masks -> FPN Snake refinement -> refined JSON
refined JSON -> AP eval + boundary IoU/F eval + plotting artifacts
```

### 2. Result Breakdown
The strict local comparison did not show a clear improvement from this first FPN Snake integration. On the PACO-LVIS val split, local R50-FPN predictions had approximately `segm_AP=18.2496`, `obj_AP=35.2744`, `part_AP=14.4501`. The Snake-refined val result was `AP=17.3396`, `obj_AP=34.1614`, `part_AP=13.6236`, so the local same-split deltas were about `-0.9100` AP, `-1.1130` object AP, and `-0.8265` part AP.

Boundary metrics show a similar pattern. Local val all-part boundary metrics changed from `IoU=0.739`, `BF=0.807`, `10333/20945` matches to refined `IoU=0.740`, `BF=0.803`, `9654/20945` matches. Mean IoU was essentially flat, boundary-F dropped slightly, and recall dropped because some refined masks no longer matched at the greedy IoU threshold. Thin-part metrics also dropped slightly: local thin parts were about `IoU=0.685`, `BF=0.842`; refined thin parts were about `IoU=0.683`, `BF=0.827`.

The optional test split run completed as an official-benchmark-aligned evaluation using the trained FPN Snake checkpoint. Refined test AP was `AP=14.0826`, `obj_AP=30.4572`, `part_AP=11.2957`. Refined test all-part boundary metrics were `IoU=0.741`, `BF=0.805`, `38543/86038` matches, with thin-part boundary metrics `IoU=0.684`, `BF=0.825`, `5443/16136` matches.

Likely reasons for weak or negative gains:

- The detector is frozen, so Snake can only adjust contours for already-detected parts; it cannot recover missed parts, fix wrong categories, or improve confidence ranking.
- The current Snake setup is shallow: one refinement step, limited FPN levels (`p2`, `p3`), fixed `64` vertices, and a short training run.
- The training target is contour alignment, while AP and greedy boundary matching are sensitive to detection presence, score ordering, mask area, and category correctness.
- Detector masks are noisy pseudo-inputs. If an initial mask is badly localized, contour-only refinement can make it worse or move it below the IoU matching threshold.
- Boundary evaluation uses greedy per-image/per-part matching at IoU >= 0.5, so small shape changes can reduce matched count even when average matched-mask shape quality is flat.

Concrete improvement directions:

- Train longer and select checkpoints using val AP/boundary metrics rather than one short run.
- Tune Snake capacity, number of steps, vertex count, FPN levels, loss weighting, and score/mask filtering.
- Add stronger feature sources such as DINO/ViT features and compare from the same detector prediction dumps.
- Train only on high-quality matched examples at first, then expand to harder examples.
- Add qualitative overlays for best/worst deltas to diagnose whether failures come from missing detections, class mismatch, mask shrink/expand artifacts, or contour instability.
- Rewrite boundary eval to stream per image so full test analysis does not require hundreds of GiB of RAM.

### 3. Guidance For Plotting
Use the plot-ready artifacts under `results/experiment_logs/run_20260601_200407`:

- `metrics_summary.csv`: AP/object AP/part AP rows for official reference, local R50-FPN, and Snake-refined outputs.
- `metrics_delta.csv`: direct before/after deltas for same-split comparisons.
- `boundary_metrics_summary.csv`: all-part, thin-part, and per-object boundary summaries.
- `boundary_delta.csv`: boundary IoU and boundary-F before/after deltas.
- `per_instance_delta.csv`: per-instance deltas for histograms, scatter plots, and best/worst examples.
- `runtime_summary.csv`: runtime and throughput by stage.
- `qualitative_examples.json`: selected examples for visual inspection if overlays are generated later.

Recommended figures:

- AP bar chart: local R50-FPN val vs Snake-refined val; show official model-zoo number separately as a reference marker, not a direct delta.
- Boundary bar chart: all-parts and thin-parts IoU/BF before vs after.
- Per-object heatmap/table: object categories with largest boundary improvements and regressions.
- Delta histogram: per-instance IoU delta and boundary-F delta.
- Scatter plot: baseline IoU vs refined IoU, colored by thin/non-thin part.
- Runtime chart: detector dump, Snake training, refinement, AP eval, and boundary eval durations.

## S3 Implementation Details

### 1. How To Run The Whole Pipeline
The authoritative runbook is `SETUP_REPORT_269contour.md`. It records the exact stage commands, output paths, status JSONs, progress JSONLs, `.done` markers, recovery notes, and test split completion details. Use it as the first source of truth before launching or resuming any run.

The practical command pattern is:

```bash
cd /home/jade1st/pj/pacojade-rory
source /home/jade1st/miniconda3/etc/profile.d/conda.sh
conda activate 269contour
```

Then run or resume the stage command files in the log directories:

- main train/val logs: `results/experiment_logs/run_20260601_200407`
- optional test logs: `results/experiment_logs/test_20260602_144243`
- main output root: `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407`
- test output root: `/xuanwu-tank/north/jade/Paco/output/snake_fpn_test_20260602_144243`

The main stage order is:

1. Verify image coverage and environment.
2. Generate sharded train detector prediction dumps.
3. Merge train prediction shards.
4. Generate val detector prediction dump.
5. Train FPN Snake from train predictions and train GT.
6. Refine val predictions using the trained Snake checkpoint.
7. Evaluate refined val AP and baseline/refined boundary metrics.
8. Generate plot-ready CSV/JSON artifacts.
9. Optionally run test prediction/refinement/eval using the frozen trained Snake checkpoint.

The important command files are already saved in the run directories, so future users should rerun those scripts instead of reconstructing long commands from memory. The large outputs under `/xuanwu-tank/north/jade/Paco` are external experiment artifacts and should not be committed to git.

### 2. Changing From FPN To DINO
DINO is not part of the completed result. Treat it as a future feature-source replacement experiment. Do not silently switch an FPN run to DINO.

The detector prediction dumps can be reused if the detector outputs are unchanged. Start the DINO experiment from Snake feature extraction/training onward:

1. Validate `paco/snake/dino_extractor.py` and confirm it returns per-point features with the dimensions expected by `SnakeRefiner`.
2. Add or verify command support for `--feature-source dino` in `tools/train_snake_refiner.py` and `tools/run_snake_refine.py`.
3. Provide the required DINO model checkpoint/config paths and record them in `SETUP_REPORT_269contour.md`.
4. Run smoke tests on a tiny subset: import test, one-image refinement, and one small training pass.
5. Rerun Snake training with `--feature-source dino` from the existing train prediction JSON.
6. Rerun val refinement, AP eval, and boundary eval.
7. Only after the DINO val result is frozen, optionally rerun test refinement/eval.

For a fair comparison, keep detector predictions, train/val/test annotations, evaluation scripts, and plotting logic fixed. The only intended experimental variable should be the contour refiner feature source and any documented DINO-specific architecture settings.
