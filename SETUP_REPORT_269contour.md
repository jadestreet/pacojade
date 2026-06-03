# PACO rory setup report: 269contour

## 1. Repository

- Clone path: `/home/jade1st/pj/pacojade-rory`
- Remote/branch: `https://github.com/jadestreet/pacojade.git`, branch `rory`
- Latest commit hash at setup: `362bae22478e4cc2e7261c7a9bb92f772b92705b`
- Main setup artifacts: `setup_pip_list.txt`, `verify_import.log`, `verify_tests_after_snake.log`, `verify_eval.log`, `verify_boundary.log`.

## 2. Environment

- Conda env name: `269contour`
- Python version: `3.9.25`
- Key package versions:
  - `torch==1.10.2+cu113`
  - `torchvision==0.11.3+cu113`
  - `detectron2==0.6`, built from PACO pinned commit `0703e08a5f589f7503a3fbfce41309c80204eec8` with CPU ops
  - `numpy==1.23.5`
  - `scipy==1.10.1`
  - `Pillow==9.5.0`
  - `opencv-python==4.11.0.86`
  - `lvis==0.5.3`
  - `ego4d==1.7.3`
  - `submitit==1.5.4`
- Final package list: `/home/jade1st/pj/pacojade-rory/setup_pip_list.txt`

## 3. Dataset and model paths

- Storage root: `/xuanwu-tank/north/jade/Paco`
- Annotation path: `/xuanwu-tank/north/jade/Paco/paco/annotations`
- COCO image path: `/xuanwu-tank/north/jade/Paco/coco`
- PACO image path: `/xuanwu-tank/north/jade/Paco/paco/images` (unused for PACO-LVIS-only work)
- Model weight path: `/xuanwu-tank/north/jade/Paco/models/r50_fpn_lvis.pth`
- Output path: `/xuanwu-tank/north/jade/Paco/output`
- Persisted conda env vars:
  - `PACO_ANNOTATION_ROOT=/xuanwu-tank/north/jade/Paco/paco/annotations`
  - `COCO_IMAGE_ROOT=/xuanwu-tank/north/jade/Paco/coco`
  - `PACO_IMAGE_ROOT=/xuanwu-tank/north/jade/Paco/paco/images`

Downloaded/prepared PACO-LVIS-only data:

- `paco_lvis_v1.zip`: downloaded and checksum matched `02ac4edb22c251e07853e6231d69aec3fad0a180f03de2f8c880650322debc80`.
- `paco_lvis_v1_val.json`: extracted under annotation path.
- `paco_lvis_v1_val_mini.json`: generated with 423 images, 7421 annotations, 5637 part annotations.
- Mini COCO images: 423/423 downloaded, 0 missing.
- Official model-zoo baseline weight: `r50_fpn_lvis.pth`, 479 MB.

## 4. Verification results

- Import test: passed; log at `/home/jade1st/pj/pacojade-rory/verify_import.log`
- CUDA availability: `True`
- Pytest after Snake implementation: `14 passed`; log at `/home/jade1st/pj/pacojade-rory/verify_tests_after_snake.log`
- Baseline eval: passed on mini PACO-LVIS with official `r50_fpn_lvis.pth`; log at `/home/jade1st/pj/pacojade-rory/verify_eval.log`
  - `segm_AP=18.163849698178637`
  - `obj_AP=44.32006326560077`
  - `part_AP=16.13188864840047`
- Baseline boundary metrics: passed; log at `/home/jade1st/pj/pacojade-rory/verify_boundary.log`
  - all parts: mean IoU `0.745`, boundary-F `0.838`, matched/GT `2777/5637`
  - thin parts: mean IoU `0.692`, boundary-F `0.856`, matched/GT `367/975`
- Snake training smoke: passed for 1 matched instance; checkpoint at `/xuanwu-tank/north/jade/Paco/output/snake_fpn_smoke/snake_refiner.pth`
- Snake refinement smoke: passed for 1 image / 20 predictions; output at `/xuanwu-tank/north/jade/Paco/output/snake_fpn_smoke_eval/lvis_instances_results.json`
- Boundary eval on refined smoke JSON: passed; log at `/home/jade1st/pj/pacojade-rory/verify_snake_boundary_smoke.log`

## 5. Important project documents and code locations

- `README.md`: upstream PACO setup, dataset env vars, annotation download commands, and model zoo reference.
- `docs/MODEL_ZOO.md`: official PACO model zoo; selected baseline is `r50_fpn_lvis` with `configs/mask_rcnn_configs/r50_attr_fpn_100_ep.py`.
- `results/PROGRESS.md`: authoritative project state and env recipe.
- `plan.md`: not present in this branch checkout.
- FPN/Snake implementation:
  - `paco/snake/fpn_extractor.py`: frozen PACO detector + FPN feature sampling interface.
  - `paco/snake/contour.py`: mask-to-contour, fixed-K resampling, contour-to-mask/RLE utilities.
  - `paco/snake/model.py`: circular-convolution Snake refinement network.
  - `paco/snake/dino_extractor.py`: placeholder that raises `NotImplementedError`.
- New CLIs:
  - `tools/train_snake_refiner.py`: train FPN Snake while detector/backbone stay frozen.
  - `tools/run_snake_refine.py`: refine baseline prediction JSON and emit evaluator-compatible JSON.
- Existing evaluation/data scripts:
  - `tools/run_eval.py`: baseline AP eval.
  - `tools/eval_part_boundary.py`: boundary metrics.
  - `tools/make_mini_subset.py`: mini PACO-LVIS generation.
  - `tools/fetch_coco_images.py`: referenced COCO image download.
- Tests:
  - `tests/test_boundary_metrics.py`
  - `tests/test_snake_contour.py`
  - `tests/test_snake_model.py`

## 6. Next actionable step

The FPN path is now implemented and smoke-tested. The next step is to run a real Snake training job beyond the 1-instance smoke test, starting with a small capped run such as `--max-instances 100`, then evaluate the refined predictions against the full mini PACO-LVIS split. DINO should remain a placeholder until the FPN Snake baseline has a meaningful refined-vs-baseline result.

## 7. Full Experiment Runbook: run_20260601_200407

This section is the active resume log for the full PACO-LVIS FPN Snake course experiment. It must be updated at every major stage start, completion, failure, and fallback decision.

Run constants:

- Main protocol: train Snake on `paco_lvis_v1_train`, evaluate on `paco_lvis_v1_val`.
- Test policy: do not tune on test; optional test prediction dump only after final val result is frozen.
- Official baseline: cite PACO model zoo / CVPR paper; local detector predictions are Snake inputs and same-split delta baselines.
- Max GPUs: `2`.
- Preferred GPUs: `4,5`.
- Worker policy: start with `num_workers=16`; reduce to `8`, then `4` only if memory or dataloader instability occurs.
- Dataset fallback order if full is not viable after speed optimization: `full -> half -> 10000 -> 5000`.
- Reduced datasets, if needed, must be deterministic, reusable annotation JSONs under `/xuanwu-tank/north/jade/Paco/paco/annotations/derived/` with metadata and seed `269`.
- Log root: `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407`.
- Main train detector output: `/xuanwu-tank/north/jade/Paco/output/r50_full_train_20260601_200407/lvis_instances_results.json`.
- Main val detector output: `/xuanwu-tank/north/jade/Paco/output/r50_full_val_20260601_200407/lvis_instances_results.json`.
- Snake output root: `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407`.

Current state recorded at `2026-06-01T22:27:20-07:00`:

- Full train COCO image coverage is complete: `45790/45790`.
- Full val COCO image coverage is complete: `2410/2410`.
- Previous single-GPU part-only train dump used GPU `4`, log `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/detector_train_part_only_stdout_stderr.log`.
- Previous single-GPU part-only train dump reached `15400/45790` images and `2970934` predictions at about `2.822` images/s, then no Python detector process was found.
- Previous single-GPU output `/xuanwu-tank/north/jade/Paco/output/r50_full_train_20260601_200407/lvis_instances_results.json` is incomplete because there is no `.done` marker; it will be quarantined before sharded replacement.

Stage tracker:

| Stage | Status | Fallback | Start | End | PID | Command / resume | Output paths | Log / progress / done | Notes / next fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dataset_train_download` | completed | full | before `2026-06-01T20:31:00-07:00` | `2026-06-01T20:31:00-07:00` | n/a | `python tools/fetch_coco_images.py --json /xuanwu-tank/north/jade/Paco/paco/annotations/paco_lvis_v1_train.json --image-root /xuanwu-tank/north/jade/Paco/coco --workers 16` | `/xuanwu-tank/north/jade/Paco/coco` | `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/download_train_stdout_stderr.log`, `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/download_train_retry_stdout_stderr.log` | Coverage complete. |
| `dataset_val_download` | completed | full | before `2026-06-01T20:31:00-07:00` | `2026-06-01T20:31:00-07:00` | n/a | `python tools/fetch_coco_images.py --json /xuanwu-tank/north/jade/Paco/paco/annotations/paco_lvis_v1_val.json --image-root /xuanwu-tank/north/jade/Paco/coco --workers 16` | `/xuanwu-tank/north/jade/Paco/coco` | `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/download_val_stdout_stderr.log` | Coverage complete. |
| `dataset_coverage_after_download` | completed | full | `2026-06-01T20:31:00-07:00` | `2026-06-01T20:31:00-07:00` | n/a | coverage verification helper | `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/dataset_coverage_after_download.csv` | `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/dataset_coverage_after_download_stdout.txt` | Train `45790/45790`, val `2410/2410`, val mini `423/423`. |
| `train_part_prediction_dump_sharded` | completed | full | `2026-06-01T22:39:09-07:00` | `2026-06-02T00:46:54-07:00` | n/a | `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_train_part_shard_0.sh`; `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_train_part_shard_1.sh` | shard JSONs under `/xuanwu-tank/north/jade/Paco/output/r50_full_train_20260601_200407/shards/` | `train_part_shard_0.progress.jsonl`, `train_part_shard_1.progress.jsonl`, shard `.status.json`, shard `.done`; stdout logs `train_part_shard_0_stdout_stderr.log`, `train_part_shard_1_stdout_stderr.log` | Completed two shards on GPUs `4` and `5` with workers `16`. Previous incomplete single-GPU JSON quarantined as `lvis_instances_results.incomplete_single_gpu_20260601_222720.json`. |
| `train_part_prediction_merge` | completed | full | `2026-06-02T00:46:54-07:00` | `2026-06-02T01:07:21-07:00` | n/a | `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_train_part_merge.sh` | `/xuanwu-tank/north/jade/Paco/output/r50_full_train_20260601_200407/lvis_instances_results.json` | `train_part_prediction_merge_stdout_stderr.log`, `train_part_prediction_merge.done`, `train_prediction_summary.json` | Completed merged full train part-only prediction JSON. |
| `val_prediction_dump` | completed | full | `2026-06-02T01:07:22-07:00` | `2026-06-02T01:37:11-07:00` | n/a | `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_detector_val_full.sh` | `/xuanwu-tank/north/jade/Paco/output/r50_full_val_20260601_200407/lvis_instances_results.json` | `detector_val_stdout_stderr.log`, `val_prediction_dump.done` | Full val detector prediction dump completed. |
| `snake_training` | completed | full | `2026-06-02T01:37:11-07:00` | `2026-06-02T12:56:26-07:00` | n/a | `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_snake_training_full.sh` | `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407/train/snake_refiner.pth` | `snake_training_stdout_stderr.log`, `snake_training.done`, `training_curve.csv` | Initial full run failed with exit `137`; repaired with streaming prediction JSON path and resumed from Snake training without rerunning detector stages. |
| `val_refinement` | completed | full | `2026-06-02T12:56:46-07:00` | `2026-06-02T13:49:33-07:00` | n/a | `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_val_refinement_full.sh` | `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407/refined_val/lvis_instances_results.json` | `val_refinement_stdout_stderr.log`, `val_refinement.done` | Refined JSON exists: `images=2410`, `refined=708130`, `skipped=14788`. |
| `refined_ap_eval` | completed | full | `2026-06-02T13:55:49-07:00` | `2026-06-02T13:57:43-07:00` | n/a | `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_refined_ap_eval.sh` | `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407/refined_ap.json` | `refined_ap_stdout_stderr.log`, `refined_ap_eval.status.json`, `refined_ap_eval.done` | Completed after `obj_names` compatibility fix. Metrics: `AP=17.339597630871168`, `obj_AP=34.16138042642012`, `part_AP=13.623617698993762`. |
| `baseline_boundary_eval` | completed | full | `2026-06-02T13:57:43-07:00` | `2026-06-02T14:04:58-07:00` | n/a | `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_baseline_boundary_eval.sh` | `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407/baseline_val_boundary_metrics.md`, `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407/baseline_val_per_instance_metrics.json` | `baseline_boundary_stdout_stderr.log`, `baseline_boundary.progress.jsonl`, `baseline_boundary.status.json`, `baseline_boundary_eval.done` | Completed: `2410/2410` images, `10333` matches, `gt_part_masks=20945`, `pred_part_masks=466352`. |
| `refined_boundary_eval` | completed | full | `2026-06-02T14:04:58-07:00` | `2026-06-02T14:12:02-07:00` | n/a | `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_refined_boundary_eval.sh` | `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407/refined_val_boundary_metrics.md`, `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407/refined_val_per_instance_metrics.json` | `refined_boundary_stdout_stderr.log`, `refined_boundary.progress.jsonl`, `refined_boundary.status.json`, `refined_boundary_eval.done` | Refined boundary evaluation completed; use progress/status files for audit trail. |
| `plot_artifact_generation` | completed | full | `2026-06-02T14:12:02-07:00` | `2026-06-02T14:12:04-07:00` | n/a | `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_plot_artifacts.sh` | `run_metadata.json`, `runtime_summary.csv`, `gpu_timeseries.csv`, `metrics_summary.csv`, `metrics_delta.csv`, `boundary_metrics_summary.csv`, `boundary_delta.csv`, `per_instance_delta.csv`, `qualitative_examples.json` | `plot_artifact_generation_stdout_stderr.log`, `plot_artifact_generation.status.json`, `plot_artifact_generation.done` | Plot-ready artifacts generated under the run log root. |
| `FULL_EXPERIMENT_REPORT.md` | completed | full | `2026-06-02T14:12:04-07:00` | `2026-06-02T14:12:04-07:00` | n/a | write root-level report | `/home/jade1st/pj/pacojade-rory/FULL_EXPERIMENT_REPORT.md` | `final_status.md` | Root-level final report generated. |

Run events:
- `2026-06-01T22:43:32-07:00` `train_part_prediction_dump_sharded` `running`: waiting for /xuanwu-tank/north/jade/Paco/output/r50_full_train_20260601_200407/shards/lvis_instances_results.shard0.done and /xuanwu-tank/north/jade/Paco/output/r50_full_train_20260601_200407/shards/lvis_instances_results.shard1.done
- `2026-06-01T22:43:33-07:00` `train_part_prediction_dump_sharded` `running`: shard0 600/22895 rate=2.9945839798598812; shard1 600/22895 rate=3.010713721118198
- `2026-06-01T22:44:24-07:00` `train_part_prediction_dump_sharded` `running`: tmux session paco_fpn_snake_200407_shards active; shard0 python pid 1893532 on GPU 4, shard1 python pid 1893531 on GPU 5; pipeline and gpu_monitor windows started
- `2026-06-01T22:46:32-07:00` `train_part_prediction_dump_sharded` `running`: waiting for /xuanwu-tank/north/jade/Paco/output/r50_full_train_20260601_200407/shards/lvis_instances_results.shard0.done and /xuanwu-tank/north/jade/Paco/output/r50_full_train_20260601_200407/shards/lvis_instances_results.shard1.done
- `2026-06-01T22:46:33-07:00` `train_part_prediction_dump_sharded` `running`: shard0 1200/22895 rate=3.0384820215268276; shard1 1200/22895 rate=3.0384322300600006
- `2026-06-01T22:46:38-07:00` `train_part_prediction_dump_sharded` `running`: pipeline window restarted with shard retry policy: failed shard retries workers=8 then workers=4 before dataset-size fallback
- `2026-06-01T22:56:33-07:00` `train_part_prediction_dump_sharded` `running`: shard0 3000/22895 rate=3.018142732121671; shard1 3000/22895 rate=3.0283249898116478
- `2026-06-01T23:06:34-07:00` `train_part_prediction_dump_sharded` `running`: shard0 4800/22895 rate=3.022136465608145; shard1 4800/22895 rate=3.0457444991795395
- `2026-06-01T23:16:37-07:00` `train_part_prediction_dump_sharded` `running`: shard0 6600/22895 rate=3.0231553313488058; shard1 6700/22895 rate=3.0540121836197227

## 8. Running Pipeline Interaction Layer

This sidecar automation layer was added to monitor and assist the already-running `run_20260601_200407` pipeline without replacing or restarting it.

- Automation root: `/xuanwu-tank/north/jade/Paco/automation`
- Config: `/xuanwu-tank/north/jade/Paco/automation/config.sh`
- Monitor: `/xuanwu-tank/north/jade/Paco/automation/monitor_experiment.sh`
- Stage review hook: `/xuanwu-tank/north/jade/Paco/automation/codex_review_stage.sh`
- Repair hook: `/xuanwu-tank/north/jade/Paco/automation/codex_repair.sh`
- Cron installer: `/xuanwu-tank/north/jade/Paco/automation/install_cron.sh`
- README: `/xuanwu-tank/north/jade/Paco/automation/README_AUTOMATION.md`
- State directory: `/xuanwu-tank/north/jade/Paco/automation/state`
- Automation logs: `/xuanwu-tank/north/jade/Paco/automation/logs`

Policy:

- Normal progress is shell/Python monitoring only; Codex is not called.
- New stage `.done` markers trigger one read-only `codex exec` review per marker.
- `failed` or `stalled` state triggers `codex_repair.sh`, up to `3` repair attempts.
- Restart attempts are disabled by default for the current live run: `MAX_RESTART_ATTEMPTS=0`.
- The monitor adopts the current pipeline and must not launch a duplicate experiment or take extra GPUs.

Validation:

- `bash -n /xuanwu-tank/north/jade/Paco/automation/*.sh` passed.
- Installed monitor dry-run classified current state as `running`, action `none`, stage `train_part_prediction_dump_sharded`.
- Cron was previewed but not installed automatically. Install with:
  `bash /xuanwu-tank/north/jade/Paco/automation/install_cron.sh`
- `2026-06-01T23:26:37-07:00` `train_part_prediction_dump_sharded` `running`: shard0 8400/22895 rate=3.020971644466272; shard1 8500/22895 rate=3.0563468320019904
- `2026-06-01T23:36:38-07:00` `train_part_prediction_dump_sharded` `running`: shard0 10300/22895 rate=3.023577305110381; shard1 10400/22895 rate=3.059027606610209
- `2026-06-01T23:46:39-07:00` `train_part_prediction_dump_sharded` `running`: shard0 12100/22895 rate=3.0255443538297135; shard1 12200/22895 rate=3.0568015958862294
- `2026-06-01T23:56:39-07:00` `train_part_prediction_dump_sharded` `running`: shard0 13900/22895 rate=3.0196368832522262; shard1 14000/22895 rate=3.0580057870983426
- `2026-06-02T00:06:42-07:00` `train_part_prediction_dump_sharded` `running`: shard0 15600/22895 rate=3.0134848682347353; shard1 15900/22895 rate=3.054526251836329
- `2026-06-02T00:16:43-07:00` `train_part_prediction_dump_sharded` `running`: shard0 17500/22895 rate=3.0122170758452467; shard1 17700/22895 rate=3.05384971708468
- `2026-06-02T00:26:52-07:00` `train_part_prediction_dump_sharded` `running`: shard0 19300/22895 rate=3.0101927114076; shard1 19500/22895 rate=3.052127458830143
- `2026-06-02T00:36:53-07:00` `train_part_prediction_dump_sharded` `running`: shard0 21100/22895 rate=3.0110793589745164; shard1 21400/22895 rate=3.0530248794368022
- `2026-06-02T00:46:54-07:00` `train_part_prediction_dump_sharded` `completed`: both shard done markers exist
- `2026-06-02T00:46:54-07:00` `train_part_prediction_merge` `running`: command=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_train_part_merge.sh log=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/train_part_prediction_merge_stdout_stderr.log
- `2026-06-02T01:07:21-07:00` `train_part_prediction_merge` `completed`: done=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/train_part_prediction_merge.done
- `2026-06-02T01:07:22-07:00` `val_prediction_dump` `running`: command=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_detector_val_full.sh log=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/detector_val_stdout_stderr.log
- `2026-06-02T01:37:11-07:00` `val_prediction_dump` `completed`: done=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/val_prediction_dump.done
- `2026-06-02T01:37:11-07:00` `snake_training` `running`: command=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_snake_training_full.sh log=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/snake_training_stdout_stderr.log
- `2026-06-02T02:44:20-07:00` `snake_training` `failed`: Command '['bash', '/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_snake_training_full.sh']' returned non-zero exit status 137.; next fix: inspect /home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/snake_training_stdout_stderr.log
- `2026-06-02T08:23:40-07:00` `status_check` `failed_at_snake_training`: completed train shard dump, train shard merge, and full val prediction dump. Pipeline stopped at `snake_training` with exit code `137` / `Killed`; no active tmux/process was found. Remaining stages not completed: `val_refinement`, `refined_ap_eval`, `baseline_boundary_eval`, `refined_boundary_eval`, `plot_artifact_generation`, `FULL_EXPERIMENT_REPORT.md`.
- `2026-06-02T08:43:17-07:00` `automation_monitor` `repair_called`: `/xuanwu-tank/north/jade/Paco/automation/monitor_experiment.sh` classified the run as `failed`, stage `snake_training`, signature `309325a139444da5`, and called `codex_repair.sh` once. Completed detector gates remain immutable: train shard 0/1 `.done`, `train_part_prediction_merge.done`, and `val_prediction_dump.done`.
- `2026-06-02T08:52:52-07:00` `snake_training` `repair_completed`: repair report `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/repair_report_309325a139444da5.md`; automation state `/xuanwu-tank/north/jade/Paco/automation/state/resume_command.json`. Root cause: full Snake trainer eagerly loaded the 15G train prediction JSON and decoded masks, causing exit `137`. Repair changed `/home/jade1st/pj/pacojade-rory/tools/train_snake_refiner.py` to stream the prediction JSON and lazily match per image; no detector outputs were deleted or rerun.
- `2026-06-02T08:54:43-07:00` `snake_training` `running`: unified monitor started safe resume in tmux `paco_fpn_snake_200407_resume`; log `/xuanwu-tank/north/jade/Paco/automation/logs/resume_20260602_085443.log`; command starts at Snake training only: `cd /home/jade1st/pj/pacojade-rory && bash /home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_snake_training_full.sh`.
- `2026-06-02T09:01:00-07:00` `automation_monitor` `running`: monitor lock/resume handling was patched. New default lock is `/xuanwu-tank/north/jade/Paco/automation/state/monitor_v2.lock` because the old tmux resume inherited `monitor.lock`. Installed monitor dry-run now reports `classification=running`, `action=none`, `stage=snake_training`, message `safe resume command is currently running`. Future resume launches write `snake_training.done` and reconnect to `run_remaining_pipeline.py`; for the already-running resume, monitor will write `snake_training.done` and launch downstream stages once `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407/train/snake_refiner.pth` exists.
- `2026-06-02T09:02:00-07:00` `automation_cron` `installed`: installed 10-minute monitor cron entry: `*/10 * * * * flock -n /xuanwu-tank/north/jade/Paco/automation/state/cron.lock /xuanwu-tank/north/jade/Paco/automation/monitor_experiment.sh >> /xuanwu-tank/north/jade/Paco/automation/logs/cron.log 2>&1`. Routine entrypoint remains `/xuanwu-tank/north/jade/Paco/automation/monitor_experiment.sh`.
- `2026-06-02T09:03:52-07:00` `automation_monitor` `state_persisted`: fixed monitor argument handling so real monitor passes no longer send `--dry-run` to `decide_monitor_action.py`. Verified `/xuanwu-tank/north/jade/Paco/automation/state/monitor_state.json` now records `last_classification=running`, `last_stage=snake_training`, and `last_monitor_time_iso=2026-06-02T09:03:52-0700`.
- `2026-06-02T12:56:26-07:00` `snake_training` `completed`: resume checkpoint exists; done marker written by automation: /xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407/train/snake_refiner.pth
- `2026-06-02T12:56:42-07:00` `train_part_prediction_dump_sharded` `running`: waiting for /xuanwu-tank/north/jade/Paco/output/r50_full_train_20260601_200407/shards/lvis_instances_results.shard0.done and /xuanwu-tank/north/jade/Paco/output/r50_full_train_20260601_200407/shards/lvis_instances_results.shard1.done
- `2026-06-02T12:56:43-07:00` `train_part_prediction_dump_sharded` `completed`: both shard done markers exist
- `2026-06-02T12:56:44-07:00` `train_part_prediction_merge` `skipped`: done marker exists: /home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/train_part_prediction_merge.done
- `2026-06-02T12:56:45-07:00` `val_prediction_dump` `skipped`: done marker exists: /home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/val_prediction_dump.done
- `2026-06-02T12:56:45-07:00` `snake_training` `skipped`: done marker exists: /home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/snake_training.done
- `2026-06-02T12:56:46-07:00` `val_refinement` `running`: command=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_val_refinement_full.sh log=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/val_refinement_stdout_stderr.log
- `2026-06-02T13:14:00-07:00` `progress_counter_patch` `completed`: added progress/status support for remaining stages without interrupting the currently running `val_refinement`. Updated `tools/run_snake_refine.py`, `tools/eval_part_boundary.py`, `results/experiment_logs/run_20260601_200407/eval_lvis_json.py`, `results/experiment_logs/run_20260601_200407/generate_plot_artifacts.py`, `results/experiment_logs/run_20260601_200407/run_remaining_pipeline.py`, and command files for val refinement, refined AP eval, baseline/refined boundary evals, and plot artifact generation. New files, when invoked by a future/restarted stage: `val_refinement.progress.jsonl`, `val_refinement.status.json`, `refined_ap_eval.status.json`, `baseline_boundary.progress.jsonl`, `baseline_boundary.status.json`, `refined_boundary.progress.jsonl`, `refined_boundary.status.json`, `plot_artifact_generation.status.json`. Validation passed: `python -m py_compile` for modified Python scripts and `bash -n` for modified command files. Note: current active `val_refinement` process was already launched before this patch, so it will not emit the new progress files unless restarted after failure/stall.
- `2026-06-02T13:41:11-07:00` `val_refinement` `failed`: Command '['bash', '/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_val_refinement_full.sh']' returned non-zero exit status 127.; next fix: inspect /home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/val_refinement_stdout_stderr.log
- `2026-06-02T13:49:33-07:00` `val_refinement` `completed`: refined JSON exists; previous shell exit 127 was caused by command file edited while bash was still executing; continuing from existing output without rerunning refinement
- `2026-06-02T13:49:35-07:00` `train_part_prediction_dump_sharded` `running`: waiting for /xuanwu-tank/north/jade/Paco/output/r50_full_train_20260601_200407/shards/lvis_instances_results.shard0.done and /xuanwu-tank/north/jade/Paco/output/r50_full_train_20260601_200407/shards/lvis_instances_results.shard1.done
- `2026-06-02T13:49:35-07:00` `train_part_prediction_dump_sharded` `completed`: both shard done markers exist
- `2026-06-02T13:49:35-07:00` `train_part_prediction_merge` `skipped`: done marker exists: /home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/train_part_prediction_merge.done
- `2026-06-02T13:49:35-07:00` `val_prediction_dump` `skipped`: done marker exists: /home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/val_prediction_dump.done
- `2026-06-02T13:49:36-07:00` `snake_training` `skipped`: done marker exists: /home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/snake_training.done
- `2026-06-02T13:49:36-07:00` `val_refinement` `skipped`: done marker exists: /home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/val_refinement.done
- `2026-06-02T13:49:36-07:00` `refined_ap_eval` `running`: command=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_refined_ap_eval.sh log=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/refined_ap_stdout_stderr.log status=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/refined_ap_eval.status.json
- `2026-06-02T13:51:39-07:00` `refined_ap_eval` `failed`: AP eval produced LVIS AP lines but failed during PACO post-processing with `AttributeError: 'LVIS' object has no attribute 'obj_names'`. Fix: patched run-local `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/eval_lvis_json.py` to attach `obj_names = {category_id: category_name}` to the `LVIS` object before calling `_evaluate_predictions_on_lvis`.
- `2026-06-02T13:57:43-07:00` `refined_ap_eval` `completed`: rerun after compatibility fix succeeded. Output `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407/refined_ap.json`; status `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/refined_ap_eval.status.json`; metrics from stdout: `AP=17.339597630871168`, `obj_AP=34.16138042642012`, `part_AP=13.623617698993762`.
- `2026-06-02T13:57:43-07:00` `baseline_boundary_eval` `running`: command=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_baseline_boundary_eval.sh log=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/baseline_boundary_stdout_stderr.log progress=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/baseline_boundary.progress.jsonl status=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/baseline_boundary.status.json
- `2026-06-02T13:51:39-07:00` `refined_ap_eval` `failed`: Command '['bash', '/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_refined_ap_eval.sh']' returned non-zero exit status 1.; next fix: inspect /home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/refined_ap_stdout_stderr.log
- `2026-06-02T13:55:49-07:00` `train_part_prediction_dump_sharded` `running`: waiting for /xuanwu-tank/north/jade/Paco/output/r50_full_train_20260601_200407/shards/lvis_instances_results.shard0.done and /xuanwu-tank/north/jade/Paco/output/r50_full_train_20260601_200407/shards/lvis_instances_results.shard1.done
- `2026-06-02T13:55:49-07:00` `train_part_prediction_dump_sharded` `completed`: both shard done markers exist
- `2026-06-02T13:55:49-07:00` `train_part_prediction_merge` `skipped`: done marker exists: /home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/train_part_prediction_merge.done
- `2026-06-02T13:55:49-07:00` `val_prediction_dump` `skipped`: done marker exists: /home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/val_prediction_dump.done
- `2026-06-02T13:55:49-07:00` `snake_training` `skipped`: done marker exists: /home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/snake_training.done
- `2026-06-02T13:55:49-07:00` `val_refinement` `skipped`: done marker exists: /home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/val_refinement.done
- `2026-06-02T13:55:49-07:00` `refined_ap_eval` `running`: command=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_refined_ap_eval.sh log=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/refined_ap_stdout_stderr.log status=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/refined_ap_eval.status.json
- `2026-06-02T13:57:43-07:00` `refined_ap_eval` `completed`: done=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/refined_ap_eval.done
- `2026-06-02T13:57:43-07:00` `baseline_boundary_eval` `running`: command=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_baseline_boundary_eval.sh log=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/baseline_boundary_stdout_stderr.log progress=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/baseline_boundary.progress.jsonl status=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/baseline_boundary.status.json
- `2026-06-02T14:04:58-07:00` `baseline_boundary_eval` `completed`: done=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/baseline_boundary_eval.done
- `2026-06-02T14:04:58-07:00` `refined_boundary_eval` `running`: command=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_refined_boundary_eval.sh log=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/refined_boundary_stdout_stderr.log progress=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/refined_boundary.progress.jsonl status=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/refined_boundary.status.json
- `2026-06-02T14:04:43-07:00` `baseline_boundary_eval` `completed`: processed `2410/2410` images, `10333` matches, `gt_part_masks=20945`, `pred_part_masks=466352`; output `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407/baseline_val_boundary_metrics.md`; per-instance output `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407/baseline_val_per_instance_metrics.json`; progress `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/baseline_boundary.progress.jsonl`; status `/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/baseline_boundary.status.json`.
- `2026-06-02T14:04:58-07:00` `refined_boundary_eval` `running`: command=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_refined_boundary_eval.sh log=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/refined_boundary_stdout_stderr.log progress=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/refined_boundary.progress.jsonl status=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/refined_boundary.status.json.

Progress/status files for boundary-stage audit:

```bash
cat /home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/refined_boundary.status.json
tail -f /home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/refined_boundary.progress.jsonl
tail -f /xuanwu-tank/north/jade/Paco/automation/logs/after_ap_fix_20260602_135548.log
```
- `2026-06-02T14:12:02-07:00` `refined_boundary_eval` `completed`: done=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/refined_boundary_eval.done
- `2026-06-02T14:12:02-07:00` `plot_artifact_generation` `running`: command=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/command_plot_artifacts.sh log=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/plot_artifact_generation_stdout_stderr.log status=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/plot_artifact_generation.status.json
- `2026-06-02T14:12:04-07:00` `plot_artifact_generation` `completed`: done=/home/jade1st/pj/pacojade-rory/results/experiment_logs/run_20260601_200407/plot_artifact_generation.done
- `2026-06-02T14:12:04-07:00` `FULL_EXPERIMENT_REPORT.md` `completed`: /home/jade1st/pj/pacojade-rory/FULL_EXPERIMENT_REPORT.md

## 9. PACO-LVIS Test Evaluation: test_20260602_144243

This section tracks the optional official-benchmark-aligned test split evaluation. It reuses the trained FPN Snake checkpoint from `run_20260601_200407`; it does not retrain Snake and does not overwrite train/val outputs.

Run constants:

- Test annotation: `/xuanwu-tank/north/jade/Paco/paco/annotations/paco_lvis_v1_test.json`.
- Test split size before download check: `9443` images, `131899` annotations, `531` categories.
- Initial image coverage check before this run: `0/9443`; all test images must be downloaded before detector/refinement stages.
- COCO image root: `/xuanwu-tank/north/jade/Paco/coco`.
- Detector checkpoint: `/xuanwu-tank/north/jade/Paco/models/r50_fpn_lvis.pth`.
- Snake checkpoint: `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407/train/snake_refiner.pth`.
- Detector test output root: `/xuanwu-tank/north/jade/Paco/output/r50_full_test_20260602_144243`.
- Snake test output root: `/xuanwu-tank/north/jade/Paco/output/snake_fpn_test_20260602_144243`.
- Test log root: `/home/jade1st/pj/pacojade-rory/results/experiment_logs/test_20260602_144243`.
- Test pipeline script: `/home/jade1st/pj/pacojade-rory/results/experiment_logs/test_20260602_144243/run_test_pipeline.sh`.
- GPU policy: original detector/test refinement used up to `2` GPUs; after user approval on `2026-06-02`, shard1 test refinement was split across GPUs `5,6,7` while shard0 continued on GPU `4`.

Stage tracker:

| Stage | Status | Start | End | Command / script | Output paths | Log / progress / done | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `test_dataset_download` | completed | `2026-06-02T14:47:18-07:00` | `2026-06-02T15:20:37-07:00` | `python tools/fetch_coco_images.py --json /xuanwu-tank/north/jade/Paco/paco/annotations/paco_lvis_v1_test.json --image-root /xuanwu-tank/north/jade/Paco/coco --workers 16` | `/xuanwu-tank/north/jade/Paco/coco/val2017/...` | `test_dataset_download_stdout_stderr.log`, `test_dataset_download.done` | Required because initial coverage was `0/9443`; completed before detector dump. |
| `test_dataset_coverage` | completed | `2026-06-02T15:20:37-07:00` | `2026-06-02T15:20:37-07:00` | `check_image_coverage.py` | `test_dataset_coverage.csv` | `test_dataset_coverage.status.json`, `test_dataset_coverage.done` | Coverage reached `9443/9443` before GPU stages. |
| `test_prediction_dump_sharded` | completed | `2026-06-02T14:47:18-07:00` | `2026-06-02T15:20:37-07:00` | `command_test_prediction_shard_0.sh`, `command_test_prediction_shard_1.sh` | `/xuanwu-tank/north/jade/Paco/output/r50_full_test_20260602_144243/shards/lvis_instances_results.shard*.json` | `test_prediction_shard_*.progress.jsonl`, `test_prediction_shard_*.status.json`, shard `.done` | Detector dump completed: shard0 `1415987` predictions, shard1 `1416502` predictions. |
| `test_prediction_merge` | completed | `2026-06-02T15:20:37-07:00` | `2026-06-02T15:32:00-07:00` | `merge_json_arrays.py` | `/xuanwu-tank/north/jade/Paco/output/r50_full_test_20260602_144243/lvis_instances_results.json` | `test_prediction_merge_stdout_stderr.log`, `test_prediction_merge.done`, `test_prediction_summary.json` | Merge output validated as JSON array with `2832489` predictions; done/status recovered after original pipeline stopped before writing marker. |
| `test_refinement_sharded` | running | `2026-06-02T15:38:01-07:00` | pending | `command_test_refinement_shard_0.sh`, `command_test_refinement_shard_1_sub{0,1,2}of3.sh` | `/xuanwu-tank/north/jade/Paco/output/snake_fpn_test_20260602_144243/refined_test/shards/shard0/lvis_instances_results.json`, `/xuanwu-tank/north/jade/Paco/output/snake_fpn_test_20260602_144243/refined_test/shards/shard1_sub*of3/lvis_instances_results.json` | `test_refinement_shard_0.progress.jsonl`, `test_refinement_shard_1_sub*of3.progress.jsonl`, `test_refinement_shard_*.status.json`, `test_refinement_sharded_split.done` | Shard0 continues on GPU `4`; old shard1 resume was stopped and split into three deterministic subshards on GPUs `5,6,7`. |
| `test_refinement_merge` | waiting | `2026-06-02T16:48:00-07:00` | pending | `continue_after_split_refinement.sh` -> `merge_json_arrays.py` | `/xuanwu-tank/north/jade/Paco/output/snake_fpn_test_20260602_144243/refined_test/lvis_instances_results.json` | `continue_after_split_refinement_stdout_stderr.log`, `test_refinement_merge_stdout_stderr.log`, `test_refinement_merge.done` | Background continuation waits for shard0 and all three shard1 subshards before merging. |
| `test_ap_eval` | pending | pending | pending | `command_test_ap_eval.sh` | `/xuanwu-tank/north/jade/Paco/output/snake_fpn_test_20260602_144243/refined_test_ap.json` | `test_ap_eval_stdout_stderr.log`, `test_ap_eval.status.json`, `test_ap_eval.done` | Evaluates refined test predictions after merge. |
| `test_boundary_eval` | pending | pending | pending | `command_test_boundary_eval.sh` | `/xuanwu-tank/north/jade/Paco/output/snake_fpn_test_20260602_144243/refined_test_boundary_metrics.md`, `/xuanwu-tank/north/jade/Paco/output/snake_fpn_test_20260602_144243/refined_test_per_instance_metrics.json` | `test_boundary_eval_stdout_stderr.log`, `test_boundary.progress.jsonl`, `test_boundary.status.json`, `test_boundary_eval.done` | Boundary evaluation on refined test predictions. |
| `test_report_update` | pending | pending | pending | append final status to this section | `SETUP_REPORT_269contour.md` | `final_status.md` | Records final paths and metrics when test pipeline completes. |

Run events:

- `2026-06-02T14:42:43-07:00` `test_pipeline_prepared`: run-local helpers and pipeline script created under `/home/jade1st/pj/pacojade-rory/results/experiment_logs/test_20260602_144243`; no test GPU stage has completed yet.
- `2026-06-02T14:47:11-07:00` `test_pipeline_started`: tmux/background script `/home/jade1st/pj/pacojade-rory/results/experiment_logs/test_20260602_144243/run_test_pipeline.sh` started.
- `2026-06-02T14:47:18-07:00` `test_dataset_download` `running`: coverage before download was `0/9443`; download log `/home/jade1st/pj/pacojade-rory/results/experiment_logs/test_20260602_144243/test_dataset_download_stdout_stderr.log`.
- `2026-06-02T15:20:37-07:00` `test_prediction_dump_sharded` `completed`: shard0 `1415987` predictions from `4721` images; shard1 `1416502` predictions from `4722` images.
- `2026-06-02T15:27:00-07:00` `test_prediction_merge` `completed_but_marker_missing`: merged detector output `/xuanwu-tank/north/jade/Paco/output/r50_full_test_20260602_144243/lvis_instances_results.json` exists and validated as JSON array with `2832489` predictions; original pipeline stopped before writing `test_prediction_merge.done`.
- `2026-06-02T15:32:00-07:00` `test_resume_prepared`: resume script `/home/jade1st/pj/pacojade-rory/results/experiment_logs/test_20260602_144243/resume_after_prediction_merge.sh` prepared to continue from `test_refinement_sharded` without rerunning detector stages.
- `2026-06-02T15:38:01-07:00` `test_refinement_sharded` `running_after_resume`: resume tmux `paco_fpn_snake_test_20260602_144243_resume` started; shard0 uses GPU `4`, shard1 uses GPU `5`; monitor `test_refinement_shard_0.status.json` and `test_refinement_shard_1.status.json`.
- `2026-06-02T16:15:03-07:00` `test_refinement_shard_1` `rerun_started`: original shard1 failed with status temp-file collision while writing `test_refinement_shard_1.status.json`; targeted rerun started in tmux `paco_fpn_snake_test_shard1_resume_20260602_161503` using GPU `5`, independent status `/home/jade1st/pj/pacojade-rory/results/experiment_logs/test_20260602_144243/test_refinement_shard_1_resume.status.json`, progress `/home/jade1st/pj/pacojade-rory/results/experiment_logs/test_20260602_144243/test_refinement_shard_1_resume.progress.jsonl`, and log `/home/jade1st/pj/pacojade-rory/results/experiment_logs/test_20260602_144243/test_refinement_shard_1_resume_stdout_stderr.log`.
- `2026-06-02T16:29:15-07:00` `test_refinement_shard_1` `stopped_for_split`: targeted shard1 resume was stopped before final output because `run_snake_refine.py` only writes final JSON at completion; partial progress was not reusable.
- `2026-06-02T16:37:43-07:00` `test_prediction_shard1_split` `completed`: `/xuanwu-tank/north/jade/Paco/output/r50_full_test_20260602_144243/shards/lvis_instances_results.shard1.json` split into three deterministic image-id subshards, each with `1574` images and prediction counts `472200`, `472136`, `472166`; metadata `/home/jade1st/pj/pacojade-rory/results/experiment_logs/test_20260602_144243/test_prediction_shard1_subshards_metadata.json`.
- `2026-06-02T16:45:07-07:00` `test_refinement_shard_1_split` `running`: sub0/sub1/sub2 launched in tmux sessions `paco_test_refine_shard1_sub0of3`, `paco_test_refine_shard1_sub1of3`, `paco_test_refine_shard1_sub2of3` on GPUs `5,6,7`; shard0 continues separately on GPU `4`.
- `2026-06-02T16:48:00-07:00` `test_after_split_continuation` `running`: continuation tmux `paco_test_continue_after_split_20260602_1648` started; it waits for shard0 and the three shard1 subshards, then runs `test_refinement_merge`, `test_ap_eval`, `test_boundary_eval`, and appends final report pointers.

Current progress commands:

```bash
cat /home/jade1st/pj/pacojade-rory/results/experiment_logs/test_20260602_144243/test_refinement_shard_0.status.json
cat /home/jade1st/pj/pacojade-rory/results/experiment_logs/test_20260602_144243/test_refinement_shard_1_sub0of3.status.json
cat /home/jade1st/pj/pacojade-rory/results/experiment_logs/test_20260602_144243/test_refinement_shard_1_sub1of3.status.json
cat /home/jade1st/pj/pacojade-rory/results/experiment_logs/test_20260602_144243/test_refinement_shard_1_sub2of3.status.json
tail -f /home/jade1st/pj/pacojade-rory/results/experiment_logs/test_20260602_144243/continue_after_split_refinement_stdout_stderr.log
```

Test split continuation update for `test_20260602_144243`:
- `2026-06-02T18:08:11-07:00` `test_refinement_sharded_split`, `test_refinement_merge`, `test_ap_eval`, and `test_boundary_eval` completed after shard1 was split across GPUs `5,6,7`.
- Test refined output: `/xuanwu-tank/north/jade/Paco/output/snake_fpn_test_20260602_144243/refined_test/lvis_instances_results.json`.
- Test AP output: `/xuanwu-tank/north/jade/Paco/output/snake_fpn_test_20260602_144243/refined_test_ap.json`.
- Test boundary output: `/xuanwu-tank/north/jade/Paco/output/snake_fpn_test_20260602_144243/refined_test_boundary_metrics.md`.
- Test logs: `/home/jade1st/pj/pacojade-rory/results/experiment_logs/test_20260602_144243`.
