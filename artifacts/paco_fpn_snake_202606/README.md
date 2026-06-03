# PACO FPN Snake Experiment Artifacts

This directory contains compact artifacts for teammate presentation, plotting,
and rerunning the FPN Snake contour-refinement experiment. Large detector and
refined prediction dumps remain on `/xuanwu-tank` and are intentionally not
committed.

## Slides And Summary

- `../../FULL_EXPERIMENT_REPORT.md`: teammate-facing write-up, results, and
  implementation notes.
- `../../SETUP_REPORT_269contour.md`: full runbook with stage status, commands,
  paths, failures, resumes, and final test completion state.
- `ap/refined_val_ap.json`: PACO-LVIS val refined AP result.
- `ap/refined_test_ap.json`: PACO-LVIS test refined AP result.
- `boundary/refined_val_boundary_metrics.md`: val refined boundary summary.
- `boundary/refined_test_boundary_metrics.md`: test refined boundary summary.
- `checkpoints/snake_refiner_fpn_full_20260601_200407.pth`: trained lightweight
  FPN Snake checkpoint.

## Plot Inputs

- `plot_artifacts/metrics_summary.csv`: AP/object AP/part AP rows.
- `plot_artifacts/metrics_delta.csv`: before/after metric deltas.
- `plot_artifacts/boundary_metrics_summary.csv`: boundary IoU/F summaries.
- `plot_artifacts/boundary_delta.csv`: boundary metric deltas.
- `plot_artifacts/per_instance_delta.csv`: per-instance delta table.
- `plot_artifacts/runtime_summary.csv`: runtime by experiment stage.
- `plot_artifacts/qualitative_examples.json`: selected qualitative example index.
- `boundary/*_per_instance_metrics.json`: per-instance boundary outputs for
  more detailed plots.

## External Large Artifacts

The following files are required only for full reruns or deep audit, and should
stay in external storage:

- `/xuanwu-tank/north/jade/Paco/output/r50_full_train_20260601_200407/lvis_instances_results.json`
- `/xuanwu-tank/north/jade/Paco/output/r50_full_val_20260601_200407/lvis_instances_results.json`
- `/xuanwu-tank/north/jade/Paco/output/r50_full_test_20260602_144243/lvis_instances_results.json`
- `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407/refined_val/lvis_instances_results.json`
- `/xuanwu-tank/north/jade/Paco/output/snake_fpn_test_20260602_144243/refined_test/lvis_instances_results.json`

These JSONs are roughly 1GB to 15GB each and are not suitable for normal git.
