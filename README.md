# Image Sequence Difference Analysis

Browser-based Python application for server-side analysis of local TIFF image sequences. The image files stay on the Linux server; the browser is only used to choose datasets, edit a quadrilateral ROI, configure processing, monitor progress, and view saved results.

## Technology Stack

- **Flask**: simple Python web server with no frontend build step. It works well for remote Linux servers and keeps local file access on the server.
- **Plain JavaScript canvas**: reliable four-corner ROI editing in the browser without a desktop GUI or image uploads.
- **tifffile + imagecodecs**: direct TIFF reading, including common compressed TIFFs.
- **NumPy + pandas**: efficient image math and tabular result output.
- **Optional PyTorch/CUDA**: GPU mode can keep decoded image tensors, reference/live reductions, difference images, ROI masks, and metric calculations on CUDA when a compatible PyTorch build is available. The intended GPU runtime is `torch==2.6.0+cu124` and `torchvision==0.21.0+cu124` for CUDA 12.4; these packages are intentionally not installed by `requirements.txt`.
- **Pillow**: preview PNGs, ROI mask rasterization, and overlay images.
- **Matplotlib**: server-side summary plots saved as PNG files.

The code is split so the difference-image algorithm can be used independently from the UI.

## Project Structure

```text
app.py                         Flask application entry point
configs/datasets.yaml          Dataset definitions
configs/roi_presets.json       Saved ROI presets
configs/algorithm_presets.json Saved algorithm presets
outputs/runs/                  Timestamped run outputs
sample_data/                   Optional generated local TIFF test data
scripts/generate_test_data.py  Synthetic TIFF sequence generator
src/diff_image_analysis/       Processing, indexing, ROI, plotting, config modules
web/templates/                 HTML UI
web/static/                    JavaScript and CSS
tests/smoke_test.py            Synthetic 3x3 and 9x9 end-to-end check
```

## Setup

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure Datasets

Edit `configs/datasets.yaml`. Each dataset lists one or more folders that directly contain `.tif` or `.tiff` files.

```yaml
datasets:
  - name: "dataset_01"
    folders:
      - "/data/images/phase_01"
      - "/data/images/phase_02"
    start_time: "2025-01-01 08:00:00"
    end_time: "2025-01-08 18:00:00"
    description: "First recording block"
    timestamp_regex: "(\\d{8}_\\d{6})"
    timestamp_format: "%Y%m%d_%H%M%S"
```

Only the configured folders are scanned. Subfolders are ignored, and there is no recursive scan option.

## Generate Test Data

The config includes a local `synthetic_test` dataset. Generate its TIFF files with:

```bash
source .venv/bin/activate
python scripts/generate_test_data.py
```

This writes 240 timestamped grayscale TIFF files into:

```text
sample_data/synthetic_test/phase_01/
sample_data/synthetic_test/phase_02/
```

The sequence contains noise, slow brightness drift, a moving bright region, and a short local event so the difference metrics produce visible changes. After generation, restart the app if it was already running, select `synthetic_test`, click **Index Dataset**, choose the time range, and click **Confirm Dataset**.

The repository also includes presets for a quick demo run:

- ROI preset: `synthetic_demo_roi`
- Algorithm preset: `synthetic_quick_5x5`
- Algorithm preset for full-quality GPU runs: `v100_full_quality`

The Algorithm step now supports two modes:

- `difference`: the existing reference-vs-live difference metrics workflow
- `tile_statistics`: scans the selected time range, computes per-tile intensity summaries, and writes one CSV row with `cell_xxx_mean` and `cell_xxx_median` columns

For a demo run, select `synthetic_test`, click **Index Dataset**, use the complete available range, click **Confirm Dataset**, apply `synthetic_demo_roi`, apply `synthetic_quick_5x5`, and run the algorithm. Use **Reset All** in the header to clear the current UI configuration and return to the first workflow step.

## Start the App

```bash
source .venv/bin/activate
python app.py
```

The app listens on `0.0.0.0:8050` by default.

## Run Analysis

1. Open the app in a browser.
2. Use the left workflow navigation: **Dataset**, **ROI**, **Algorithm**, then **Run**.
3. In **Dataset**, select a dataset and click **Index Dataset**. Then choose the complete available range or a custom start/end timestamp and click **Confirm Dataset**. Custom timestamp controls are only shown when **Custom range** is selected; after indexing, their date options and minute-based time groups come from parsed image timestamps, so unavailable days and times are not selectable. End-minute selections include all images through that selected minute. After confirmation, the UI opens **ROI** and loads the first image automatically.
4. In **ROI**, move the four ROI corner points on the preview image or apply an ROI preset, then click **Confirm ROI**.
5. Choose `grid_size`, default `3`. A `9` value creates an 81-cell ROI-following grid.
6. Open **ROI presets** only when you want to load, save, or overwrite presets.
7. In **Algorithm**, choose the algorithm type first. `difference` exposes the full reference-image configuration. `tile_statistics` exposes only `processing_stride_images`, `live_average_size_images`, and `smoothing_window_images`, then click **Confirm Algorithm**.
8. In **Run**, choose **GPU (CUDA)** or **CPU**. GPU is selected by default. Use **Check GPU** to verify PyTorch/CUDA availability; if GPU is selected and CUDA is unavailable, the run returns an error instead of silently falling back to CPU.
9. Click **Run Algorithm** and watch the progress panel. The run button stays disabled until Dataset, ROI, and Algorithm all show a green check in the workflow navigation. During a run, **Cancel Run** requests a cooperative stop; the app finishes the current processing step, writes the partial run artifacts, and marks the job as cancelled. The progress panel also shows a rough remaining-time estimate based on the recent progress rate. Open **Run log** to inspect coarse intermediate steps such as setup, cache initialization, mask preparation, reference refreshes, 5% progress marks, output writing, and final cache stats.

The Flask process keeps one shared in-memory UI/job state. Opening the same URL in another browser window reconnects to the same selected configuration and any active run. Only one analysis run can be active at a time; a second start request is rejected until the current run finishes or is cancelled.

## ROI Presets

ROI presets are saved in `configs/roi_presets.json`. A preset contains:

- preset name
- dataset name or `global`
- source image shape
- top-left, top-right, bottom-right, bottom-left corner coordinates
- creation timestamp
- optional comment

On later starts, dataset-matching and global ROI presets are available from the ROI preset selector.

## Algorithm Presets

Algorithm presets are saved in `configs/algorithm_presets.json`. Presets include:

- reference window size
- reference gap
- live averaging size
- processing stride
- absolute difference threshold
- smoothing window
- image downscale factor
- keep `image_downscale_factor` at `1.0` for identical full-resolution analysis; lower values intentionally downscale and change the result
- mean or median reference mode
- reference refresh interval in minutes; `0` recomputes the reference for every processed image, while values such as `60` reuse a reference image for roughly one hour before rebuilding it
- decoded image cache size; this LRU cache avoids repeatedly reading the same overlapping TIFF windows from disk, using RAM in CPU mode and VRAM in GPU mode
- grid size
- compute backend, saved in run configs as `gpu` or `cpu`
- output directory
- preview image settings
- optional run name and comment

## Outputs

Each run creates a folder:

```text
outputs/runs/YYYYMMDD_HHMMSS_run_name/
```

The folder contains:

- `results.csv`
- `run_config.json`
- `dataset_config_used.json`
- `roi_config.json`
- `processing_log.txt`
- `summary_plot.png`
- optional preview images such as `roi_grid_overlay.png`, `reference_example.png`, `live_example.png`, and `diff_example.png`

`results.csv` includes timestamp, image path, dataset name, processed index, status, global metrics, variable grid-cell metrics, and summary regional metrics.

## Load Previous Results

Use the **Load Results** tab:

1. Select a run folder from `outputs/runs/`.
2. Load the run.
3. Inspect the saved run, dataset, and ROI configs.
4. Select one or more metric columns.
5. Use the interactive SVG plots to zoom into a time range by dragging across a plot. Remove a metric either by unchecking it in the metric list or by clicking the cross on its plot.
6. Click **Save Plot PNG** to generate a static PNG plot in that run folder.

The result viewer detects available metric columns from `results.csv`, so it works with `3x3`, `9x9`, or any other `grid_size x grid_size` run, including `tile_statistics` runs that only output per-cell mean/median summary columns.

## Smoke Test

Run the synthetic end-to-end check:

```bash
source .venv/bin/activate
python tests/smoke_test.py
```

It creates temporary TIFF images, runs processing with `grid_size=3` and `grid_size=9`, and verifies representative output columns and plots.
