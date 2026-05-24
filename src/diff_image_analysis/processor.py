"""Reference-image difference processing independent of the browser UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import traceback
from typing import Callable, Any

import numpy as np
import pandas as pd
from PIL import Image

from .config import AlgorithmConfig, DatasetConfig, dataset_config_json, format_datetime
from .image_io import load_image_float32, normalize_for_display
from .indexing import filter_records_by_time
from .metrics import DEFAULT_PLOT_METRICS, compute_difference_metrics
from .plotting import save_metrics_plot
from .roi import build_grid_masks, corners_dict_to_array, save_roi_grid_overlay, scale_corners


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class RunResult:
    """Filesystem outputs created by one processing run."""

    run_folder: Path
    results_csv: Path
    summary_plot: Path


def run_difference_analysis(
    all_records: pd.DataFrame,
    dataset: DatasetConfig,
    roi_config: dict[str, Any],
    algorithm_config: AlgorithmConfig,
    range_start: datetime | None = None,
    range_end: datetime | None = None,
    progress_callback: ProgressCallback | None = None,
    selected_algorithm_preset: str = "",
    selected_roi_preset: str = "",
) -> RunResult:
    """Run the reference-image difference algorithm and persist run artifacts."""
    algorithm_config.validate()
    output_root = Path(algorithm_config.output_directory)
    run_folder = _make_run_folder(output_root, algorithm_config.run_name)
    run_folder.mkdir(parents=True, exist_ok=True)

    logs: list[str] = []
    start_time = datetime.now()
    logs.append(f"run_start: {format_datetime(start_time)}")
    logs.append(f"dataset: {dataset.name}")
    logs.append(f"algorithm_preset: {selected_algorithm_preset or '(unspecified)'}")
    logs.append(f"roi_preset: {selected_roi_preset or '(unspecified)'}")
    logs.append(f"range_start: {format_datetime(range_start)}")
    logs.append(f"range_end: {format_datetime(range_end)}")

    missing_timestamps = int(all_records["timestamp"].isna().sum()) if not all_records.empty else 0
    if missing_timestamps:
        logs.append(f"missing_timestamps: {missing_timestamps}")

    timestamped_all = all_records[all_records["timestamp"].notna()].reset_index(drop=True).copy()
    selected = filter_records_by_time(timestamped_all, range_start, range_end)
    total_selected = int(len(selected))

    _write_json(run_folder / "run_config.json", algorithm_config.to_dict())
    _write_json(run_folder / "dataset_config_used.json", dataset_config_json(dataset))
    _write_json(run_folder / "roi_config.json", roi_config)

    if total_selected == 0:
        results = pd.DataFrame(columns=_base_columns())
        results_csv = run_folder / "results.csv"
        results.to_csv(results_csv, index=False)
        summary_plot = run_folder / "summary_plot.png"
        save_metrics_plot(results, DEFAULT_PLOT_METRICS, summary_plot)
        _finish_log(run_folder, logs, start_time)
        return RunResult(run_folder=run_folder, results_csv=results_csv, summary_plot=summary_plot)

    corners_original = corners_dict_to_array(roi_config["corners"])
    rows: list[dict[str, Any]] = []
    masks_cache: dict[tuple[int, int], Any] = {}
    preview_saved = 0

    for selected_order, selected_row in selected.iterrows():
        source_index = int(selected_row["source_index"])
        timestamp = selected_row["timestamp"].to_pydatetime()
        image_path = str(selected_row["image_path"])
        base_row = {
            "timestamp": format_datetime(timestamp),
            "image_path": image_path,
            "dataset_name": dataset.name,
            "processed_index": source_index,
        }

        status = "ok"
        status_message = "processing"
        progress_payload = {
            "dataset_name": dataset.name,
            "total_images": total_selected,
            "processed_images": int(selected_order),
            "current_image_index": source_index,
            "current_timestamp": format_datetime(timestamp),
            "percentage": float(selected_order / total_selected * 100.0),
            "status_message": status_message,
        }
        _emit(progress_callback, progress_payload)

        if selected_order % algorithm_config.processing_stride_images != 0:
            rows.append({**base_row, "status": "skipped_stride"})
            logs.append(f"skipped_stride: index={source_index} path={image_path}")
            continue

        live_start = source_index - algorithm_config.live_average_size_images + 1
        if live_start < 0:
            rows.append({**base_row, "status": "insufficient_live_window"})
            logs.append(f"insufficient_live_window: index={source_index}")
            continue

        ref_end = source_index - algorithm_config.reference_gap_images
        ref_start = ref_end - algorithm_config.reference_window_size_images + 1
        if ref_start < 0 or ref_end >= len(timestamped_all):
            rows.append({**base_row, "status": "insufficient_reference"})
            logs.append(f"insufficient_reference: index={source_index}")
            continue

        try:
            live_paths = timestamped_all.iloc[live_start : source_index + 1]["image_path"].tolist()
            live_images = [
                load_image_float32(path, algorithm_config.image_downscale_factor)
                for path in live_paths
            ]
            live_image = live_images[-1] if len(live_images) == 1 else np.mean(live_images, axis=0)
        except Exception as exc:
            status = "unreadable_live_window"
            rows.append({**base_row, "status": status})
            logs.append(f"{status}: index={source_index} path={image_path} error={exc}")
            continue

        try:
            ref_paths = timestamped_all.iloc[ref_start : ref_end + 1]["image_path"].tolist()
            ref_images = [
                load_image_float32(path, algorithm_config.image_downscale_factor)
                for path in ref_paths
            ]
            ref_stack = np.stack(ref_images, axis=0)
            if algorithm_config.use_median_reference:
                reference_image = np.median(ref_stack, axis=0).astype(np.float32)
            else:
                reference_image = np.mean(ref_stack, axis=0).astype(np.float32)
        except Exception as exc:
            status = "unreadable_reference_window"
            rows.append({**base_row, "status": status})
            logs.append(f"{status}: index={source_index} path={image_path} error={exc}")
            continue

        if live_image.shape != reference_image.shape:
            status = "image_shape_mismatch"
            rows.append({**base_row, "status": status})
            logs.append(
                f"{status}: index={source_index} live_shape={live_image.shape} "
                f"reference_shape={reference_image.shape}"
            )
            continue

        image_shape = (int(live_image.shape[0]), int(live_image.shape[1]))
        if image_shape not in masks_cache:
            scaled_corners = scale_corners(corners_original, algorithm_config.image_downscale_factor)
            masks_cache[image_shape] = build_grid_masks(
                image_shape=image_shape,
                corners=scaled_corners,
                grid_size=algorithm_config.grid_size,
            )
        masks = masks_cache[image_shape]
        if not masks.full_mask.any():
            rows.append({**base_row, "status": "empty_roi"})
            logs.append(f"empty_roi: index={source_index}")
            continue

        diff = np.abs(live_image.astype(np.float32) - reference_image.astype(np.float32))
        metrics = compute_difference_metrics(
            diff=diff,
            full_mask=masks.full_mask,
            cell_masks=masks.cell_masks,
            threshold=algorithm_config.difference_threshold_abs,
        )
        rows.append({**base_row, "status": status, **metrics})

        if algorithm_config.save_preview_images and preview_saved < algorithm_config.preview_image_count:
            suffix = "" if preview_saved == 0 else f"_{preview_saved + 1}"
            _save_png(reference_image, run_folder / f"reference_example{suffix}.png")
            _save_png(live_image, run_folder / f"live_example{suffix}.png")
            _save_png(diff, run_folder / f"diff_example{suffix}.png")
            if preview_saved == 0:
                save_roi_grid_overlay(
                    live_image,
                    scale_corners(corners_original, algorithm_config.image_downscale_factor),
                    algorithm_config.grid_size,
                    run_folder / "roi_grid_overlay.png",
                )
            preview_saved += 1

    results = pd.DataFrame(rows)
    results = _smooth_results(results, algorithm_config.smoothing_window_images)
    results_csv = run_folder / "results.csv"
    results.to_csv(results_csv, index=False)
    summary_plot = run_folder / "summary_plot.png"
    try:
        save_metrics_plot(results, DEFAULT_PLOT_METRICS, summary_plot)
    except Exception:
        logs.append("summary_plot_error:\n" + traceback.format_exc())

    _emit(
        progress_callback,
        {
            "dataset_name": dataset.name,
            "total_images": total_selected,
            "processed_images": total_selected,
            "current_image_index": int(selected.iloc[-1]["source_index"]),
            "current_timestamp": format_datetime(selected.iloc[-1]["timestamp"].to_pydatetime()),
            "percentage": 100.0,
            "status_message": "finished",
        },
    )
    _finish_log(run_folder, logs, start_time)
    return RunResult(run_folder=run_folder, results_csv=results_csv, summary_plot=summary_plot)


def _smooth_results(results: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 1 or results.empty:
        return results
    ignored = {"processed_index"}
    numeric_cols = [
        col
        for col in results.columns
        if col not in ignored and pd.api.types.is_numeric_dtype(results[col])
    ]
    smoothed = results.copy()
    for col in numeric_cols:
        smoothed[col] = smoothed[col].rolling(window=window, min_periods=1).mean()
    return smoothed


def _make_run_folder(output_root: Path, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = _slugify(run_name.strip()) if run_name.strip() else "run"
    folder = output_root / f"{stamp}_{safe_name}"
    counter = 2
    while folder.exists():
        folder = output_root / f"{stamp}_{safe_name}_{counter}"
        counter += 1
    return folder


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return cleaned or "run"


def _base_columns() -> list[str]:
    return ["timestamp", "image_path", "dataset_name", "processed_index", "status"]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def _save_png(image: np.ndarray, path: Path) -> None:
    Image.fromarray(normalize_for_display(image), mode="L").save(path)


def _emit(callback: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if callback is not None:
        callback(payload)


def _finish_log(run_folder: Path, logs: list[str], start_time: datetime) -> None:
    end_time = datetime.now()
    logs.append(f"run_end: {format_datetime(end_time)}")
    logs.append(f"duration_seconds: {(end_time - start_time).total_seconds():.3f}")
    with (run_folder / "processing_log.txt").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(logs))
        fh.write("\n")
