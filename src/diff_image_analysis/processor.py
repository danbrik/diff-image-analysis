"""Reference-image difference processing independent of the browser UI."""

from __future__ import annotations

from collections import OrderedDict
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

from .compute import ComputeContext, get_compute_context, torch_module
from .config import AlgorithmConfig, DatasetConfig, dataset_config_json, format_datetime
from .image_io import load_image_float32, normalize_for_display
from .indexing import filter_records_by_time
from .metrics import (
    DEFAULT_PLOT_METRICS,
    MetricRegions,
    build_metric_regions,
    compute_difference_metrics_from_indices,
)
from .plotting import save_metrics_plot
from .roi import build_grid_masks, corners_dict_to_array, save_roi_grid_overlay, scale_corners


ProgressCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]
LogCallback = Callable[[str], None]


class RunCancelled(Exception):
    """Raised internally when a run is cancelled cooperatively."""


@dataclass(slots=True)
class RunResult:
    """Filesystem outputs created by one processing run."""

    run_folder: Path
    results_csv: Path
    summary_plot: Path
    cancelled: bool = False


@dataclass(slots=True)
class ReferenceCache:
    """Cached reference image reused until the configured refresh interval expires."""

    image: Any
    anchor_timestamp: datetime
    ref_start: int
    ref_end: int


@dataclass(slots=True)
class ComputeMetricRegions:
    """ROI/cell pixel indices prepared for the selected compute backend."""

    cpu: MetricRegions
    full_indices: Any
    cell_indices: dict[str, Any]


class ImageCache:
    """Small LRU cache for decoded images used by overlapping windows.

    CPU runs cache decoded NumPy arrays in RAM. GPU runs cache CUDA tensors in VRAM
    after TIFF decode, so reference/live/diff work can stay on the GPU.
    """

    def __init__(
        self,
        max_size: int,
        downscale_factor: float,
        compute_context: ComputeContext,
    ) -> None:
        self.max_size = max(0, int(max_size))
        self.downscale_factor = downscale_factor
        self.compute_context = compute_context
        self._images: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def load(self, path: str) -> Any:
        """Load one image, reusing the decoded array when it is still cached."""
        if self.max_size <= 0:
            self.misses += 1
            return self._load_uncached(path)
        cached = self._images.get(path)
        if cached is not None:
            self._images.move_to_end(path)
            self.hits += 1
            return cached
        self.misses += 1
        image = self._load_uncached(path)
        self._images[path] = image
        self._images.move_to_end(path)
        while len(self._images) > self.max_size:
            self._images.popitem(last=False)
        return image

    def load_many(self, paths: list[str], cancel_check: CancelCheck | None = None) -> list[Any]:
        """Load multiple images through the same LRU cache."""
        images = []
        for path in paths:
            _raise_if_cancelled(cancel_check)
            images.append(self.load(path))
        return images

    @property
    def current_size(self) -> int:
        """Number of decoded images currently held in cache."""
        return len(self._images)

    def _load_uncached(self, path: str) -> Any:
        image = load_image_float32(path, self.downscale_factor)
        if self.compute_context.backend != "gpu":
            return image
        torch = torch_module()
        return torch.as_tensor(image, dtype=torch.float32, device=self.compute_context.device)


def run_difference_analysis(
    all_records: pd.DataFrame,
    dataset: DatasetConfig,
    roi_config: dict[str, Any],
    algorithm_config: AlgorithmConfig,
    range_start: datetime | None = None,
    range_end: datetime | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    log_callback: LogCallback | None = None,
    selected_algorithm_preset: str = "",
    selected_roi_preset: str = "",
) -> RunResult:
    """Run the reference-image difference algorithm and persist run artifacts."""
    algorithm_config.validate()
    compute_context = get_compute_context(algorithm_config.compute_backend)
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
    logs.append(f"compute_backend: {compute_context.backend}")
    logs.append(f"compute_device: {compute_context.device_name or compute_context.device}")
    logs.append(
        f"reference_refresh_interval_minutes: {algorithm_config.reference_refresh_interval_minutes}"
    )
    logs.append(f"image_cache_size_images: {algorithm_config.image_cache_size_images}")
    _log_step(
        logs,
        log_callback,
        f"run folder prepared: {run_folder}",
    )
    _log_step(
        logs,
        log_callback,
        f"compute backend initialized: {compute_context.backend} ({compute_context.device_name or compute_context.device})",
    )

    missing_timestamps = int(all_records["timestamp"].isna().sum()) if not all_records.empty else 0
    if missing_timestamps:
        _log_step(logs, log_callback, f"missing timestamps in indexed dataset: {missing_timestamps}")

    timestamped_all = all_records[all_records["timestamp"].notna()].reset_index(drop=True).copy()
    selected = filter_records_by_time(timestamped_all, range_start, range_end)
    total_selected = int(len(selected))
    _log_step(
        logs,
        log_callback,
        f"selected {total_selected} timestamped images from {len(timestamped_all)} available timestamped records",
    )

    _write_json(run_folder / "run_config.json", algorithm_config.to_dict())
    _write_json(run_folder / "dataset_config_used.json", dataset_config_json(dataset))
    _write_json(run_folder / "roi_config.json", roi_config)
    _log_step(logs, log_callback, "run, dataset, and ROI configuration files written")

    if total_selected == 0:
        _log_step(logs, log_callback, "selected range contains no images; writing empty result files")
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
    metric_regions_cache: dict[tuple[int, int], ComputeMetricRegions] = {}
    reference_cache: ReferenceCache | None = None
    cancelled = False
    status_counts: dict[str, int] = {}
    image_cache = ImageCache(
        max_size=algorithm_config.image_cache_size_images,
        downscale_factor=algorithm_config.image_downscale_factor,
        compute_context=compute_context,
    )
    _log_step(
        logs,
        log_callback,
        f"image cache initialized: max_size={image_cache.max_size}, downscale={algorithm_config.image_downscale_factor}",
    )
    preview_saved = 0
    next_progress_log = 5

    try:
        _log_step(logs, log_callback, "processing loop started")
        for selected_order, selected_row in selected.iterrows():
            _raise_if_cancelled(cancel_check)
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
                "status_message": f"{status_message} on {compute_context.backend}",
                "compute_backend": compute_context.backend,
                "compute_device": compute_context.device_name or compute_context.device,
            }
            _emit(progress_callback, progress_payload)
            while progress_payload["percentage"] >= next_progress_log and next_progress_log < 100:
                _log_step(
                    logs,
                    log_callback,
                    f"progress {next_progress_log}%: selected position {selected_order}/{total_selected}, source index {source_index}",
                )
                next_progress_log += 5

            if selected_order % algorithm_config.processing_stride_images != 0:
                rows.append({**base_row, "status": "skipped_stride"})
                _count_status(status_counts, "skipped_stride")
                continue

            live_start = source_index - algorithm_config.live_average_size_images + 1
            if live_start < 0:
                rows.append({**base_row, "status": "insufficient_live_window"})
                _count_status(status_counts, "insufficient_live_window")
                continue

            ref_end = source_index - algorithm_config.reference_gap_images
            ref_start = ref_end - algorithm_config.reference_window_size_images + 1
            if ref_start < 0 or ref_end >= len(timestamped_all):
                rows.append({**base_row, "status": "insufficient_reference"})
                _count_status(status_counts, "insufficient_reference")
                continue

            try:
                live_paths = timestamped_all.iloc[live_start : source_index + 1]["image_path"].tolist()
                live_images = image_cache.load_many(live_paths, cancel_check=cancel_check)
                _raise_if_cancelled(cancel_check)
                live_image = _build_live_image(live_images, compute_context)
            except RunCancelled:
                raise
            except Exception as exc:
                status = "unreadable_live_window"
                rows.append({**base_row, "status": status})
                _count_status(status_counts, status)
                logs.append(f"{status}: index={source_index} path={image_path} error={exc}")
                continue

            try:
                if _reference_needs_refresh(
                    reference_cache,
                    timestamp,
                    algorithm_config.reference_refresh_interval_minutes,
                ):
                    ref_paths = timestamped_all.iloc[ref_start : ref_end + 1]["image_path"].tolist()
                    ref_images = image_cache.load_many(ref_paths, cancel_check=cancel_check)
                    _raise_if_cancelled(cancel_check)
                    reference_image = _build_reference_image(
                        ref_images,
                        use_median=algorithm_config.use_median_reference,
                        compute_context=compute_context,
                    )
                    reference_cache = ReferenceCache(
                        image=reference_image,
                        anchor_timestamp=timestamp,
                        ref_start=ref_start,
                        ref_end=ref_end,
                    )
                    if algorithm_config.reference_refresh_interval_minutes > 0:
                        _log_step(
                            logs,
                            log_callback,
                            "reference refreshed: "
                            f"index={source_index}, timestamp={format_datetime(timestamp)}, "
                            f"ref_start={ref_start}, ref_end={ref_end}",
                        )
                else:
                    reference_image = reference_cache.image
            except RunCancelled:
                raise
            except Exception as exc:
                status = "unreadable_reference_window"
                rows.append({**base_row, "status": status})
                _count_status(status_counts, status)
                logs.append(f"{status}: index={source_index} path={image_path} error={exc}")
                continue

            _raise_if_cancelled(cancel_check)
            if live_image.shape != reference_image.shape:
                status = "image_shape_mismatch"
                rows.append({**base_row, "status": status})
                _count_status(status_counts, status)
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
                metric_regions_cache[image_shape] = _prepare_metric_regions(
                    masks_cache[image_shape].full_mask,
                    masks_cache[image_shape].cell_masks,
                    compute_context,
                )
                _log_step(
                    logs,
                    log_callback,
                    f"ROI/grid masks prepared: shape={image_shape}, grid={algorithm_config.grid_size}x{algorithm_config.grid_size}",
                )
            masks = masks_cache[image_shape]
            metric_regions = metric_regions_cache[image_shape]
            if metric_regions.cpu.full_indices.size == 0:
                rows.append({**base_row, "status": "empty_roi"})
                _count_status(status_counts, "empty_roi")
                continue

            diff = _compute_diff(live_image, reference_image, compute_context)
            metrics = _compute_metrics(
                diff=diff,
                regions=metric_regions,
                threshold=algorithm_config.difference_threshold_abs,
                compute_context=compute_context,
            )
            rows.append({**base_row, "status": status, **metrics})

            if algorithm_config.save_preview_images and preview_saved < algorithm_config.preview_image_count:
                suffix = "" if preview_saved == 0 else f"_{preview_saved + 1}"
                reference_preview = _to_numpy(reference_image)
                live_preview = _to_numpy(live_image)
                diff_preview = _to_numpy(diff)
                _save_png(reference_preview, run_folder / f"reference_example{suffix}.png")
                _save_png(live_preview, run_folder / f"live_example{suffix}.png")
                _save_png(diff_preview, run_folder / f"diff_example{suffix}.png")
                if preview_saved == 0:
                    save_roi_grid_overlay(
                        live_preview,
                        scale_corners(corners_original, algorithm_config.image_downscale_factor),
                        algorithm_config.grid_size,
                        run_folder / "roi_grid_overlay.png",
                    )
                preview_saved += 1
                _log_step(logs, log_callback, f"preview image set saved: count={preview_saved}")
    except RunCancelled:
        cancelled = True
        _log_step(logs, log_callback, "cancel requested: stopping after last completed processing step")

    results = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_base_columns())
    for status_name, count in sorted(status_counts.items()):
        _log_step(logs, log_callback, f"status count {status_name}: {count}")
    _log_step(logs, log_callback, f"processing loop finished with {len(results)} result rows")
    results = _smooth_results(results, algorithm_config.smoothing_window_images)
    results_csv = run_folder / "results.csv"
    _log_step(logs, log_callback, f"writing results CSV: {results_csv}")
    results.to_csv(results_csv, index=False)
    summary_plot = run_folder / "summary_plot.png"
    try:
        _log_step(logs, log_callback, f"creating summary plot: {summary_plot}")
        save_metrics_plot(results, DEFAULT_PLOT_METRICS, summary_plot)
        _log_step(logs, log_callback, "summary plot saved")
    except Exception:
        logs.append("summary_plot_error:\n" + traceback.format_exc())
        _log_step(logs, log_callback, "summary plot failed; traceback written to processing log")
    _log_step(logs, log_callback, f"image cache hits: {image_cache.hits}")
    _log_step(logs, log_callback, f"image cache misses: {image_cache.misses}")
    _log_step(logs, log_callback, f"image cache final size: {image_cache.current_size}")

    final_processed = total_selected if not cancelled else min(len(rows), total_selected)
    final_percentage = 100.0 if not cancelled else float(final_processed / total_selected * 100.0)
    final_status = "cancelled" if cancelled else "finished"
    if cancelled:
        logs.append(f"run_cancelled_after_rows: {len(rows)}")
        _log_step(logs, log_callback, f"run cancelled after {len(rows)} rows")
    else:
        _log_step(logs, log_callback, "run finished successfully")

    _emit(
        progress_callback,
        {
            "dataset_name": dataset.name,
            "total_images": total_selected,
            "processed_images": final_processed,
            "current_image_index": int(selected.iloc[min(max(final_processed - 1, 0), total_selected - 1)]["source_index"]),
            "current_timestamp": format_datetime(
                selected.iloc[min(max(final_processed - 1, 0), total_selected - 1)]["timestamp"].to_pydatetime()
            ),
            "percentage": final_percentage,
            "status_message": final_status,
            "compute_backend": compute_context.backend,
            "compute_device": compute_context.device_name or compute_context.device,
        },
    )
    _finish_log(run_folder, logs, start_time)
    return RunResult(
        run_folder=run_folder,
        results_csv=results_csv,
        summary_plot=summary_plot,
        cancelled=cancelled,
    )


def _reference_needs_refresh(
    cache: ReferenceCache | None,
    timestamp: datetime,
    refresh_interval_minutes: float,
) -> bool:
    """Return True when the reference image should be rebuilt for this timestamp."""
    if cache is None or refresh_interval_minutes <= 0:
        return True
    elapsed_minutes = (timestamp - cache.anchor_timestamp).total_seconds() / 60.0
    return elapsed_minutes >= refresh_interval_minutes


def _build_live_image(images: list[Any], compute_context: ComputeContext) -> Any:
    """Build the live image on the selected compute backend."""
    if len(images) == 1:
        if compute_context.backend == "gpu":
            return images[-1]
        return images[-1].astype(np.float32, copy=False)
    return _reduce_images(images, use_median=False, compute_context=compute_context)


def _build_reference_image(
    images: list[Any],
    use_median: bool,
    compute_context: ComputeContext,
) -> Any:
    """Build a mean or median reference image on the selected compute backend."""
    return _reduce_images(images, use_median=use_median, compute_context=compute_context)


def _reduce_images(
    images: list[Any],
    use_median: bool,
    compute_context: ComputeContext,
) -> Any:
    if compute_context.backend == "gpu":
        torch = torch_module()
        with torch.no_grad():
            tensor_stack = torch.stack(images, dim=0)
            if use_median:
                reduced = torch.median(tensor_stack, dim=0).values
            else:
                reduced = torch.mean(tensor_stack, dim=0)
            return reduced

    stack = np.stack(images, axis=0).astype(np.float32, copy=False)
    if use_median:
        return np.median(stack, axis=0).astype(np.float32)
    return np.mean(stack, axis=0).astype(np.float32)


def _compute_diff(
    live_image: Any,
    reference_image: Any,
    compute_context: ComputeContext,
) -> Any:
    """Compute absolute difference on the selected compute backend."""
    if compute_context.backend == "gpu":
        torch = torch_module()
        with torch.no_grad():
            return torch.abs(live_image - reference_image)
    return np.abs(live_image.astype(np.float32) - reference_image.astype(np.float32))


def _prepare_metric_regions(
    full_mask: np.ndarray,
    cell_masks: dict[str, np.ndarray],
    compute_context: ComputeContext,
) -> ComputeMetricRegions:
    """Prepare ROI/cell index arrays on CPU or GPU once per image shape."""
    cpu_regions = build_metric_regions(full_mask, cell_masks)
    if compute_context.backend != "gpu":
        return ComputeMetricRegions(
            cpu=cpu_regions,
            full_indices=cpu_regions.full_indices,
            cell_indices=cpu_regions.cell_indices,
        )

    torch = torch_module()
    full_indices = torch.as_tensor(
        cpu_regions.full_indices,
        dtype=torch.long,
        device=compute_context.device,
    )
    cell_indices = {
        name: torch.as_tensor(indices, dtype=torch.long, device=compute_context.device)
        for name, indices in cpu_regions.cell_indices.items()
    }
    return ComputeMetricRegions(cpu=cpu_regions, full_indices=full_indices, cell_indices=cell_indices)


def _compute_metrics(
    diff: Any,
    regions: ComputeMetricRegions,
    threshold: float,
    compute_context: ComputeContext,
) -> dict[str, float]:
    """Compute metrics on CPU or GPU and return plain Python floats."""
    if compute_context.backend == "gpu":
        return _compute_metrics_torch(diff, regions, threshold)
    return compute_difference_metrics_from_indices(
        diff=diff,
        full_indices=regions.full_indices,
        cell_indices=regions.cell_indices,
        threshold=threshold,
    )


def _compute_metrics_torch(diff: Any, regions: ComputeMetricRegions, threshold: float) -> dict[str, float]:
    """Compute global and per-cell metrics on CUDA tensors."""
    flat_diff = diff.reshape(-1)
    metrics = _torch_basic_metrics(flat_diff[regions.full_indices], threshold, prefix="")

    cell_p95_values: list[float] = []
    affected_count = 0
    for name, indices in regions.cell_indices.items():
        cell_values = flat_diff[indices]
        cell_metrics = _torch_basic_cell_metrics(cell_values, threshold)
        for key, value in cell_metrics.items():
            metrics[f"{name}_{key}"] = value
        p95 = cell_metrics["p95_abs_diff"]
        if np.isfinite(p95):
            cell_p95_values.append(float(p95))
            if p95 > threshold:
                affected_count += 1

    if cell_p95_values:
        sorted_p95 = sorted(cell_p95_values, reverse=True)
        metrics["max_cell_p95_abs_diff"] = sorted_p95[0]
        metrics["top2_cell_p95_abs_diff_mean"] = float(np.mean(sorted_p95[:2]))
    else:
        metrics["max_cell_p95_abs_diff"] = float("nan")
        metrics["top2_cell_p95_abs_diff_mean"] = float("nan")
    total_cells = max(1, len(regions.cell_indices))
    metrics["affected_cell_count"] = float(affected_count)
    metrics["affected_cell_ratio"] = float(affected_count / total_cells)
    return metrics


def _torch_basic_metrics(values: Any, threshold: float, prefix: str) -> dict[str, float]:
    """Return the same metric set as the NumPy path for a CUDA tensor."""
    torch = torch_module()
    if values.numel() == 0:
        return _nan_basic_metrics(prefix)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return _nan_basic_metrics(prefix)
    quantiles = torch.quantile(
        finite,
        torch.tensor([0.95, 0.99], dtype=torch.float32, device=finite.device),
    )
    return {
        f"{prefix}mean_abs_diff": float(torch.mean(finite).item()),
        f"{prefix}median_abs_diff": float(torch.median(finite).item()),
        f"{prefix}p95_abs_diff": float(quantiles[0].item()),
        f"{prefix}p99_abs_diff": float(quantiles[1].item()),
        f"{prefix}max_abs_diff": float(torch.max(finite).item()),
        f"{prefix}area_ratio_above_threshold": float(torch.mean((finite > threshold).float()).item()),
    }


def _torch_basic_cell_metrics(values: Any, threshold: float) -> dict[str, float]:
    basic = _torch_basic_metrics(values, threshold, prefix="")
    return {
        "mean_abs_diff": basic["mean_abs_diff"],
        "p95_abs_diff": basic["p95_abs_diff"],
        "area_ratio_above_threshold": basic["area_ratio_above_threshold"],
    }


def _nan_basic_metrics(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}mean_abs_diff": float("nan"),
        f"{prefix}median_abs_diff": float("nan"),
        f"{prefix}p95_abs_diff": float("nan"),
        f"{prefix}p99_abs_diff": float("nan"),
        f"{prefix}max_abs_diff": float("nan"),
        f"{prefix}area_ratio_above_threshold": float("nan"),
    }


def _to_numpy(image: Any) -> np.ndarray:
    """Return a CPU float32 array for preview/output helpers."""
    if isinstance(image, np.ndarray):
        return image.astype(np.float32, copy=False)
    return image.detach().cpu().numpy().astype(np.float32, copy=False)


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


def _log_step(logs: list[str], callback: LogCallback | None, message: str) -> None:
    line = f"{format_datetime(datetime.now())} step: {message}"
    logs.append(line)
    if callback is not None:
        callback(line)


def _count_status(status_counts: dict[str, int], status: str) -> None:
    status_counts[status] = status_counts.get(status, 0) + 1


def _raise_if_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and cancel_check():
        raise RunCancelled


def _finish_log(run_folder: Path, logs: list[str], start_time: datetime) -> None:
    end_time = datetime.now()
    logs.append(f"run_end: {format_datetime(end_time)}")
    logs.append(f"duration_seconds: {(end_time - start_time).total_seconds():.3f}")
    with (run_folder / "processing_log.txt").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(logs))
        fh.write("\n")
