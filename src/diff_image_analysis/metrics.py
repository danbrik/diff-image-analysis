"""Metric computation for difference images."""

from __future__ import annotations

import numpy as np


GLOBAL_METRIC_NAMES = [
    "mean_abs_diff",
    "median_abs_diff",
    "p95_abs_diff",
    "p99_abs_diff",
    "max_abs_diff",
    "area_ratio_above_threshold",
]

SUMMARY_METRIC_NAMES = [
    "max_cell_p95_abs_diff",
    "top2_cell_p95_abs_diff_mean",
    "affected_cell_count",
    "affected_cell_ratio",
]

DEFAULT_PLOT_METRICS = [
    "p95_abs_diff",
    "area_ratio_above_threshold",
    "max_cell_p95_abs_diff",
    "top2_cell_p95_abs_diff_mean",
    "affected_cell_count",
    "affected_cell_ratio",
]


def compute_difference_metrics(
    diff: np.ndarray,
    full_mask: np.ndarray,
    cell_masks: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, float]:
    """Compute global, per-cell, and summary regional metrics."""
    metrics: dict[str, float] = {}
    roi_values = diff[full_mask]
    metrics.update(_basic_metrics(roi_values, threshold, prefix=""))

    cell_p95_values: list[float] = []
    affected_count = 0
    for name, mask in cell_masks.items():
        values = diff[mask]
        cell_metrics = _basic_cell_metrics(values, threshold)
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
    total_cells = max(1, len(cell_masks))
    metrics["affected_cell_count"] = float(affected_count)
    metrics["affected_cell_ratio"] = float(affected_count / total_cells)
    return metrics


def _basic_metrics(values: np.ndarray, threshold: float, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}mean_abs_diff": float("nan"),
            f"{prefix}median_abs_diff": float("nan"),
            f"{prefix}p95_abs_diff": float("nan"),
            f"{prefix}p99_abs_diff": float("nan"),
            f"{prefix}max_abs_diff": float("nan"),
            f"{prefix}area_ratio_above_threshold": float("nan"),
        }
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}mean_abs_diff": float("nan"),
            f"{prefix}median_abs_diff": float("nan"),
            f"{prefix}p95_abs_diff": float("nan"),
            f"{prefix}p99_abs_diff": float("nan"),
            f"{prefix}max_abs_diff": float("nan"),
            f"{prefix}area_ratio_above_threshold": float("nan"),
        }
    return {
        f"{prefix}mean_abs_diff": float(np.mean(finite)),
        f"{prefix}median_abs_diff": float(np.median(finite)),
        f"{prefix}p95_abs_diff": float(np.percentile(finite, 95)),
        f"{prefix}p99_abs_diff": float(np.percentile(finite, 99)),
        f"{prefix}max_abs_diff": float(np.max(finite)),
        f"{prefix}area_ratio_above_threshold": float(np.mean(finite > threshold)),
    }


def _basic_cell_metrics(values: np.ndarray, threshold: float) -> dict[str, float]:
    basic = _basic_metrics(values, threshold, prefix="")
    return {
        "mean_abs_diff": basic["mean_abs_diff"],
        "p95_abs_diff": basic["p95_abs_diff"],
        "area_ratio_above_threshold": basic["area_ratio_above_threshold"],
    }
