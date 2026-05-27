"""Matplotlib plotting helpers for result CSV files."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

_mpl_config_dir = Path(tempfile.gettempdir()) / "diff_image_analysis_matplotlib"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))
_xdg_cache_dir = Path(tempfile.gettempdir()) / "diff_image_analysis_cache"
_xdg_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_cache_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .metrics import DEFAULT_PLOT_METRICS


def save_metrics_plot(
    results: pd.DataFrame,
    metrics: list[str] | None,
    output_path: str | Path,
    title: str = "Difference-image metrics",
) -> list[str]:
    """Save a timestamped time-series plot for selected metric columns."""
    metrics = metrics or DEFAULT_PLOT_METRICS
    available = [metric for metric in metrics if metric in results.columns]
    if not available:
        available = _numeric_metric_columns(results)[:6]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not available:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.text(0.5, 0.5, "No plottable metric columns", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return []

    fig, axes = plt.subplots(len(available), 1, figsize=(12, max(3, 2.2 * len(available))), sharex=True)
    if len(available) == 1:
        axes = [axes]

    timestamps = pd.to_datetime(results.get("timestamp"), errors="coerce")
    for ax, metric in zip(axes, available):
        values = pd.to_numeric(results[metric], errors="coerce")
        valid = timestamps.notna() & values.notna()
        filtered_timestamps = timestamps[valid]
        filtered_values = values[valid]
        if filtered_values.empty:
            ax.text(0.5, 0.5, "No plottable values", ha="center", va="center", transform=ax.transAxes)
        else:
            marker = "o" if len(filtered_values) <= 200 else None
            markersize = 2.5 if marker else None
            ax.plot(filtered_timestamps, filtered_values, linewidth=1.4, marker=marker, markersize=markersize)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
    axes[0].set_title(title)
    axes[-1].set_xlabel("timestamp")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return available


def _numeric_metric_columns(results: pd.DataFrame) -> list[str]:
    ignored = {"timestamp", "image_path", "dataset_name", "processed_index", "status"}
    return [col for col in results.columns if col not in ignored and pd.api.types.is_numeric_dtype(results[col])]
