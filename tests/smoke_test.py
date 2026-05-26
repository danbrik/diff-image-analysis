"""Synthetic end-to-end smoke test for 3x3 and 9x9 processing."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diff_image_analysis.config import AlgorithmConfig, DatasetConfig  # noqa: E402
from diff_image_analysis.indexing import index_dataset  # noqa: E402
from diff_image_analysis.processor import run_difference_analysis  # noqa: E402
from diff_image_analysis.roi import build_grid_masks  # noqa: E402


def main() -> None:
    temp_root = Path(tempfile.mkdtemp(prefix="diff-image-analysis-"))
    try:
        image_dir = temp_root / "images"
        image_dir.mkdir()
        start = datetime(2025, 1, 1, 8, 0, 0)
        for idx in range(18):
            timestamp = start + timedelta(minutes=idx)
            base = np.full((64, 80), idx, dtype=np.float32)
            base[20:45, 25:55] += idx * 3
            tifffile.imwrite(image_dir / f"frame_{timestamp:%Y%m%d_%H%M%S}.tif", base.astype(np.uint16))

        dataset = DatasetConfig(
            name="synthetic",
            folders=[str(image_dir)],
            start_time="2025-01-01 08:00:00",
            end_time="2025-01-01 08:17:00",
            description="Synthetic smoke test",
            timestamp_regex=r"(\d{8}_\d{6})",
            timestamp_format="%Y%m%d_%H%M%S",
        )
        records = index_dataset(dataset)
        assert len(records) == 18
        corners = {
            "top_left": [10, 10],
            "top_right": [70, 8],
            "bottom_right": [72, 55],
            "bottom_left": [8, 58],
        }

        masks_3 = build_grid_masks((64, 80), corners, 3)
        masks_9 = build_grid_masks((64, 80), corners, 9)
        assert len(masks_3.cell_masks) == 9
        assert len(masks_9.cell_masks) == 81
        assert "cell_009" in masks_3.cell_masks
        assert "cell_081" in masks_9.cell_masks

        for grid_size, expected_cell in [(3, "cell_009"), (9, "cell_081")]:
            cfg = AlgorithmConfig(
                reference_window_size_images=3,
                reference_gap_images=1,
                live_average_size_images=1,
                processing_stride_images=1,
                difference_threshold_abs=2.0,
                smoothing_window_images=1,
                image_downscale_factor=1.0,
                use_median_reference=False,
                reference_refresh_interval_minutes=0.0 if grid_size == 3 else 5.0,
                image_cache_size_images=8,
                grid_size=grid_size,
                compute_backend="cpu",
                output_directory=str(temp_root / "outputs" / "runs"),
                save_preview_images=True,
                preview_image_count=1,
                run_name=f"grid_{grid_size}",
            )
            result = run_difference_analysis(
                all_records=records,
                dataset=dataset,
                roi_config={
                    "preset_name": "synthetic_roi",
                    "dataset_name": "synthetic",
                    "image_shape": [64, 80],
                    "corners": corners,
                    "created_at": "2025-01-01 08:00:00",
                    "comment": "",
                },
                algorithm_config=cfg,
            )
            output = pd.read_csv(result.results_csv)
            assert "p95_abs_diff" in output.columns
            assert f"{expected_cell}_p95_abs_diff" in output.columns
            assert (result.run_folder / "summary_plot.png").exists()
            assert (result.run_folder / "roi_grid_overlay.png").exists()
        print("smoke test passed")
    finally:
        shutil.rmtree(temp_root)


if __name__ == "__main__":
    main()
