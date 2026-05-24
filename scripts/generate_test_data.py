"""Generate deterministic synthetic TIFF image sequences for local UI testing."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import tifffile


def generate_dataset(
    output_root: Path,
    image_count: int,
    height: int,
    width: int,
    start_time: datetime,
    interval_seconds: int,
    seed: int,
    clean_existing: bool = True,
) -> None:
    """Write a two-folder synthetic TIFF sequence with timestamped filenames."""
    rng = np.random.default_rng(seed)
    phase_01 = output_root / "phase_01"
    phase_02 = output_root / "phase_02"
    phase_01.mkdir(parents=True, exist_ok=True)
    phase_02.mkdir(parents=True, exist_ok=True)
    if clean_existing:
        for phase in (phase_01, phase_02):
            for old_file in phase.glob("*.tif"):
                old_file.unlink()
            for old_file in phase.glob("*.tiff"):
                old_file.unlink()

    yy, xx = np.mgrid[0:height, 0:width]
    background = 900 + 120 * (xx / max(1, width - 1)) + 70 * (yy / max(1, height - 1))

    for idx in range(image_count):
        timestamp = start_time + timedelta(seconds=idx * interval_seconds)
        phase = phase_01 if idx < image_count // 2 else phase_02
        drift = idx * 1.8
        flicker = 18.0 * np.sin(idx / 4.0)
        noise = rng.normal(0, 10.0, size=(height, width))

        image = background + drift + flicker + noise

        # A slowly moving structure gives the reference-difference algorithm
        # something spatially coherent to detect without needing real imagery.
        center_x = int(width * (0.28 + 0.45 * idx / max(1, image_count - 1)))
        center_y = int(height * (0.45 + 0.12 * np.sin(idx / 8.0)))
        radius_x = max(8, width // 14)
        radius_y = max(6, height // 16)
        moving_blob = ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2 <= 1
        image[moving_blob] += 180 + idx * 1.5

        # A short-lived local event creates an obvious spike in affected cells.
        if image_count * 0.52 <= idx <= image_count * 0.67:
            y0 = int(height * 0.22)
            y1 = int(height * 0.46)
            x0 = int(width * 0.58)
            x1 = int(width * 0.82)
            image[y0:y1, x0:x1] += 280

        # Dark corner vignette makes the ROI preview less visually flat.
        vignette = 1.0 - 0.18 * np.hypot((xx - width / 2) / width, (yy - height / 2) / height)
        image *= vignette

        image = np.clip(image, 0, 65535).astype(np.uint16)
        path = phase / f"synthetic_{timestamp:%Y%m%d_%H%M%S}.tif"
        tifffile.imwrite(path, image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="sample_data/synthetic_test")
    parser.add_argument("--image-count", type=int, default=240)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--start-time", default="2025-03-01 08:00:00")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete existing TIFF files in the synthetic phase folders before writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
    output_root = Path(args.output_root)
    generate_dataset(
        output_root=output_root,
        image_count=args.image_count,
        height=args.height,
        width=args.width,
        start_time=start_time,
        interval_seconds=args.interval_seconds,
        seed=args.seed,
        clean_existing=not args.keep_existing,
    )
    print(f"Wrote {args.image_count} TIFF files under {output_root}")
    print(f"First timestamp: {start_time:%Y-%m-%d %H:%M:%S}")
    end_time = start_time + timedelta(seconds=(args.image_count - 1) * args.interval_seconds)
    print(f"Last timestamp:  {end_time:%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
