"""Quadrilateral ROI and variable grid mask generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from .image_io import normalize_for_display


CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")


@dataclass(slots=True)
class GridMasks:
    """Full ROI mask plus named cell masks."""

    full_mask: np.ndarray
    cell_masks: dict[str, np.ndarray]
    cell_corners: dict[str, list[list[float]]]
    grid_lines: list[tuple[list[float], list[float]]]


def corners_dict_to_array(corners: dict[str, Iterable[float]]) -> np.ndarray:
    """Return corners in top-left, top-right, bottom-right, bottom-left order."""
    return np.asarray([corners[name] for name in CORNER_ORDER], dtype=np.float32)


def corners_array_to_dict(corners: np.ndarray) -> dict[str, list[float]]:
    """Convert ordered corner array to the persisted dictionary format."""
    return {name: [float(pt[0]), float(pt[1])] for name, pt in zip(CORNER_ORDER, corners)}


def scale_corners(corners: np.ndarray, factor: float) -> np.ndarray:
    """Scale x/y ROI coordinates by an image resize factor."""
    return corners.astype(np.float32) * float(factor)


def bilinear_point(corners: np.ndarray, u: float, v: float) -> np.ndarray:
    """Interpolate a point inside a quadrilateral using bilinear coordinates."""
    top_left, top_right, bottom_right, bottom_left = corners
    return (
        (1 - u) * (1 - v) * top_left
        + u * (1 - v) * top_right
        + u * v * bottom_right
        + (1 - u) * v * bottom_left
    )


def build_grid_masks(
    image_shape: tuple[int, int],
    corners: dict[str, Iterable[float]] | np.ndarray,
    grid_size: int,
) -> GridMasks:
    """Build a full ROI mask and one mask per bilinear grid cell."""
    if grid_size < 1:
        raise ValueError("grid_size must be >= 1")
    corner_array = (
        corners_dict_to_array(corners) if isinstance(corners, dict) else corners.astype(np.float32)
    )
    full_mask = polygon_mask(image_shape, corner_array)
    total_cells = grid_size * grid_size
    pad_width = max(3, len(str(total_cells)))
    cell_masks: dict[str, np.ndarray] = {}
    cell_corners: dict[str, list[list[float]]] = {}
    index = 1
    for row in range(grid_size):
        v0 = row / grid_size
        v1 = (row + 1) / grid_size
        for col in range(grid_size):
            u0 = col / grid_size
            u1 = (col + 1) / grid_size
            pts = np.asarray(
                [
                    bilinear_point(corner_array, u0, v0),
                    bilinear_point(corner_array, u1, v0),
                    bilinear_point(corner_array, u1, v1),
                    bilinear_point(corner_array, u0, v1),
                ],
                dtype=np.float32,
            )
            name = f"cell_{index:0{pad_width}d}"
            cell_masks[name] = polygon_mask(image_shape, pts) & full_mask
            cell_corners[name] = [[float(x), float(y)] for x, y in pts]
            index += 1
    return GridMasks(
        full_mask=full_mask,
        cell_masks=cell_masks,
        cell_corners=cell_corners,
        grid_lines=grid_lines(corner_array, grid_size),
    )


def polygon_mask(image_shape: tuple[int, int], corners: np.ndarray) -> np.ndarray:
    """Rasterize a polygon into a boolean mask with shape (height, width)."""
    height, width = image_shape
    image = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(image)
    draw.polygon([(float(x), float(y)) for x, y in corners], fill=1)
    return np.asarray(image, dtype=bool)


def grid_lines(corners: np.ndarray, grid_size: int) -> list[tuple[list[float], list[float]]]:
    """Return grid line endpoints for overlay drawing."""
    lines: list[tuple[list[float], list[float]]] = []
    for idx in range(grid_size + 1):
        u = idx / grid_size
        v = idx / grid_size
        vertical_start = bilinear_point(corners, u, 0.0)
        vertical_end = bilinear_point(corners, u, 1.0)
        horizontal_start = bilinear_point(corners, 0.0, v)
        horizontal_end = bilinear_point(corners, 1.0, v)
        lines.append((vertical_start.tolist(), vertical_end.tolist()))
        lines.append((horizontal_start.tolist(), horizontal_end.tolist()))
    return lines


def save_roi_grid_overlay(
    image: np.ndarray,
    corners: dict[str, Iterable[float]] | np.ndarray,
    grid_size: int,
    output_path: str | Path,
) -> None:
    """Save a readable ROI/grid overlay PNG for a run folder."""
    corner_array = (
        corners_dict_to_array(corners) if isinstance(corners, dict) else corners.astype(np.float32)
    )
    base = Image.fromarray(normalize_for_display(image), mode="L").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    polygon = [(float(x), float(y)) for x, y in corner_array]
    draw.polygon(polygon, fill=(255, 0, 0, 45), outline=(255, 40, 40, 220))
    for start, end in grid_lines(corner_array, grid_size):
        draw.line([tuple(start), tuple(end)], fill=(255, 210, 0, 210), width=1)
    for x, y in corner_array:
        radius = 4
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 255, 255, 240),
            outline=(255, 0, 0, 255),
        )
    result = Image.alpha_composite(base, overlay).convert("RGB")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)

