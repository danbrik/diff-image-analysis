"""TIFF loading and browser preview helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile


def load_image_float32(path: str | Path, downscale_factor: float = 1.0) -> np.ndarray:
    """Load a TIFF as grayscale float32, optionally downscaling after conversion."""
    image = tifffile.imread(str(path))
    gray = to_grayscale_float32(image)
    if downscale_factor != 1.0:
        gray = resize_float32(gray, downscale_factor)
    return gray.astype(np.float32, copy=False)


def to_grayscale_float32(image: np.ndarray) -> np.ndarray:
    """Convert grayscale or multichannel image arrays to 2-D float32."""
    arr = np.asarray(image)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            return arr[..., :3].astype(np.float32).mean(axis=-1)
        if arr.shape[0] in (3, 4):
            return arr[:3, ...].astype(np.float32).mean(axis=0)
        return arr[0, ...].astype(np.float32, copy=False)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def resize_float32(image: np.ndarray, factor: float) -> np.ndarray:
    """Resize a 2-D float32 image with bilinear interpolation."""
    if factor <= 0:
        raise ValueError("downscale factor must be > 0")
    height, width = image.shape
    new_size = (max(1, int(round(width * factor))), max(1, int(round(height * factor))))
    pil = Image.fromarray(image.astype(np.float32), mode="F")
    resized = pil.resize(new_size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Normalize a grayscale image to uint8 for browser previews and PNG output."""
    arr = image.astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


def preview_png_bytes(path: str | Path, max_dimension: int = 1400) -> tuple[bytes, dict[str, object]]:
    """Load an image and return a display PNG plus original/preview shape metadata."""
    image = load_image_float32(path, downscale_factor=1.0)
    original_height, original_width = image.shape
    scale = min(1.0, max_dimension / max(original_height, original_width))
    preview = resize_float32(image, scale) if scale < 1.0 else image
    preview_uint8 = normalize_for_display(preview)
    png = Image.fromarray(preview_uint8, mode="L")
    bio = BytesIO()
    png.save(bio, format="PNG")
    meta = {
        "original_shape": [int(original_height), int(original_width)],
        "preview_shape": [int(preview_uint8.shape[0]), int(preview_uint8.shape[1])],
        "scale": float(scale),
    }
    return bio.getvalue(), meta

