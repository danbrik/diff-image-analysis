"""Configuration and preset persistence helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
import json


DATETIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M")


def parse_datetime(value: str | None) -> datetime | None:
    """Parse datetime strings used by YAML config and browser datetime inputs."""
    if not value:
        return None
    text = str(value).strip()
    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Could not parse datetime: {value}") from exc


def format_datetime(value: datetime | None) -> str | None:
    """Format datetimes for JSON responses and CSV output."""
    if value is None:
        return None
    return value.strftime("%Y-%m-%d %H:%M:%S")


@dataclass(slots=True)
class DatasetConfig:
    """A configured image dataset made of one or more direct image folders."""

    name: str
    folders: list[str]
    start_time: str
    end_time: str
    timestamp_regex: str
    timestamp_format: str
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetConfig":
        return cls(
            name=str(data["name"]),
            folders=[str(p) for p in data.get("folders", [])],
            start_time=str(data.get("start_time", "")),
            end_time=str(data.get("end_time", "")),
            timestamp_regex=str(data["timestamp_regex"]),
            timestamp_format=str(data["timestamp_format"]),
            description=str(data.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AlgorithmConfig:
    """Parameters controlling reference-image difference processing."""

    reference_window_size_images: int = 10
    reference_gap_images: int = 1
    live_average_size_images: int = 1
    processing_stride_images: int = 1
    difference_threshold_abs: float = 25.0
    smoothing_window_images: int = 1
    image_downscale_factor: float = 1.0
    use_median_reference: bool = False
    reference_refresh_interval_minutes: float = 0.0
    image_cache_size_images: int = 1024
    grid_size: int = 3
    compute_backend: str = "gpu"
    output_directory: str = "outputs/runs"
    save_preview_images: bool = True
    preview_image_count: int = 1
    run_name: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AlgorithmConfig":
        if data is None:
            return cls()
        defaults = cls()
        merged = {**asdict(defaults), **data}
        cfg = cls(
            reference_window_size_images=int(merged["reference_window_size_images"]),
            reference_gap_images=int(merged["reference_gap_images"]),
            live_average_size_images=int(merged["live_average_size_images"]),
            processing_stride_images=int(merged["processing_stride_images"]),
            difference_threshold_abs=float(merged["difference_threshold_abs"]),
            smoothing_window_images=int(merged["smoothing_window_images"]),
            image_downscale_factor=float(merged["image_downscale_factor"]),
            use_median_reference=bool(merged["use_median_reference"]),
            reference_refresh_interval_minutes=float(merged["reference_refresh_interval_minutes"]),
            image_cache_size_images=int(merged["image_cache_size_images"]),
            grid_size=int(merged["grid_size"]),
            compute_backend=str(merged.get("compute_backend", "gpu")).strip().lower() or "gpu",
            output_directory=str(merged["output_directory"]),
            save_preview_images=bool(merged["save_preview_images"]),
            preview_image_count=int(merged["preview_image_count"]),
            run_name=str(merged.get("run_name", "")),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Raise ValueError when the processing configuration is invalid."""
        positive_ints = {
            "reference_window_size_images": self.reference_window_size_images,
            "live_average_size_images": self.live_average_size_images,
            "processing_stride_images": self.processing_stride_images,
            "smoothing_window_images": self.smoothing_window_images,
            "grid_size": self.grid_size,
        }
        for name, value in positive_ints.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.reference_gap_images < 0:
            raise ValueError("reference_gap_images must be >= 0")
        if self.image_downscale_factor <= 0:
            raise ValueError("image_downscale_factor must be > 0")
        if self.reference_refresh_interval_minutes < 0:
            raise ValueError("reference_refresh_interval_minutes must be >= 0")
        if self.image_cache_size_images < 0:
            raise ValueError("image_cache_size_images must be >= 0")
        if self.compute_backend not in {"cpu", "gpu"}:
            raise ValueError("compute_backend must be 'cpu' or 'gpu'")
        if self.preview_image_count < 0:
            raise ValueError("preview_image_count must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RoiPreset:
    """Named quadrilateral ROI preset."""

    preset_name: str
    dataset_name: str
    image_shape: list[int]
    corners: dict[str, list[float]]
    created_at: str
    comment: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoiPreset":
        return cls(
            preset_name=str(data["preset_name"]),
            dataset_name=str(data.get("dataset_name", "global")),
            image_shape=[int(v) for v in data.get("image_shape", [])],
            corners={k: [float(x), float(y)] for k, (x, y) in data["corners"].items()},
            created_at=str(data.get("created_at", "")),
            comment=str(data.get("comment", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfigStore:
    """Load and save YAML/JSON configs used by the application."""

    def __init__(
        self,
        datasets_path: str | Path = "configs/datasets.yaml",
        roi_presets_path: str | Path = "configs/roi_presets.json",
        algorithm_presets_path: str | Path = "configs/algorithm_presets.json",
    ) -> None:
        self.datasets_path = Path(datasets_path)
        self.roi_presets_path = Path(roi_presets_path)
        self.algorithm_presets_path = Path(algorithm_presets_path)

    def load_datasets(self) -> list[DatasetConfig]:
        """Load dataset definitions from datasets.yaml."""
        with self.datasets_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return [DatasetConfig.from_dict(item) for item in raw.get("datasets", [])]

    def load_roi_presets(self) -> list[RoiPreset]:
        """Load ROI presets, returning an empty list when the file is missing."""
        data = self._load_json_list(self.roi_presets_path)
        return [RoiPreset.from_dict(item) for item in data]

    def save_roi_preset(self, preset: RoiPreset, overwrite: bool = False) -> list[RoiPreset]:
        """Save an ROI preset, optionally replacing a preset with the same name and dataset."""
        presets = self.load_roi_presets()
        kept: list[RoiPreset] = []
        replaced = False
        for existing in presets:
            same_key = (
                existing.preset_name == preset.preset_name
                and existing.dataset_name == preset.dataset_name
            )
            if same_key:
                if not overwrite:
                    raise ValueError("ROI preset already exists; choose overwrite or a new name")
                kept.append(preset)
                replaced = True
            else:
                kept.append(existing)
        if not replaced:
            kept.append(preset)
        self._write_json_list(self.roi_presets_path, [p.to_dict() for p in kept])
        return kept

    def load_algorithm_presets(self) -> list[dict[str, Any]]:
        """Load algorithm presets as dictionaries."""
        return self._load_json_list(self.algorithm_presets_path)

    def save_algorithm_preset(
        self, preset: dict[str, Any], overwrite: bool = False
    ) -> list[dict[str, Any]]:
        """Save an algorithm preset with duplicate-name protection."""
        presets = self.load_algorithm_presets()
        name = str(preset["preset_name"])
        kept: list[dict[str, Any]] = []
        replaced = False
        for existing in presets:
            if existing.get("preset_name") == name:
                if not overwrite:
                    raise ValueError("Algorithm preset already exists; choose overwrite or a new name")
                kept.append(preset)
                replaced = True
            else:
                kept.append(existing)
        if not replaced:
            kept.append(preset)
        self._write_json_list(self.algorithm_presets_path, kept)
        return kept

    @staticmethod
    def now_string() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _load_json_list(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return list(data.get("presets", []))
        if isinstance(data, list):
            return data
        return []

    @staticmethod
    def _write_json_list(path: Path, values: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(values, fh, indent=2)
            fh.write("\n")


def dataset_config_json(dataset: DatasetConfig) -> dict[str, Any]:
    """Return a JSON-serializable copy of the selected dataset config."""
    return dataset.to_dict()


def algorithm_preset_payload(
    preset_name: str,
    config: AlgorithmConfig,
    comment: str = "",
) -> dict[str, Any]:
    """Build a persisted algorithm preset record."""
    return {
        "preset_name": preset_name,
        **config.to_dict(),
        "created_at": ConfigStore.now_string(),
        "comment": comment,
    }
