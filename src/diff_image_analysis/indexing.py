"""Direct-folder TIFF indexing and timestamp parsing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import pandas as pd

from .config import DatasetConfig, format_datetime


TIFF_SUFFIXES = {".tif", ".tiff"}


def index_dataset(dataset: DatasetConfig) -> pd.DataFrame:
    """Scan configured direct folders and return image records as a dataframe.

    Only files directly inside each configured folder are considered. Subfolders are
    intentionally ignored.
    """
    pattern = re.compile(dataset.timestamp_regex)
    rows: list[dict[str, object]] = []
    for folder_text in dataset.folders:
        folder = Path(folder_text).expanduser()
        if not folder.exists() or not folder.is_dir():
            continue
        for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or path.suffix.lower() not in TIFF_SUFFIXES:
                continue
            timestamp = _parse_timestamp(path.name, pattern, dataset.timestamp_format)
            rows.append(
                {
                    "timestamp": timestamp,
                    "image_path": str(path),
                    "dataset_name": dataset.name,
                    "source_folder": str(folder),
                    "timestamp_missing": timestamp is None,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "image_path",
                "dataset_name",
                "source_folder",
                "timestamp_missing",
            ]
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["_sort_missing"] = df["timestamp"].isna()
    df["_sort_name"] = df["image_path"].map(lambda p: Path(str(p)).name.lower())
    df = df.sort_values(["_sort_missing", "timestamp", "_sort_name"], na_position="last")
    df = df.drop(columns=["_sort_missing", "_sort_name"]).reset_index(drop=True)
    return df


def filter_records_by_time(
    records: pd.DataFrame,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Return timestamped records that fall inside an optional closed time range."""
    if records.empty:
        return records.copy()
    timestamped = records[records["timestamp"].notna()].copy()
    if start is not None:
        timestamped = timestamped[timestamped["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        timestamped = timestamped[timestamped["timestamp"] <= pd.Timestamp(end)]
    return timestamped.reset_index(drop=False).rename(columns={"index": "source_index"})


def dataset_summary(records: pd.DataFrame) -> dict[str, object]:
    """Summarize indexed records for UI display."""
    timestamped = records[records["timestamp"].notna()] if not records.empty else records
    first_ts = None
    last_ts = None
    if not timestamped.empty:
        first_ts = format_datetime(timestamped["timestamp"].iloc[0].to_pydatetime())
        last_ts = format_datetime(timestamped["timestamp"].iloc[-1].to_pydatetime())
    missing = int(records["timestamp"].isna().sum()) if not records.empty else 0
    return {
        "file_count": int(len(records)),
        "missing_timestamp_count": missing,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
    }


def _parse_timestamp(name: str, pattern: re.Pattern[str], timestamp_format: str) -> datetime | None:
    match = pattern.search(name)
    if not match:
        return None
    text = match.group(1) if match.groups() else match.group(0)
    try:
        return datetime.strptime(text, timestamp_format)
    except ValueError:
        return None
