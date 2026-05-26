"""Flask browser application for server-side image sequence analysis."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import sys
import threading
import time
import uuid
from typing import Any

import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_file

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diff_image_analysis.config import (  # noqa: E402
    AlgorithmConfig,
    ConfigStore,
    RoiPreset,
    algorithm_preset_payload,
    parse_datetime,
)
from diff_image_analysis.compute import check_gpu_status  # noqa: E402
from diff_image_analysis.image_io import preview_png_bytes  # noqa: E402
from diff_image_analysis.indexing import dataset_summary, filter_records_by_time, index_dataset  # noqa: E402
from diff_image_analysis.metrics import DEFAULT_PLOT_METRICS  # noqa: E402
from diff_image_analysis.plotting import save_metrics_plot  # noqa: E402
from diff_image_analysis.processor import run_difference_analysis  # noqa: E402


app = Flask(
    __name__,
    template_folder=str(ROOT / "web" / "templates"),
    static_folder=str(ROOT / "web" / "static"),
)

CONFIG = ConfigStore(
    datasets_path=ROOT / "configs" / "datasets.yaml",
    roi_presets_path=ROOT / "configs" / "roi_presets.json",
    algorithm_presets_path=ROOT / "configs" / "algorithm_presets.json",
)

DATASETS = {dataset.name: dataset for dataset in CONFIG.load_datasets()}
INDEX_CACHE: dict[str, pd.DataFrame] = {}
PREVIEW_CACHE: dict[str, tuple[bytes, dict[str, Any]]] = {}
JOBS: dict[str, dict[str, Any]] = {}
RUN_ROOTS = {(ROOT / "outputs" / "runs").resolve()}
STATE_LOCK = threading.Lock()


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/datasets")
def api_datasets() -> Any:
    datasets = []
    for dataset in DATASETS.values():
        indexed = dataset.name in INDEX_CACHE
        summary = dataset_summary(INDEX_CACHE[dataset.name]) if indexed else {}
        datasets.append({**dataset.to_dict(), "indexed": indexed, **summary})
    return jsonify({"datasets": datasets})


@app.post("/api/datasets/<dataset_name>/index")
def api_index_dataset(dataset_name: str) -> Any:
    dataset = _get_dataset(dataset_name)
    records = index_dataset(dataset)
    INDEX_CACHE[dataset_name] = records
    return jsonify(
        {
            "dataset": dataset.to_dict(),
            "summary": dataset_summary(records),
            "availability": _availability_summary(records),
        }
    )


@app.get("/api/datasets/<dataset_name>/range-count")
def api_range_count(dataset_name: str) -> Any:
    records = _ensure_indexed(dataset_name)
    start, end = _request_time_range()
    selected = filter_records_by_time(records, start, end)
    summary = dataset_summary(records)
    count = int(len(selected))
    warning = "The selected time range contains no timestamped images." if count == 0 else ""
    return jsonify(
        {
            "count": count,
            "run_enabled": count > 0,
            "warning": warning,
            "availability": _availability_summary(records),
            **summary,
        }
    )


@app.get("/api/datasets/<dataset_name>/available-times")
def api_available_times(dataset_name: str) -> Any:
    records = _ensure_indexed(dataset_name)
    date_text = request.args.get("date", "")
    timestamped = records[records["timestamp"].notna()].copy()
    if date_text:
        timestamped = timestamped[timestamped["timestamp"].dt.strftime("%Y-%m-%d") == date_text]
    values = []
    for timestamp in timestamped["timestamp"].sort_values():
        dt = timestamp.to_pydatetime()
        values.append(
            {
                "timestamp": dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "date": dt.strftime("%Y-%m-%d"),
                "time": dt.strftime("%H:%M:%S"),
            }
        )
    return jsonify({"date": date_text, "times": values})


@app.get("/api/datasets/<dataset_name>/preview-info")
def api_preview_info(dataset_name: str) -> Any:
    records = _ensure_indexed(dataset_name)
    start, end = _request_time_range()
    selected = filter_records_by_time(records, start, end)
    if selected.empty:
        abort(404, description="No timestamped images in selected range")
    image_path = str(selected.iloc[0]["image_path"])
    png, meta = preview_png_bytes(image_path)
    token = uuid.uuid4().hex
    PREVIEW_CACHE[token] = (png, meta)
    return jsonify(
        {
            "image_url": f"/api/previews/{token}.png",
            "image_path": image_path,
            **meta,
        }
    )


@app.get("/api/previews/<token>.png")
def api_preview_image(token: str) -> Any:
    cached = PREVIEW_CACHE.get(token)
    if cached is None:
        abort(404)
    png, _meta = cached
    return send_file(BytesIO(png), mimetype="image/png")


@app.get("/api/roi-presets")
def api_roi_presets() -> Any:
    dataset_name = request.args.get("dataset", "")
    presets = [
        preset.to_dict()
        for preset in CONFIG.load_roi_presets()
        if not dataset_name or preset.dataset_name in (dataset_name, "global")
    ]
    return jsonify({"presets": presets})


@app.post("/api/roi-presets")
def api_save_roi_preset() -> Any:
    data = request.get_json(force=True)
    preset = RoiPreset(
        preset_name=str(data["preset_name"]).strip(),
        dataset_name=str(data.get("dataset_name") or "global"),
        image_shape=[int(v) for v in data["image_shape"]],
        corners={k: [float(x), float(y)] for k, (x, y) in data["corners"].items()},
        created_at=ConfigStore.now_string(),
        comment=str(data.get("comment", "")),
    )
    overwrite = bool(data.get("overwrite", False))
    presets = CONFIG.save_roi_preset(preset, overwrite=overwrite)
    return jsonify({"presets": [p.to_dict() for p in presets], "saved": preset.to_dict()})


@app.get("/api/algorithm-presets")
def api_algorithm_presets() -> Any:
    presets = CONFIG.load_algorithm_presets()
    return jsonify({"presets": presets, "defaults": AlgorithmConfig().to_dict()})


@app.get("/api/compute/gpu-status")
def api_gpu_status() -> Any:
    return jsonify(check_gpu_status())


@app.post("/api/algorithm-presets")
def api_save_algorithm_preset() -> Any:
    data = request.get_json(force=True)
    preset_name = str(data["preset_name"]).strip()
    config = AlgorithmConfig.from_dict(data.get("config", {}))
    comment = str(data.get("comment", ""))
    overwrite = bool(data.get("overwrite", False))
    payload = algorithm_preset_payload(preset_name, config, comment=comment)
    presets = CONFIG.save_algorithm_preset(payload, overwrite=overwrite)
    return jsonify({"presets": presets, "saved": payload})


@app.post("/api/runs")
def api_start_run() -> Any:
    data = request.get_json(force=True)
    dataset_name = str(data["dataset_name"])
    dataset = _get_dataset(dataset_name)
    records = _ensure_indexed(dataset_name)
    start, end = _payload_time_range(data)
    selected = filter_records_by_time(records, start, end)
    if selected.empty:
        abort(400, description="Selected time range contains no timestamped images")

    algorithm_config = AlgorithmConfig.from_dict(data.get("algorithm_config", {}))
    if algorithm_config.compute_backend == "gpu":
        gpu_status = check_gpu_status()
        if not gpu_status["available"]:
            return jsonify({"error": gpu_status["message"], "gpu_status": gpu_status}), 400
    output_root = Path(algorithm_config.output_directory)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    algorithm_config.output_directory = str(output_root)
    RUN_ROOTS.add(output_root.resolve())
    roi_config = data["roi_config"]
    job_id = uuid.uuid4().hex
    with STATE_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "state": "running",
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "progress": {
                "dataset_name": dataset_name,
                "total_images": int(len(selected)),
                "processed_images": 0,
                "percentage": 0.0,
                "status_message": "queued",
                "compute_backend": algorithm_config.compute_backend,
                "compute_device": "gpu" if algorithm_config.compute_backend == "gpu" else "cpu",
            },
            "result": None,
            "error": None,
        }

    def update_progress(progress: dict[str, Any]) -> None:
        with STATE_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["progress"] = progress

    def worker() -> None:
        try:
            result = run_difference_analysis(
                all_records=records,
                dataset=dataset,
                roi_config=roi_config,
                algorithm_config=algorithm_config,
                range_start=start,
                range_end=end,
                progress_callback=update_progress,
                selected_algorithm_preset=str(data.get("algorithm_preset_name", "")),
                selected_roi_preset=str(data.get("roi_preset_name", "")),
            )
            with STATE_LOCK:
                JOBS[job_id]["state"] = "finished"
                JOBS[job_id]["result"] = {
                    "run_id": result.run_folder.name,
                    "run_folder": str(result.run_folder),
                    "results_csv": str(result.results_csv),
                    "summary_plot": str(result.summary_plot),
                }
        except Exception as exc:
            with STATE_LOCK:
                JOBS[job_id]["state"] = "failed"
                JOBS[job_id]["error"] = str(exc)

    thread = threading.Thread(target=worker, name=f"analysis-run-{job_id}", daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})


@app.get("/api/runs/<job_id>/status")
def api_run_status(job_id: str) -> Any:
    with STATE_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            abort(404)
        return jsonify(job)


@app.get("/api/results/runs")
def api_list_result_runs() -> Any:
    runs = []
    for root in sorted(RUN_ROOTS):
        root.mkdir(parents=True, exist_ok=True)
        for path in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
            results_csv = path / "results.csv"
            runs.append(
                {
                    "run_id": path.name,
                    "path": str(path),
                    "has_results": results_csv.exists(),
                    "mtime": path.stat().st_mtime,
                }
            )
    runs.sort(key=lambda item: item["mtime"], reverse=True)
    return jsonify({"runs": runs})


@app.get("/api/results/runs/<run_id>")
def api_result_run_details(run_id: str) -> Any:
    run_folder = _safe_run_folder(run_id)
    results_csv = run_folder / "results.csv"
    if not results_csv.exists():
        abort(404, description="results.csv not found")
    results = pd.read_csv(results_csv)
    metric_columns = [
        col
        for col in results.columns
        if col not in {"timestamp", "image_path", "dataset_name", "processed_index", "status"}
    ]
    return jsonify(
        {
            "run_id": run_id,
            "row_count": int(len(results)),
            "metric_columns": metric_columns,
            "default_metrics": [m for m in DEFAULT_PLOT_METRICS if m in metric_columns],
            "run_config": _read_json_if_exists(run_folder / "run_config.json"),
            "dataset_config_used": _read_json_if_exists(run_folder / "dataset_config_used.json"),
            "roi_config": _read_json_if_exists(run_folder / "roi_config.json"),
            "summary_plot_url": f"/api/results/runs/{run_id}/files/summary_plot.png"
            if (run_folder / "summary_plot.png").exists()
            else None,
        }
    )


@app.post("/api/results/runs/<run_id>/plot")
def api_result_plot(run_id: str) -> Any:
    run_folder = _safe_run_folder(run_id)
    data = request.get_json(force=True)
    metrics = [str(metric) for metric in data.get("metrics", [])]
    results = pd.read_csv(run_folder / "results.csv")
    output = run_folder / f"selected_metrics_{int(time.time())}.png"
    plotted = save_metrics_plot(results, metrics, output, title=f"{run_id} metrics")
    return jsonify(
        {
            "plot_url": f"/api/results/runs/{run_id}/files/{output.name}",
            "plotted_metrics": plotted,
        }
    )


@app.get("/api/results/runs/<run_id>/data")
def api_result_data(run_id: str) -> Any:
    run_folder = _safe_run_folder(run_id)
    results = pd.read_csv(run_folder / "results.csv")
    metrics = [metric for metric in request.args.getlist("metric") if metric in results.columns]
    if not metrics:
        metrics = [metric for metric in DEFAULT_PLOT_METRICS if metric in results.columns]
    timestamps = pd.to_datetime(results.get("timestamp"), errors="coerce")
    payload_rows = []
    for idx, timestamp in enumerate(timestamps):
        row: dict[str, Any] = {
            "timestamp": timestamp.isoformat() if pd.notna(timestamp) else None,
        }
        for metric in metrics:
            value = pd.to_numeric(pd.Series([results.at[idx, metric]]), errors="coerce").iloc[0]
            row[metric] = None if pd.isna(value) else float(value)
        payload_rows.append(row)
    return jsonify({"run_id": run_id, "metrics": metrics, "rows": payload_rows})


@app.get("/api/results/runs/<run_id>/files/<filename>")
def api_result_file(run_id: str, filename: str) -> Any:
    allowed_suffixes = {".png", ".csv", ".txt", ".json"}
    run_folder = _safe_run_folder(run_id)
    path = (run_folder / filename).resolve()
    if run_folder.resolve() not in path.parents and path != run_folder.resolve():
        abort(400)
    if path.suffix.lower() not in allowed_suffixes or not path.exists():
        abort(404)
    return send_file(path)


def _get_dataset(dataset_name: str):
    dataset = DATASETS.get(dataset_name)
    if dataset is None:
        abort(404, description=f"Unknown dataset: {dataset_name}")
    return dataset


def _ensure_indexed(dataset_name: str) -> pd.DataFrame:
    if dataset_name not in INDEX_CACHE:
        INDEX_CACHE[dataset_name] = index_dataset(_get_dataset(dataset_name))
    return INDEX_CACHE[dataset_name]


def _request_time_range() -> tuple[datetime | None, datetime | None]:
    if request.args.get("mode") != "custom":
        return None, None
    return parse_datetime(request.args.get("start")), parse_datetime(request.args.get("end"))


def _payload_time_range(data: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
    if data.get("time_mode") != "custom":
        return None, None
    return parse_datetime(data.get("range_start")), parse_datetime(data.get("range_end"))


def _safe_run_folder(run_id: str) -> Path:
    for root in RUN_ROOTS:
        path = (root / run_id).resolve()
        if root in path.parents and path.exists():
            return path
    abort(404, description="Run folder not found")


def _read_json_if_exists(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _availability_summary(records: pd.DataFrame) -> dict[str, Any]:
    if records.empty or "timestamp" not in records.columns:
        return {"available_dates": [], "day_counts": {}, "first_timestamp": None, "last_timestamp": None}
    timestamped = records[records["timestamp"].notna()].copy()
    if timestamped.empty:
        return {"available_dates": [], "day_counts": {}, "first_timestamp": None, "last_timestamp": None}
    dates = timestamped["timestamp"].dt.strftime("%Y-%m-%d")
    day_counts = dates.value_counts().sort_index().astype(int).to_dict()
    first_ts = timestamped["timestamp"].min()
    last_ts = timestamped["timestamp"].max()
    return {
        "available_dates": list(day_counts.keys()),
        "day_counts": day_counts,
        "first_timestamp": first_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "last_timestamp": last_ts.strftime("%Y-%m-%d %H:%M:%S"),
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, threaded=True)
