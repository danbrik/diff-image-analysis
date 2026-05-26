"""Optional CPU/GPU compute backend helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EXPECTED_TORCH_VERSION = "2.6.0+cu124"
EXPECTED_TORCHVISION_VERSION = "0.21.0+cu124"
EXPECTED_CUDA_VERSION = "12.4"

_TORCH: Any | None = None


@dataclass(slots=True)
class ComputeContext:
    """Resolved compute backend for a processing run."""

    backend: str
    device: str
    device_name: str = ""
    torch_version: str = ""
    cuda_version: str = ""


def check_gpu_status() -> dict[str, Any]:
    """Return CUDA/Torch availability and version details for the UI."""
    status: dict[str, Any] = {
        "available": False,
        "torch_installed": False,
        "torch_version": "",
        "torchvision_installed": False,
        "torchvision_version": "",
        "cuda_available": False,
        "cuda_version": "",
        "device_count": 0,
        "device_name": "",
        "device_capability": "",
        "expected_torch_version": EXPECTED_TORCH_VERSION,
        "expected_torchvision_version": EXPECTED_TORCHVISION_VERSION,
        "expected_cuda_version": EXPECTED_CUDA_VERSION,
        "version_matches": False,
        "message": "",
    }
    try:
        torch = _import_torch()
    except Exception as exc:
        status["message"] = (
            f"PyTorch is not available: {exc}. Expected later runtime: "
            f"torch {EXPECTED_TORCH_VERSION}, torchvision {EXPECTED_TORCHVISION_VERSION}, CUDA {EXPECTED_CUDA_VERSION}."
        )
        return status

    status["torch_installed"] = True
    status["torch_version"] = str(getattr(torch, "__version__", ""))
    status["cuda_version"] = str(getattr(torch.version, "cuda", "") or "")
    try:
        import torchvision  # type: ignore[import-not-found]

        status["torchvision_installed"] = True
        status["torchvision_version"] = str(getattr(torchvision, "__version__", ""))
    except Exception:
        status["torchvision_installed"] = False

    cuda_available = bool(torch.cuda.is_available())
    status["cuda_available"] = cuda_available
    if not cuda_available:
        status["message"] = "PyTorch is installed, but CUDA is not available."
        return status

    device_count = int(torch.cuda.device_count())
    status["device_count"] = device_count
    if device_count > 0:
        status["device_name"] = str(torch.cuda.get_device_name(0))
        capability = torch.cuda.get_device_capability(0)
        status["device_capability"] = f"{capability[0]}.{capability[1]}"
    status["available"] = True
    status["version_matches"] = (
        status["torch_version"] == EXPECTED_TORCH_VERSION
        and status["torchvision_version"] == EXPECTED_TORCHVISION_VERSION
        and status["cuda_version"] == EXPECTED_CUDA_VERSION
    )
    status["message"] = (
        f"CUDA GPU available: {status['device_name'] or 'device 0'} "
        f"(torch {status['torch_version']}, torchvision {status['torchvision_version'] or 'not installed'}, "
        f"CUDA {status['cuda_version'] or 'unknown'})."
    )
    return status


def get_compute_context(backend: str) -> ComputeContext:
    """Resolve and validate the requested compute backend."""
    normalized = str(backend or "gpu").strip().lower()
    if normalized not in {"cpu", "gpu"}:
        raise ValueError("compute_backend must be 'cpu' or 'gpu'")
    if normalized == "cpu":
        return ComputeContext(backend="cpu", device="cpu")

    status = check_gpu_status()
    if not status["available"]:
        raise RuntimeError(status["message"] or "GPU backend requested, but CUDA GPU is not available.")
    return ComputeContext(
        backend="gpu",
        device="cuda",
        device_name=str(status.get("device_name") or "CUDA device 0"),
        torch_version=str(status.get("torch_version") or ""),
        cuda_version=str(status.get("cuda_version") or ""),
    )


def torch_module() -> Any:
    """Return the imported torch module, raising if PyTorch is unavailable."""
    return _import_torch()


def _import_torch() -> Any:
    global _TORCH
    if _TORCH is None:
        import torch  # type: ignore[import-not-found]

        _TORCH = torch
    return _TORCH
