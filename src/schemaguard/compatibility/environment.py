"""Environment discovery with lazy optional imports and secret filtering."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from ..models.registry import ModelSpec
from .contracts import DeviceRecord, EnvironmentReport, PackageRecord

_SAFE_ENVIRONMENT_KEYS = (
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONNOUSERSITE",
)


def _nvidia_smi() -> tuple[str | None, str | None, float | None, float | None]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None, None, None, None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None, None, None, None
    fields = [field.strip() for field in completed.stdout.splitlines()[0].split(",")]
    if len(fields) < 4:
        return None, None, None, None
    try:
        return fields[0], fields[1], float(fields[2]), float(fields[3])
    except ValueError:
        return fields[0], fields[1], None, None


def _torch_device(gpu_name: str | None, driver: str | None) -> DeviceRecord:
    smi_name, smi_driver, smi_total, smi_free = _nvidia_smi()
    name = smi_name or gpu_name
    try:
        import torch  # type: ignore[import-not-found]

        available = bool(torch.cuda.is_available())
        if available:
            properties = torch.cuda.get_device_properties(0)
            total = float(properties.total_memory / (1024**2))
            free, _ = torch.cuda.mem_get_info(0)
            return DeviceRecord(
                device="cuda",
                available=True,
                name=torch.cuda.get_device_name(0),
                total_memory_mib=total,
                free_memory_mib=float(free / (1024**2)),
                driver=smi_driver or driver,
                cuda_visible=True,
            )
    except Exception as exc:  # telemetry is evidence, never a fabricated zero
        return DeviceRecord(
            device="cuda",
            available=False,
            name=name,
            total_memory_mib=smi_total,
            free_memory_mib=smi_free,
            driver=smi_driver or driver,
            cuda_visible=False,
            reason=f"CUDA telemetry unavailable: {type(exc).__name__}: {exc}",
        )
    return DeviceRecord(
        device="cuda",
        available=False,
        name=name,
        total_memory_mib=smi_total,
        free_memory_mib=smi_free,
        driver=smi_driver or driver,
        cuda_visible=False,
        reason="CUDA is not available to PyTorch",
    )


def _package_record(model: ModelSpec) -> PackageRecord:
    import_name = model.class_path.split(".", 1)[0]
    try:
        observed = importlib.metadata.version(model.package)
    except importlib.metadata.PackageNotFoundError:
        return PackageRecord(
            name=model.package,
            expected_version=model.expected_version,
            import_name=import_name,
            installed=False,
            status="FAIL",
            error="Package is not installed",
        )
    status: Literal["PASS", "FAIL"] = "PASS" if observed == model.expected_version else "FAIL"
    return PackageRecord(
        name=model.package,
        expected_version=model.expected_version,
        observed_version=observed,
        import_name=import_name,
        installed=True,
        status=status,
        error=None if status == "PASS" else "Exact version mismatch",
    )


def detect_environment(models: Iterable[ModelSpec] = ()) -> EnvironmentReport:
    """Capture reproducibility-relevant machine facts without collecting secrets."""

    _, driver, smi_total, smi_free = _nvidia_smi()
    gpu = _torch_device(None, driver)
    try:
        import psutil  # type: ignore[import-not-found]

        installed_ram = float(psutil.virtual_memory().total / (1024**2))
        available_ram = float(psutil.virtual_memory().available / (1024**2))
        physical = psutil.cpu_count(logical=False)
    except Exception:
        installed_ram = None
        available_ram = None
        physical = None
    try:
        import torch  # type: ignore[import-not-found]

        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
    except Exception:
        torch_version = None
        torch_cuda = None
    relevant = {
        key: os.environ[key]
        for key in _SAFE_ENVIRONMENT_KEYS
        if key in os.environ and key != "CUDA_VISIBLE_DEVICES"
    }
    relevant["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    conda_environment = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_environment and ("\\" in conda_environment or "/" in conda_environment):
        conda_environment = Path(conda_environment).name
    return EnvironmentReport(
        python_executable=sys.executable,
        python_version=platform.python_version(),
        conda_environment=conda_environment,
        operating_system=f"{platform.system()} {platform.release()}",
        cpu=platform.processor() or platform.machine(),
        logical_cpu_count=os.cpu_count() or 1,
        physical_cpu_count=physical,
        installed_ram_mib=installed_ram,
        available_ram_mib=available_ram,
        gpu=gpu,
        pytorch_version=torch_version,
        pytorch_cuda_build=torch_cuda,
        cuda_visible=gpu.available,
        relevant_environment=relevant,
        packages=[_package_record(model) for model in models],
    )
