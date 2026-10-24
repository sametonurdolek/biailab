"""Seeds, batch sizing, and JSON helpers shared by the Colab entry points."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch


def runtime_record(cwd: Path | None = None) -> dict:
    """Versions stored beside a run. They do not belong to the resume signature."""
    import platform
    import subprocess

    commit = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or Path.cwd()),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "git_commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    cv2.setNumThreads(1)


def digest(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def gpu_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def recommend_batch(kind: str, grid: int, vram_gb: float) -> tuple[int, int]:
    """Batch and accumulation aimed at an effective batch near 8.

    Full-resolution tone loss, not the 1.8M-parameter backbone, dominates memory.
    """
    if vram_gb <= 0:
        return 1, 1
    if kind == "global":
        batch = 16 if vram_gb >= 16 else 8
    elif grid <= 32:
        batch = 8 if vram_gb >= 16 else 4
    elif grid <= 64:
        batch = 4 if vram_gb >= 16 else 2
    elif vram_gb >= 24:
        batch = 4
    elif vram_gb >= 14:
        batch = 2
    else:
        batch = 1
    return batch, max(1, int(round(8 / batch)))


def fmt(value, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def write_history(path: Path, rows: list[dict]) -> None:
    keys = [
        "epoch", "stage", "train_loss", "cal_psnr", "cal_ssim", "cal_mae",
        "cal_regret", "structural_pass", "promoted", "global_collapse_fraction",
    ]
    lines = [",".join(keys)]
    for row in rows:
        cells = []
        for key in keys:
            value = row.get(key, "")
            cells.append("" if value is None else str(value))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def recovery_label(recovery: float | None, low: float, high: float) -> str:
    """Exploratory label. It is not a claim that a run succeeded."""
    if recovery is None or not np.isfinite(recovery):
        return "UNMEASURED"
    if recovery < low:
        return "BELOW_20"
    if recovery > high:
        return "ABOVE_70"
    return "INTERMEDIATE"


def file_sha256(path: Path) -> str:
    digest_ = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest_.update(chunk)
    return digest_.hexdigest()
