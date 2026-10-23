# %% [markdown]
# EXP09 — Distill the discrete oracle, then image loss
#
# Stage 1 trains the head on oracle class probabilities. Stage 2 thaws the upper encoder, still on the oracle. Stage 3 trains the full network with a soft mixture of the frozen tone bank as the image surrogate. Validation and export always use argmax plus the exact hybrid compose, not the soft mixture.
#
# Research question: Does oracle distillation before image loss train the discrete bank more stably than EXP01?
#
# Expected, not measured: under 20% after both stages means the bank is hard even with distillation. Over 70%, and above EXP01, makes this schedule the discrete candidate.
#
# Tau is the median positive calibration-cell regret of this bank. It is computed before the optimizer starts and stored in config.json. It is not the V2 scale-regret constant.
#
# Fixed operator, not learned: s=31 closing, Gaussian sigma 31/6, Z = clip(x / max(B, 0.1)).
# The network writes controls only. ALLOW_TEST_SPLIT stays False.
# Copy this whole experiment folder to Drive before Run All. A lone file cannot run.
# Expected text above is not a measured result.

# %%
import importlib
import subprocess
import sys
from pathlib import Path

needed = []
for module_name, pip_name in (
    ("numpy", "numpy"),
    ("cv2", "opencv-python-headless"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
):
    try:
        importlib.import_module(module_name)
    except ImportError:
        needed.append(pip_name)
try:
    import torch
except ImportError:
    needed.append("torch")
if needed:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])

# %%
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    print("Not Colab. DRIVE_ROOT is used as a local path.")

# %%
# The only path that should need an edit.
DRIVE_ROOT = "/content/drive/MyDrive"
ALLOW_TEST_SPLIT = False
RESUME = True
INPUT_MODE = "x"  # x, z, or xz.
LOSS_PRESET = "mse"  # mse | mse_ssim | mse_ssim_edge | mse_structural_smooth
CLAMP_BACKWARD = "exact"  # exact | straight_through
WARM_START = False
LIGHTWEIGHT = False
BATCH_SIZE = None  # None uses the VRAM recommendation
ACCUMULATION = None
EPOCHS = None  # None keeps 40

# %%
def _walk(root: Path, name: str, depth: int):
    if not root.exists() or depth < 0:
        return None
    if root.name == name and (root / "shared" / "engine.py").is_file():
        return root
    if depth == 0 or not root.is_dir():
        return None
    try:
        children = list(root.iterdir())
    except OSError:
        return None
    for child in children:
        if child.is_dir() and not child.name.startswith(".") and child.name not in {"node_modules", "__pycache__"}:
            found = _walk(child, name, depth - 1)
            if found is not None:
                return found
    return None


def locate_package(drive_root: str) -> Path:
    name = "TESTLY_AFFINE_TONE_LEARNED_CONTROL_COLABS_01"
    seeds = [Path.cwd(), Path.cwd().parent]
    if "__file__" in globals():
        here = Path(__file__).resolve().parent
        seeds = [here, here.parent, here.parent.parent, *seeds]
    for seed in seeds:
        if (seed / "shared" / "engine.py").is_file() and seed.name == name:
            return seed
        nested = seed / name
        if (nested / "shared" / "engine.py").is_file():
            return nested
    root = Path(drive_root)
    relatives = [
        Path(name),
        Path("spatial-nlm-spatial-control-study") / "experiments" / name,
        Path("biailab") / "spatial-nlm-spatial-control-study" / "experiments" / name,
        Path("experiments") / name,
    ]
    for relative in relatives:
        candidate = root / relative
        if (candidate / "shared" / "engine.py").is_file():
            return candidate
    print("Searching DRIVE_ROOT for the experiment folder.")
    found = _walk(root, name, 3)
    if found is not None:
        return found
    raise SystemExit(
        "PACKAGE_NOT_FOUND. Copy the whole TESTLY_AFFINE_TONE_LEARNED_CONTROL_COLABS_01 folder "
        "onto Drive. A lone script cannot run."
    )


# %%
if ALLOW_TEST_SPLIT:
    raise SystemExit("ALLOW_TEST_SPLIT must stay False")
package = locate_package(DRIVE_ROOT)
if str(package) not in sys.path:
    sys.path.insert(0, str(package))
from shared.engine import run
from shared.presets import build_exp09

cfg = build_exp09(package)
cfg["drive_root"] = DRIVE_ROOT
cfg["allow_test_split"] = ALLOW_TEST_SPLIT
cfg["resume"] = RESUME
cfg["input_mode"] = INPUT_MODE
cfg["loss_preset"] = LOSS_PRESET
cfg["clamp_backward"] = CLAMP_BACKWARD
cfg["warm_start"] = WARM_START
cfg["lightweight"] = LIGHTWEIGHT
cfg["batch_size"] = BATCH_SIZE
cfg["accumulation"] = ACCUMULATION
if EPOCHS is not None:
    cfg["epochs"] = EPOCHS
run(cfg)
