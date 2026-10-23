# %% [markdown]
# EXP01 — Discrete G128 tone classifier
#
# The network emits K class logits on a 128×128 grid. K is the length of the frozen BENCH_01 S05 bank (V01–V05). It is not a BENCH_03 re-optimization. Argmax selects a preset. Deployment is the canonical hybrid: hard tone inside cells, and a 5×5 dirty-guided blend on the label boundary. The clean page is an offline teacher only.
#
# Research question: Can a G128 classifier over the frozen five S05 presets recover spatial tone headroom?
#
# Expected, not measured: under 20% development recovery of the frozen-bank hybrid headroom over S05 V03 means this discrete controller looks difficult. Over 70% makes it a strong candidate. Those thresholds are exploratory labels.
#
# Measured values appear only in the end-of-run summary.
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
INPUT_MODE = "x"  # x, z, or xz. This does not change the discrete head.
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
from shared.presets import build_exp01

cfg = build_exp01(package)
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
