# %% [markdown]
# EXP05 — Global page black/white
#
# The network emits two numbers for the whole page. Every pixel then uses the same exact black/white tone. This is the cheapest controller and the right first run.
#
# Research question: Are two page-level numbers enough to control the confirmed tone operator?
#
# Expected, not measured: BENCH_02 page-oracle headroom above the best frozen preset S05 V04 was 0.414 dB, under the 0.50 dB material line. Recovery in this script uses the frozen five-preset page oracle against S05 V03. Under 20% means that page gap is not being learned. Over 70% means the page oracle is learnable even if the absolute gap stays small. A constant map is this architecture, not a failure.
#
# SUPERVISION image compares the toned page with the clean target. oracle_param regresses the offline page lattice (b*, w*) and does not feed the clean page to the network. The lattice is a search grid of the fallback box, not an optimum.
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
SUPERVISION = "image"  # image or oracle_param. oracle_param turns image loss off unless IMAGE_WITH_ORACLE is True.
IMAGE_WITH_ORACLE = False

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
from shared.presets import build_exp05

cfg = build_exp05(package)
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
cfg["supervision"] = SUPERVISION
cfg["image_with_oracle"] = IMAGE_WITH_ORACLE
run(cfg)
