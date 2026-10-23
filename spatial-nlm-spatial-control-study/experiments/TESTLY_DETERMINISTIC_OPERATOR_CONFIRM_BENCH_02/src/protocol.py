"""BENCH_02 confirmation protocol.

Imports the BENCH_01 operators without modifying that tree.
The proceed gate below was written before any BENCH_02 score existed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH01 = Path(r"C:\testly_external_workspace\spatial-nlm\TESTLY_DETERMINISTIC_OPERATOR_30X5_BENCH_01")
DENSE = Path(r"C:\testly_external_workspace\spatial-nlm\TESTLY_ACTION_SPACE_DENSE_S_ORACLE_01")
SOURCE = DENSE / "source"
PROTOCOL_VERSION = 1
SEED = 20261023
N_BOOT_HOLDOUT = 20000
N_BOOT_DEV = 20000
WORKERS = 4
OLD_CACHE_HASH = "ec7b8ae751c0d08622d4577653d4b43a18931de63c09a50ec18f9cc78ce44c08"
SPLIT_SHA = "ab03834bcf1cb56a752228287b830eb30a33c9ff4fb2cb0a6f7d3816c52561e1"
S_VALUES = (15, 21, 31, 45, 63, 91)
REFLECT = cv2.BORDER_REFLECT_101
MATERIAL_HEADROOM_DB = 0.50

# Proceed to the 123 only if at least one predeclared candidate passes every clause.
# Severe pages are flagged. A candidate is catastrophic when more than a quarter of
# the holdout pages fall past the severe line. This fraction was fixed before scores.
GATE = {
    "mean_delta_psnr_gt": 0.50,
    "ci95_lower_gt": 0.0,
    "mean_delta_ssim_gt": 0.0,
    "severe_psnr_lt": -1.0,
    "severe_ssim_lt": -0.05,
    "max_severe_page_fraction": 0.25,
    "candidates": ["S05_V03", "S05_V05", "S03_V04", "S03_V05"],
    "material_spatial_headroom_db": MATERIAL_HEADROOM_DB,
    "structural_ink_edge_ratio_max": 1.25,
}

EXPECTED = {
    "S03": {
        "V01": {"gain": 1.10, "bias": -0.05},
        "V02": {"gain": 1.25, "bias": -0.12},
        "V03": {"gain": 1.45, "bias": -0.22},
        "V04": {"gain": 1.70, "bias": -0.35},
        "V05": {"gain": 2.00, "bias": -0.50},
    },
    "S04": {
        "V01": {"black": 0.02},
        "V02": {"black": 0.05},
        "V03": {"black": 0.10},
        "V04": {"black": 0.18},
        "V05": {"black": 0.28},
    },
    "S05": {
        "V01": {"black": 0.02, "white": 0.98},
        "V02": {"black": 0.05, "white": 0.95},
        "V03": {"black": 0.10, "white": 0.92},
        "V04": {"black": 0.15, "white": 0.88},
        "V05": {"black": 0.22, "white": 0.82},
    },
    "S19": {
        "V01": {"sigma": 16.0, "gain": 0.40},
        "V02": {"sigma": 32.0, "gain": 0.60},
        "V03": {"sigma": 48.0, "gain": 0.80},
        "V04": {"sigma": 80.0, "gain": 1.00},
        "V05": {"sigma": 120.0, "gain": 1.30},
    },
    "S28": {
        "V01": {"sigma": 1.0, "amount": 0.4},
        "V02": {"sigma": 1.2, "amount": 0.8},
        "V03": {"sigma": 1.5, "amount": 1.2},
        "V04": {"sigma": 2.0, "amount": 1.6},
        "V05": {"sigma": 3.0, "amount": 2.2},
    },
}

PRIMARY = ("S03_V04", "S03_V05", "S05_V03", "S05_V05")
FAMILIES = ("S03", "S04", "S05", "S19", "S28")
TONE_ON_SCALES = ("S03", "S05")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache_name(pair_id: str) -> str:
    return hashlib.sha256(f"{pair_id}|{OLD_CACHE_HASH}".encode()).hexdigest() + ".npz"


def nearly(a, b, tol=1e-9) -> bool:
    if set(a) != set(b):
        return False
    return all(abs(float(a[k]) - float(b[k])) <= tol for k in a)


def import_bench01():
    ops = load_module("bench01_ops", BENCH01 / "src" / "operators.py")
    can = load_module("bench01_can", BENCH01 / "src" / "canonical.py")
    cat = load_module("bench01_cat", BENCH01 / "src" / "catalog.py")
    found = {}
    for system in cat.SYSTEMS:
        if system["id"] not in EXPECTED:
            continue
        found[system["id"]] = {p["variant"]: dict(p["params"]) for p in system["presets"]}
    if set(found) != set(EXPECTED):
        raise SystemExit("BENCH01_IMPORT_MISMATCH_STOP missing system")
    for sid, variants in EXPECTED.items():
        for vid, params in variants.items():
            if vid not in found[sid] or not nearly(found[sid][vid], params):
                raise SystemExit(f"BENCH01_IMPORT_MISMATCH_STOP {sid} {vid} {found[sid].get(vid)}")
    matrix_path = BENCH01 / "experiment_matrix.json"
    sidecar = (BENCH01 / "experiment_matrix.sha256").read_text(encoding="utf-8").strip()
    # BENCH_01 hashed the newline-normalized text, which is what its runner checked.
    file_sha = hashlib.sha256(matrix_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    if file_sha != sidecar:
        raise SystemExit("BENCH01_IMPORT_MISMATCH_STOP matrix sidecar")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    rows = matrix["variants"] if isinstance(matrix, dict) else matrix
    for sid, variants in EXPECTED.items():
        for vid, params in variants.items():
            hits = [r for r in rows if r.get("system_id") == sid and r.get("variant_id") == vid]
            if len(hits) != 1:
                raise SystemExit(f"BENCH01_IMPORT_MISMATCH_STOP matrix rows {sid} {vid} {len(hits)}")
            got = hits[0].get("parameters") or hits[0].get("params") or {}
            slim = {k: got[k] for k in params}
            if not nearly(slim, params):
                raise SystemExit(f"BENCH01_IMPORT_MISMATCH_STOP matrix params {sid} {vid} {slim}")
    return ops, can, cat


def load_splits():
    subsets = json.loads((SOURCE / "dense_subsets.json").read_text(encoding="utf-8"))
    split = json.loads((SOURCE / "V1" / "manifests" / "split.json").read_text(encoding="utf-8"))
    split_sha = sha256_file(SOURCE / "V1" / "manifests" / "split.json")
    if split_sha != SPLIT_SHA:
        raise SystemExit(f"SPLIT_HASH_MISMATCH {split_sha}")
    cal = list(subsets["calibration"])
    dev = list(subsets["development_eval"])
    if len(cal) != 31 or len(dev) != 123:
        raise SystemExit("SUBSET_COUNT_MISMATCH")
    if subsets.get("test_used") is not False:
        raise SystemExit("TEST_SPLIT_FLAG")
    test = set(split["test"])
    if set(cal) & test or set(dev) & test or set(cal) & set(dev):
        raise SystemExit("TEST_SPLIT_CONTAMINATION")
    if not set(cal) <= set(split["val"]) or not set(dev) <= set(split["val"]):
        raise SystemExit("SUBSET_NOT_IN_VAL")
    manifest = json.loads((BENCH01 / "benchmark_10" / "benchmark_manifest.json").read_text(encoding="utf-8"))
    discovery = [s["pair_id"] for s in manifest["samples"][:10]]
    if len(discovery) != 10 or len(set(discovery)) != 10:
        raise SystemExit("DISCOVERY10_COUNT")
    if not set(discovery) <= set(cal):
        raise SystemExit("DISCOVERY10_NOT_IN_CALIBRATION")
    holdout = [pid for pid in cal if pid not in set(discovery)]
    if len(holdout) != 21:
        raise SystemExit(f"HOLDOUT_COUNT {len(holdout)}")
    return {
        "calibration": cal,
        "development_eval": dev,
        "discovery10": discovery,
        "holdout21": holdout,
        "test_n": len(test),
        "split_sha256": split_sha,
    }


def load_page(pair_id: str, split_val: set[str]) -> dict:
    if pair_id not in split_val:
        raise RuntimeError("refusing a page outside the frozen validation split")
    path = SOURCE / "V2" / "cache" / cache_name(pair_id)
    with np.load(path, allow_pickle=False) as z:
        dirty = np.ascontiguousarray(z["dirty"], dtype=np.float32)
        target = np.ascontiguousarray(z["T"], dtype=np.float32)
        ink = z["ink"]
        edge = z["edge"]
        bg = z["bg"]
        meta = json.loads(str(z["metadata"]))
    if meta.get("id") != pair_id:
        raise RuntimeError("cache id mismatch")
    return {"pair_id": pair_id, "dirty": dirty, "target": target, "ink": ink, "edge": edge, "bg": bg}


def psnr_ssim(y: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    err = np.ascontiguousarray(y, dtype=np.float32) - np.ascontiguousarray(target, dtype=np.float32)
    psnr = float(-10.0 * np.log10(np.mean(err * err, dtype=np.float64) + 1e-12))
    blur = lambda a: cv2.GaussianBlur(a, (11, 11), 1.5, borderType=REFLECT)
    my, mt = blur(y), blur(target)
    vy = np.maximum(blur(y * y) - my * my, 0)
    vt = np.maximum(blur(target * target) - mt * mt, 0)
    cov = blur(y * target) - my * mt
    ss = ((2 * my * mt + 0.01 ** 2) * (2 * cov + 0.03 ** 2)) / ((my * my + mt * mt + 0.01 ** 2) * (vy + vt + 0.03 ** 2))
    if min(target.shape) > 10:
        ssim = float(ss[5:-5, 5:-5].mean())
    else:
        ssim = float(ss.mean())
    return psnr, ssim


def _sobel(x: np.ndarray):
    return (
        cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8, borderType=REFLECT),
        cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8, borderType=REFLECT),
    )


def structural(y: np.ndarray, page: dict) -> dict:
    """Same reductions as V2 metrics_np. Masks come from the frozen cache."""
    target = page["target"]
    e = y.astype(np.float32) - target
    ae = np.abs(e)
    gx, gy = _sobel(y)
    tx, ty = _sobel(target)
    ge = (np.abs(gx - tx) + np.abs(gy - ty)) / 2
    ye = np.rint(np.linspace(0, target.shape[0], 129)).astype(np.int64)
    xe = np.rint(np.linspace(0, target.shape[1], 129)).astype(np.int64)
    band = np.zeros(target.shape, bool)
    for y0 in ye[1:-1]:
        band[max(0, int(y0) - 1):min(target.shape[0], int(y0) + 2)] = True
    for x0 in xe[1:-1]:
        band[:, max(0, int(x0) - 1):min(target.shape[1], int(x0) + 2)] = True
    edge = page["edge"].astype(bool)
    boundary = band & ~edge
    interior = (~band) & ~edge
    cond = lambda a, m: float(a[m].mean()) if np.any(m) else None
    be = float(ae[boundary].mean()) if boundary.any() else None
    ie = float(ae[interior].mean()) if interior.any() else None
    psnr, ssim = psnr_ssim(y, target)
    return {
        "psnr": psnr,
        "ssim": ssim,
        "mae": float(ae.mean()),
        "ink_mae": cond(ae, page["ink"].astype(bool)),
        "edge_mae": cond(ae, edge),
        "bg_mae": cond(ae, page["bg"].astype(bool)),
        "gradient_error": float(ge.mean()),
        "lighten": cond((e > 0.05).astype(np.float32), page["ink"].astype(bool)),
        "darken": cond((e < -0.05).astype(np.float32), page["ink"].astype(bool)),
        "seam_error": be,
        "seam_gradient": float(ge[boundary].mean()) if boundary.any() else None,
        "seam_ratio": (be / max(ie, 1e-9)) if be is not None and ie is not None else None,
    }


def spatial_detail(dirty, images, target):
    """Canonical G128 hard/hybrid plus area fractions. Matches BENCH_01 spatial_oracle."""
    k, h, w = images.shape
    ye = np.rint(np.linspace(0, h, 129)).astype(np.int64)
    xe = np.rint(np.linspace(0, w, 129)).astype(np.int64)
    area = np.diff(ye)[:, None] * np.diff(xe)[None, :]
    costs = np.empty((k, 128, 128), np.float64)
    for i in range(k):
        ii = cv2.integral((images[i] - target) ** 2, sdepth=cv2.CV_64F)
        sums = (
            ii[ye[1:, None], xe[1:]]
            - ii[ye[:-1, None], xe[1:]]
            - ii[ye[1:, None], xe[:-1]]
            + ii[ye[:-1, None], xe[:-1]]
        )
        costs[i] = sums / np.maximum(area, 1)
    lab = costs.argmin(0).astype(np.int64)
    yi = np.searchsorted(ye, np.arange(h), side="right") - 1
    xi = np.searchsorted(xe, np.arange(w), side="right") - 1
    labels = lab[yi[:, None], xi]
    boundary = np.zeros(labels.shape, np.uint8)
    boundary[1:] |= labels[1:] != labels[:-1]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    band = cv2.dilate(boundary, np.ones((5, 5), np.uint8)).astype(bool)
    hard = np.zeros((h, w), np.float32)
    num = np.zeros((h, w), np.float32)
    den = np.zeros((h, w), np.float32)
    box = lambda a: cv2.boxFilter(a, cv2.CV_32F, (9, 9), normalize=True, borderType=REFLECT)
    guide = np.ascontiguousarray(dirty, dtype=np.float32)
    mean_i = box(guide)
    var_i = np.maximum(box(guide * guide) - mean_i * mean_i, 0)
    for j in np.unique(labels).tolist():
        indicator = (labels == j).astype(np.float32)
        image = images[int(j)]
        hard += indicator * image
        if band.any():
            mean_c = box(indicator)
            a = (box(guide * indicator) - mean_i * mean_c) / (var_i + 0.001)
            b = mean_c - a * mean_i
            weight = np.maximum(box(a) * guide + box(b), 0)
            num += weight * image
            den += weight
    hybrid = hard.copy()
    if band.any():
        hybrid[band] = num[band] / np.maximum(den[band], 1e-12)
    hybrid = np.clip(hybrid, 0, 1).astype(np.float32)
    hard = np.clip(hard, 0, 1).astype(np.float32)
    total = float(area.sum())
    fractions = {int(j): float(area[lab == j].sum() / total) for j in range(k)}
    return hard, hybrid, fractions


def tone_diag(y: np.ndarray, dirty: np.ndarray, z: np.ndarray | None, black: float | None, white: float | None) -> dict:
    d = y.astype(np.float32) - dirty
    out = {
        "frac_eq_0": float(np.mean(y <= 1e-6)),
        "frac_eq_1": float(np.mean(y >= 1.0 - 1e-6)),
        "frac_lt_dirty": float(np.mean(d < -1e-6)),
        "frac_gt_dirty": float(np.mean(d > 1e-6)),
        "mean_darkening": float((-d[d < -1e-6]).mean()) if np.any(d < -1e-6) else 0.0,
        "mean_brightening": float(d[d > 1e-6].mean()) if np.any(d > 1e-6) else 0.0,
        "black_clip_frac": None,
        "white_clip_frac": None,
    }
    if z is not None and black is not None and white is not None:
        out["black_clip_frac"] = float(np.mean(z <= float(black)))
        out["white_clip_frac"] = float(np.mean(z >= float(white)))
    return out


def region_pack(y: np.ndarray, dirty: np.ndarray, target: np.ndarray) -> dict:
    e2 = (y.astype(np.float64) - target.astype(np.float64)) ** 2
    total = float(e2.sum())
    packed = {}
    for name, eps in (("1/255", 1.0 / 255.0), ("3/255", 3.0 / 255.0)):
        above = target > dirty + eps
        near = np.abs(target - dirty) <= eps
        below = target < dirty - eps
        regions = {}
        for key, mask in (("above", above), ("near", near), ("below", below)):
            regions[key] = {
                "pixel_fraction": float(mask.mean()),
                "mse": float(e2[mask].mean()) if mask.any() else None,
                "sse_share": float(e2[mask].sum() / total) if total > 0 and mask.any() else 0.0,
            }
        packed[name] = regions
    return packed


def dirty_descriptors(x: np.ndarray) -> dict:
    blur = cv2.GaussianBlur(x, (0, 0), 48, borderType=REFLECT)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "frac_below_0_2": float(np.mean(x < 0.2)),
        "frac_above_0_9": float(np.mean(x > 0.9)),
        "low_frequency_variance": float(blur.var()),
        "low_frequency_range": float(np.percentile(blur, 95) - np.percentile(blur, 5)),
    }


def entropy_bits(fractions: dict) -> float:
    ps = np.array([v for v in fractions.values() if v > 0], dtype=np.float64)
    if ps.size == 0:
        return 0.0
    return float(-(ps * np.log2(ps)).sum())


def affine_from_endpoints(black: float, white: float) -> dict:
    span = float(white) - float(black)
    return {"gain": 1.0 / span, "bias": -float(black) / span, "black": float(black), "white": float(white)}


def apply_frozen(ops, dirty: np.ndarray, z: np.ndarray, sid: str, params: dict) -> np.ndarray:
    """Algebraic form of the imported operator. Phase 0 checks it against apply_operator."""
    if sid == "S03":
        return ops._affine(z, params["gain"], params["bias"])
    if sid == "S04":
        return ops._black_white(z, params["black"], 1.0)
    if sid == "S05":
        return ops._black_white(z, params["black"], params["white"])
    if sid == "S28":
        blur = ops._gauss(z, params["sigma"])
        return ops._clip(z + float(params["amount"]) * (z - blur))
    if sid == "S19":
        return ops.apply_operator(dirty, {"system_id": "S19", **params})
    raise RuntimeError(sid)


def endpoints_from_affine(gain: float, bias: float) -> dict:
    gain = float(gain)
    bias = float(bias)
    return {"gain": gain, "bias": bias, "black": -bias / gain, "white": (1.0 - bias) / gain}
