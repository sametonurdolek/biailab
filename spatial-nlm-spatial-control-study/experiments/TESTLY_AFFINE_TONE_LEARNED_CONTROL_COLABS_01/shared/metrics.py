"""Canonical page metrics. Reductions match the V2 metrics_np used by BENCH_02."""
from __future__ import annotations

import cv2
import numpy as np

REFLECT = cv2.BORDER_REFLECT_101


def _sobel(x: np.ndarray):
    return (
        cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8, borderType=REFLECT),
        cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8, borderType=REFLECT),
    )


def structure_masks(target: np.ndarray) -> dict:
    """Target-derived evaluation masks. They are never a network input."""
    gx, gy = _sobel(target)
    magnitude = np.hypot(gx, gy)
    positive = magnitude[magnitude > 0.01]
    threshold = max(0.025, float(np.percentile(positive, 60)) if positive.size else 0.025)
    edge = cv2.dilate((magnitude > threshold).astype(np.uint8), np.ones((3, 3), np.uint8))
    return {
        "ink": (target <= 0.75).astype(np.uint8),
        "bg": (target >= 0.90).astype(np.uint8),
        "edge": edge,
    }


def page_metrics(y: np.ndarray, target: np.ndarray, masks: dict, grid: int = 128) -> dict:
    y = np.ascontiguousarray(y, dtype=np.float32)
    target = np.ascontiguousarray(target, dtype=np.float32)
    err = y - target
    ae = np.abs(err)
    gx, gy = _sobel(y)
    tx, ty = _sobel(target)
    ge = (np.abs(gx - tx) + np.abs(gy - ty)) / 2
    blur = lambda a: cv2.GaussianBlur(a, (11, 11), 1.5, borderType=REFLECT)
    my, mt = blur(y), blur(target)
    vy = np.maximum(blur(y * y) - my * my, 0)
    vt = np.maximum(blur(target * target) - mt * mt, 0)
    cov = blur(y * target) - my * mt
    ss = ((2 * my * mt + 0.01 ** 2) * (2 * cov + 0.03 ** 2)) / (
        (my * my + mt * mt + 0.01 ** 2) * (vy + vt + 0.03 ** 2)
    )
    ssim = float(ss[5:-5, 5:-5].mean()) if min(target.shape) > 10 else float(ss.mean())
    cond = lambda a, m: float(a[m.astype(bool)].mean()) if np.any(m) else None
    ye = np.rint(np.linspace(0, target.shape[0], grid + 1)).astype(np.int64)
    xe = np.rint(np.linspace(0, target.shape[1], grid + 1)).astype(np.int64)
    band = np.zeros(target.shape, bool)
    for y0 in ye[1:-1]:
        band[max(0, int(y0) - 1):min(target.shape[0], int(y0) + 2)] = True
    for x0 in xe[1:-1]:
        band[:, max(0, int(x0) - 1):min(target.shape[1], int(x0) + 2)] = True
    edge = masks["edge"].astype(bool)
    boundary = band & ~edge
    psnr = float(-10.0 * np.log10(np.mean(err * err, dtype=np.float64) + 1e-12))
    return {
        "psnr": psnr,
        "ssim": ssim,
        "mae": float(ae.mean()),
        "ink_mae": cond(ae, masks["ink"]),
        "edge_mae": cond(ae, masks["edge"]),
        "bg_mae": cond(ae, masks["bg"]),
        "gradient_error": float(ge.mean()),
        "seam_error": float(ae[boundary].mean()) if boundary.any() else None,
    }


def mean_metrics(rows: list[dict]) -> dict:
    keys = ("psnr", "ssim", "mae", "ink_mae", "edge_mae", "bg_mae", "gradient_error")
    out = {}
    for key in keys:
        vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def recovery_ratio(learned: float, fixed: float, oracle: float | None) -> float | None:
    if oracle is None or not np.isfinite(oracle) or oracle <= fixed + 1e-6:
        return None
    return float((learned - fixed) / (oracle - fixed))
