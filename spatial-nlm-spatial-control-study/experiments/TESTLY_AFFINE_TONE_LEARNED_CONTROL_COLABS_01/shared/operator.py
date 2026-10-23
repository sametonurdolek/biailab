"""Frozen s=31 paper normalization and the post-normalization affine tone family.

Validation and export always use the exact clip. Straight-through clamp is a
training backward option and is never the deployed operator.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.autograd import Function

REFLECT = cv2.BORDER_REFLECT_101
FLOOR = 0.1
FIXED_S = 31
SIGMA = FIXED_S / 6.0


def normalize_paper(x: np.ndarray, scale: int) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    kernel = np.ones((int(scale), int(scale)), np.uint8)
    background0 = cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel, borderType=REFLECT)
    sigma = float(scale) / 6.0
    background = cv2.GaussianBlur(
        background0, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=REFLECT
    )
    return np.clip(x / np.maximum(background, FLOOR), 0, 1).astype(np.float32)


def normalize_s31(x: np.ndarray) -> np.ndarray:
    """Canonical closing, Gaussian, and division. s is not a learned output."""
    return normalize_paper(x, FIXED_S)


def tone_bw_np(z: np.ndarray, black: float, white: float) -> np.ndarray:
    span = float(white) - float(black)
    if span <= 1e-6:
        raise RuntimeError("black/white span collapsed")
    return np.clip((z - float(black)) / span, 0, 1).astype(np.float32)


def tone_affine_np(z: np.ndarray, gain: float, bias: float) -> np.ndarray:
    return np.clip(float(gain) * z + float(bias), 0, 1).astype(np.float32)


def affine_from_endpoints(black: float, white: float) -> tuple[float, float]:
    span = float(white) - float(black)
    return 1.0 / span, -float(black) / span


class _HardClampSTE(Function):
    """Forward is the exact clip. Backward passes the incoming gradient through."""

    @staticmethod
    def forward(ctx, y):
        return y.clamp(0, 1)

    @staticmethod
    def backward(ctx, grad):
        return grad


def clamp_tone(y: torch.Tensor, backward: str) -> torch.Tensor:
    if backward == "straight_through":
        return _HardClampSTE.apply(y)
    if backward == "exact":
        return y.clamp(0, 1)
    raise RuntimeError(f"unknown clamp backward {backward}")


def endpoints_from_raw(raw: torch.Tensor, bounds: dict, min_span: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Map unbounded (u, v) onto the legal black/white box.

    Guarantees black_lo <= black, black + min_span <= white, and white <= white_hi,
    with white also kept at or above white_lo when that is feasible inside the box.
    """
    u = torch.sigmoid(raw[..., 0])
    v = torch.sigmoid(raw[..., 1])
    black_lo = float(bounds["black_lo"])
    black_hi = min(float(bounds["black_hi"]), float(bounds["white_hi"]) - float(min_span))
    white_lo = float(bounds["white_lo"])
    white_hi = float(bounds["white_hi"])
    if black_hi < black_lo:
        raise RuntimeError("endpoint bounds do not leave room for min_span")
    black = black_lo + (black_hi - black_lo) * u
    low_span = torch.clamp(white_lo - black, min=float(min_span))
    high_span = torch.clamp(white_hi - black, min=low_span)
    span = low_span + (high_span - low_span) * v
    return black, black + span


def gain_bias_from_raw(raw: torch.Tensor, bounds: dict) -> tuple[torch.Tensor, torch.Tensor]:
    u = torch.sigmoid(raw[..., 0])
    v = torch.sigmoid(raw[..., 1])
    gain = float(bounds["gain_lo"]) + (float(bounds["gain_hi"]) - float(bounds["gain_lo"])) * u
    bias = float(bounds["bias_lo"]) + (float(bounds["bias_hi"]) - float(bounds["bias_lo"])) * v
    return gain, bias


def apply_bw(z: torch.Tensor, black: torch.Tensor, white: torch.Tensor, backward: str, min_span: float) -> torch.Tensor:
    while black.ndim < z.ndim:
        black = black.unsqueeze(-1)
        white = white.unsqueeze(-1)
    span = (white - black).clamp_min(float(min_span))
    return clamp_tone((z - black) / span, backward)


def apply_gain_bias(z: torch.Tensor, gain: torch.Tensor, bias: torch.Tensor, backward: str) -> torch.Tensor:
    while gain.ndim < z.ndim:
        gain = gain.unsqueeze(-1)
        bias = bias.unsqueeze(-1)
    return clamp_tone(gain * z + bias, backward)


def upsample_maps(maps: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Bilinear expansion of a GxG control field to the canonical page."""
    if maps.shape[-2:] == (height, width):
        return maps
    return torch.nn.functional.interpolate(maps, size=(height, width), mode="bilinear", align_corners=False)


def grid_edges(height: int, width: int, grid: int) -> tuple[np.ndarray, np.ndarray]:
    ye = np.rint(np.linspace(0, height, grid + 1)).astype(np.int64)
    xe = np.rint(np.linspace(0, width, grid + 1)).astype(np.int64)
    return ye, xe


def expand_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    ye, xe = grid_edges(height, width, labels.shape[0])
    yi = np.searchsorted(ye, np.arange(height), side="right") - 1
    xi = np.searchsorted(xe, np.arange(width), side="right") - 1
    return labels[yi[:, None], xi]


def compose_from_labels(dirty: np.ndarray, images: np.ndarray, labels: np.ndarray, mode: str) -> np.ndarray:
    """Hard mosaic, or the canonical hybrid: hard interiors plus a 5x5 guided boundary.

    `labels` are the controller's classes. The clean target is not read here.
    """
    _k, height, width = images.shape
    full = expand_labels(labels.astype(np.int64), (height, width))
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    hard = images[full, rows, cols]
    if mode == "hard":
        return np.clip(hard, 0, 1).astype(np.float32)
    if mode != "hybrid":
        raise RuntimeError(mode)
    boundary = np.zeros((height, width), np.uint8)
    boundary[1:] |= full[1:] != full[:-1]
    boundary[:, 1:] |= full[:, 1:] != full[:, :-1]
    band = cv2.dilate(boundary, np.ones((5, 5), np.uint8)).astype(bool)
    out = hard.copy()
    if not band.any():
        return np.clip(out, 0, 1).astype(np.float32)
    guide = np.ascontiguousarray(dirty, dtype=np.float32)
    box = lambda a: cv2.boxFilter(a, cv2.CV_32F, (9, 9), normalize=True, borderType=REFLECT)
    mean_i = box(guide)
    var_i = np.maximum(box(guide * guide) - mean_i * mean_i, 0)
    num = np.zeros((height, width), np.float32)
    den = np.zeros((height, width), np.float32)
    for label in np.unique(full):
        indicator = (full == label).astype(np.float32)
        mean_c = box(indicator)
        coeff = (box(guide * indicator) - mean_i * mean_c) / (var_i + 0.001)
        intercept = mean_c - coeff * mean_i
        weight = np.maximum(box(coeff) * guide + box(intercept), 0)
        num += weight * images[int(label)]
        den += weight
    blended = num / np.maximum(den, 1e-12)
    np.copyto(out, blended, where=band)
    return np.clip(out, 0, 1).astype(np.float32)


def cell_area_weights(height: int, width: int, grid: int) -> np.ndarray:
    """Mean-normalized cell area. Edge cells from rounded edges are smaller."""
    ye, xe = grid_edges(height, width, grid)
    area = (np.diff(ye)[:, None] * np.diff(xe)[None, :]).astype(np.float32)
    return area / max(float(area.mean()), 1e-12)


def interp_manifold(t: torch.Tensor, knots: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Piecewise-linear black(t), white(t) on a measured knot list.

    This is the deterministic map behind the 1-D controller. It does not fit a
    curve. The caller has to supply knots with status measured_from_bench03.
    """
    order = sorted(knots, key=lambda knot: float(knot["t"]))
    if len(order) < 2:
        raise RuntimeError("manifold needs at least two knots")
    ts = t.new_tensor([float(knot["t"]) for knot in order])
    blacks = t.new_tensor([float(knot["black"]) for knot in order])
    whites = t.new_tensor([float(knot["white"]) for knot in order])
    if torch.any(ts[1:] <= ts[:-1]):
        raise RuntimeError("manifold knot parameters are not strictly increasing")
    if float(ts[0]) < -1e-6 or float(ts[-1]) > 1 + 1e-6:
        raise RuntimeError("manifold knot parameters must lie in [0, 1]")
    if torch.any(whites < blacks + 1e-6) or torch.any(blacks < -1e-6) or torch.any(whites > 1 + 1e-6):
        raise RuntimeError("manifold knots leave the legal black/white range")
    flat = t.reshape(-1).clamp(float(ts[0]), float(ts[-1]))
    segment = torch.bucketize(flat, ts[1:]).clamp(max=len(order) - 2)
    t0, t1 = ts[segment], ts[segment + 1]
    alpha = (flat - t0) / (t1 - t0).clamp_min(1e-6)
    black = blacks[segment] * (1 - alpha) + blacks[segment + 1] * alpha
    white = whites[segment] * (1 - alpha) + whites[segment + 1] * alpha
    return black.view_as(t), white.view_as(t)


def cell_costs(image: np.ndarray, target: np.ndarray, grid: int) -> np.ndarray:
    height, width = target.shape
    ye, xe = grid_edges(height, width, grid)
    area = np.diff(ye)[:, None] * np.diff(xe)[None, :]
    integral = cv2.integral((image.astype(np.float32) - target) ** 2, sdepth=cv2.CV_64F)
    sums = (
        integral[ye[1:, None], xe[1:]]
        - integral[ye[:-1, None], xe[1:]]
        - integral[ye[1:, None], xe[:-1]]
        + integral[ye[:-1, None], xe[:-1]]
    )
    return (sums / np.maximum(area, 1)).astype(np.float32)
