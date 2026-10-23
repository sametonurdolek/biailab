"""Offline targets. The clean page is used only here and in the loss, never as a network input."""
from __future__ import annotations

import numpy as np

from .operator import cell_costs, tone_bw_np


def soft_targets(costs: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    regret = costs - costs.min(0, keepdims=True)
    z = -regret / float(tau)
    z = z - z.max(0, keepdims=True)
    q = np.exp(z)
    q /= np.maximum(q.sum(0, keepdims=True), 1e-12)
    return regret.astype(np.float32), q.astype(np.float32)


def calibrate_tau(regret_samples: np.ndarray) -> float:
    """Tau is the median positive cell regret on the calibration pages.

    It is computed before training and then frozen. It is not a copied constant
    from the scale-selection experiment, whose MSE scale is different.
    """
    positive = regret_samples[regret_samples > 1e-12]
    if positive.size == 0:
        raise RuntimeError("calibration regret is uniformly zero; the tone bank has no local decision")
    tau = float(np.median(positive))
    if tau <= 0:
        raise RuntimeError("non-positive calibrated tau")
    return tau


def bank_costs(z: np.ndarray, target: np.ndarray, presets: list[dict], grid: int) -> np.ndarray:
    costs = []
    for preset in presets:
        image = tone_bw_np(z, preset["black"], preset["white"])
        costs.append(cell_costs(image, target, grid))
    return np.stack(costs).astype(np.float32)


def lattice(bounds: dict, step: float, min_span: float) -> list[dict]:
    """Uniform samples of the declared fallback box. A search grid, not an optimum."""
    blacks = np.arange(bounds["black_lo"], bounds["black_hi"] + 1e-9, step)
    whites = np.arange(bounds["white_lo"], bounds["white_hi"] + 1e-9, step)
    points = []
    for black in blacks:
        for white in whites:
            if white >= black + min_span - 1e-9:
                points.append({"black": float(black), "white": float(white)})
    if len(points) < 2:
        raise RuntimeError("oracle lattice is empty")
    return points


def best_page_endpoint(z: np.ndarray, target: np.ndarray, points: list[dict]) -> dict:
    best = None
    best_mse = None
    for point in points:
        image = tone_bw_np(z, point["black"], point["white"])
        mse = float(np.mean((image - target) ** 2, dtype=np.float64))
        if best_mse is None or mse < best_mse:
            best_mse = mse
            best = {**point, "mse": mse}
    return best


def best_cell_endpoints(z: np.ndarray, target: np.ndarray, points: list[dict], grid: int) -> tuple[np.ndarray, np.ndarray]:
    costs = bank_costs(z, target, points, grid)
    labels = costs.argmin(0)
    black = np.zeros(labels.shape, np.float32)
    white = np.zeros(labels.shape, np.float32)
    for index, point in enumerate(points):
        black[labels == index] = point["black"]
        white[labels == index] = point["white"]
    return black, white
