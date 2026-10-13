"""PSNR and SSIM on images scaled to [0, 1]. Peak is 1."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(restored: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB. restored/clean are (B, 1, H, W) or (H, W)."""
    restored, clean = _pair(restored, clean)
    mse = (restored - clean).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def ssim(restored: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Per-image mean SSIM, 11x11 Gaussian, the usual stabilization constants."""
    restored, clean = _pair(restored, clean)
    window = _gaussian_window(restored.device, restored.dtype)
    pad = window.shape[-1] // 2
    mu_x = F.conv2d(restored, window, padding=pad)
    mu_y = F.conv2d(clean, window, padding=pad)
    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(restored * restored, window, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(clean * clean, window, padding=pad) - mu_y2
    sigma_xy = F.conv2d(restored * clean, window, padding=pad) - mu_xy
    c1 = 0.01**2
    c2 = 0.03**2
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    score = num / den
    return score.mean(dim=(1, 2, 3))


def _pair(restored: torch.Tensor, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if restored.ndim == 2:
        restored = restored[None, None]
        clean = clean[None, None]
    elif restored.ndim == 3:
        restored = restored[None]
        clean = clean[None]
    return restored, clean


_WINDOW: torch.Tensor | None = None


def _gaussian_window(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    global _WINDOW
    if _WINDOW is None or _WINDOW.device != device or _WINDOW.dtype != dtype:
        coords = torch.arange(11, device=device, dtype=dtype) - 5
        kernel = torch.exp(-0.5 * (coords / 1.5) ** 2)
        kernel = kernel / kernel.sum()
        window = kernel[:, None] * kernel[None, :]
        _WINDOW = window.view(1, 1, 11, 11).contiguous()
    return _WINDOW
