"""Float32 losses. Weights are the caller's lambdas; nothing here is hidden."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def sobel_pair(x: torch.Tensor):
    kernel = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    padded = F.pad(x[:, None], (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(padded, kernel[None, None])[:, 0]
    gy = F.conv2d(padded, kernel.T[None, None])[:, 0]
    return gx, gy


def ssim_loss(y: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 minus the V2 Gaussian SSIM, crop 5, data range 1."""
    y = y[:, None]
    target = target[:, None]
    coords = torch.arange(11, device=y.device, dtype=y.dtype) - 5
    kernel = torch.exp(-0.5 * (coords / 1.5) ** 2)
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None, :]).view(1, 1, 11, 11)
    blur = lambda a: F.conv2d(F.pad(a, (5, 5, 5, 5), mode="reflect"), window)
    my, mt = blur(y), blur(target)
    vy = (blur(y * y) - my * my).clamp_min(0)
    vt = (blur(target * target) - mt * mt).clamp_min(0)
    cov = blur(y * target) - my * mt
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ss = ((2 * my * mt + c1) * (2 * cov + c2)) / ((my * my + mt * mt + c1) * (vy + vt + c2))
    if min(y.shape[-2:]) > 10:
        ss = ss[:, :, 5:-5, 5:-5]
    return 1 - ss.mean()


def image_loss(y: torch.Tensor, target: torch.Tensor, lambdas: dict, black=None, white=None) -> dict:
    y = y.float()
    target = target.float()
    mse = (y - target).square().mean()
    terms = {"mse": mse}
    if lambdas.get("ssim", 0):
        terms["ssim"] = ssim_loss(y, target)
    if lambdas.get("edge", 0):
        gx, gy = sobel_pair(y)
        tx, ty = sobel_pair(target)
        terms["edge"] = ((gx - tx).abs() + (gy - ty).abs()).mean() / 2
    if lambdas.get("smooth", 0) and black is not None:
        tv = (black[:, :, 1:] - black[:, :, :-1]).abs().mean() + (black[:, 1:, :] - black[:, :-1, :]).abs().mean()
        if white is not None:
            tv = tv + (white[:, :, 1:] - white[:, :, :-1]).abs().mean() + (white[:, 1:, :] - white[:, :-1, :]).abs().mean()
        terms["smooth"] = tv
    total = sum(terms[k] * float(lambdas.get(k, 0 if k != "mse" else 1)) for k in terms)
    terms["total"] = total
    return terms


def classification_loss(logits: torch.Tensor, q: torch.Tensor, regret: torch.Tensor, weight: torch.Tensor, lambdas: dict) -> dict:
    logp = F.log_softmax(logits.float(), dim=1)
    p = logp.exp()
    w = weight.float()
    denom = w.sum().clamp_min(1e-12)
    kl = ((torch.xlogy(q.float(), q.float().clamp_min(1e-12)) - q.float() * logp).sum(1) * w).sum() / denom
    reg = ((p * regret.float()).sum(1) * w).sum() / denom
    terms = {"kl": kl, "regret": reg}
    total = float(lambdas.get("kl", 1.0)) * kl + float(lambdas.get("regret", 0.0)) * reg
    terms["total"] = total
    return terms


LOSS_PRESETS = {
    "mse": {"mse": 1.0, "ssim": 0.0, "edge": 0.0, "smooth": 0.0},
    "mse_ssim": {"mse": 1.0, "ssim": 0.2, "edge": 0.0, "smooth": 0.0},
    "mse_ssim_edge": {"mse": 1.0, "ssim": 0.2, "edge": 0.1, "smooth": 0.0},
    "mse_structural_smooth": {"mse": 1.0, "ssim": 0.2, "edge": 0.1, "smooth": 0.01},
}
