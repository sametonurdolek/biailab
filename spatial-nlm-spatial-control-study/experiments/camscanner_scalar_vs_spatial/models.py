"""Bandwidth predictors. They do not output an image.

A scalar net is the same trunk with global pooling, so a win for the spatial
net is not explained by a different backbone. The last layer is initialized
near a constant h chosen on the validation sweep. Bounds are the sweep grid,
so the net cannot leave the range the fixed-h baseline was allowed to use.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual path keeps the activation scale alive. A plain stack of
        # ReLUs was shrinking to ~0.02 and the bandwidth map could not move.
        return x + self.conv2(self.act(self.conv1(self.act(x))))


class HNet(nn.Module):
    def __init__(self, width: int = 32, depth: int = 4, spatial: bool = True) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.stem = nn.Conv2d(1, width, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[_ResidualBlock(width) for _ in range(depth)])
        self.head = nn.Conv2d(width, 1, kernel_size=1)
        self.spatial = spatial
        # Large enough that a spatial residual exists at step 0. Exact zeros
        # block the trunk, because the bandwidth gradient passes through here.
        nn.init.normal_(self.head.weight, mean=0.0, std=0.2)
        nn.init.zeros_(self.head.bias)

    @property
    def trunk(self) -> nn.Sequential:
        return nn.Sequential(self.stem, self.blocks)

    def set_constant_h(self, h0: float, h_lo: float, h_hi: float) -> None:
        """Start as a flat map at h0, which must lie inside (h_lo, h_hi)."""
        if not (h_lo < h0 < h_hi):
            raise ValueError(f"h0={h0} is outside ({h_lo}, {h_hi})")
        fraction = (h0 - h_lo) / (h_hi - h_lo)
        fraction = min(max(fraction, 1e-4), 1 - 1e-4)
        bias = math.log(fraction / (1 - fraction))
        with torch.no_grad():
            self.head.bias.fill_(bias)

    def forward(self, noisy: torch.Tensor, h_lo: float, h_hi: float) -> torch.Tensor:
        features = self.blocks(self.stem(noisy))
        if not self.spatial:
            features = features.mean(dim=(2, 3), keepdim=True)
        raw = self.head(features)
        if not self.spatial:
            raw = raw.expand(-1, -1, noisy.shape[2], noisy.shape[3])
        unit = torch.sigmoid(raw)
        return h_lo + (h_hi - h_lo) * unit
