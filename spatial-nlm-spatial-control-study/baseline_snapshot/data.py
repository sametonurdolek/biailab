"""Kodak lossless images and the two noise families.

Split is by filename and is fixed before any score is computed:
kodim01-12 train, kodim13-16 validation, kodim17-24 test.

Stationary noise is AWGN with sigma = 25/255.
Spatial noise is a linear ramp between 15/255 and 45/255 along a random
axis and direction. The ramp is resampled every crop, so a network cannot
win by memorizing "the right half is noisier".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
KODAK_DIR = ROOT / "data" / "kodak"

TRAIN_IDS = tuple(range(1, 13))
VAL_IDS = tuple(range(13, 17))
TEST_IDS = tuple(range(17, 25))

SIGMA_AWGN = 25.0 / 255.0
SIGMA_LO = 15.0 / 255.0
SIGMA_HI = 45.0 / 255.0


def load_gray(image_id: int) -> torch.Tensor:
    path = KODAK_DIR / f"kodim{image_id:02d}.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def load_split(ids: tuple[int, ...]) -> list[torch.Tensor]:
    return [load_gray(i) for i in ids]


def ramp_sigma(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Linear sigma ramp. Axis and direction are random."""
    along_x = bool(rng.integers(0, 2))
    flip = bool(rng.integers(0, 2))
    length = width if along_x else height
    coord = np.linspace(0.0, 1.0, length, dtype=np.float32)
    if flip:
        coord = coord[::-1].copy()
    sigma_1d = SIGMA_LO + (SIGMA_HI - SIGMA_LO) * coord
    if along_x:
        return np.broadcast_to(sigma_1d[None, :], (height, width)).copy()
    return np.broadcast_to(sigma_1d[:, None], (height, width)).copy()


def add_awgn(clean: torch.Tensor, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    noise = rng.normal(0.0, SIGMA_AWGN, size=clean.shape).astype(np.float32)
    noisy = np.clip(clean.detach().contiguous().numpy() + noise, 0.0, 1.0)
    sigma = np.full(clean.shape, SIGMA_AWGN, dtype=np.float32)
    return torch.from_numpy(noisy), torch.from_numpy(sigma)


def add_spatial(clean: torch.Tensor, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    sigma = ramp_sigma(clean.shape[0], clean.shape[1], rng)
    noise = rng.normal(0.0, 1.0, size=clean.shape).astype(np.float32) * sigma
    noisy = np.clip(clean.detach().contiguous().numpy() + noise, 0.0, 1.0)
    return torch.from_numpy(noisy), torch.from_numpy(sigma)


def sample_crop(
    images: list[torch.Tensor],
    crop: int,
    protocol: str,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = images[int(rng.integers(0, len(images)))]
    height, width = image.shape
    if height < crop or width < crop:
        raise ValueError(f"crop {crop} does not fit image {height}x{width}")
    y = int(rng.integers(0, height - crop + 1))
    x = int(rng.integers(0, width - crop + 1))
    patch = image[y : y + crop, x : x + crop]
    if bool(rng.integers(0, 2)):
        patch = torch.flip(patch, dims=[1])
    if bool(rng.integers(0, 2)):
        patch = torch.flip(patch, dims=[0])
    if protocol == "awgn":
        noisy, sigma = add_awgn(patch, rng)
    elif protocol == "spatial":
        noisy, sigma = add_spatial(patch, rng)
    else:
        raise ValueError(protocol)
    return patch, noisy, sigma


def batch_crops(
    images: list[torch.Tensor],
    batch: int,
    crop: int,
    protocol: str,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cleans, noisies, sigmas = [], [], []
    for _ in range(batch):
        clean, noisy, sigma = sample_crop(images, crop, protocol, rng)
        cleans.append(clean)
        noisies.append(noisy)
        sigmas.append(sigma)
    return (
        torch.stack(cleans)[:, None],
        torch.stack(noisies)[:, None],
        torch.stack(sigmas)[:, None],
    )


def realize(
    clean: torch.Tensor,
    protocol: str,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if protocol == "awgn":
        noisy, sigma = add_awgn(clean, rng)
    elif protocol == "spatial":
        noisy, sigma = add_spatial(clean, rng)
    else:
        raise ValueError(protocol)
    return noisy, sigma
