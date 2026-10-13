"""Classical Non-Local Means with a spatial bandwidth map.

The estimator is Buades, Coll, Morel (CVPR 2005). Nothing in the weight
formula is learned. The only free field is h(i), the bandwidth at the
reference pixel:

    D(i, j) = mean_{k in patch, channels} (u(i+k) - u(j+k))^2
    w(i, j) = exp( - D(i, j) / h(i)^2 )
    u_hat(i) = sum_j w(i, j) u(j) / sum_j w(i, j)

j runs over the search window. Patch and search sizes are the classical
odd windows (default 7 and 21). A constant h is the same function with a
constant map, so a fixed-h baseline cannot win by using a different operator.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nlm(
    image: torch.Tensor,
    h_map: torch.Tensor,
    patch: int = 7,
    search: int = 21,
) -> torch.Tensor:
    """Denoise `image` with bandwidth `h_map`.

    image: (B, C, H, W) or (C, H, W) or (H, W), values on a linear scale.
    h_map: broadcastable to (B, 1, H, W), strictly positive, same units as image.
    """
    if patch % 2 == 0 or search % 2 == 0:
        raise ValueError("patch and search windows must be odd")
    image, squeezed = _as_bchw(image)
    h_map = _as_hmap(h_map, image)
    if not torch.isfinite(image).all() or not torch.isfinite(h_map).all():
        raise ValueError("nlm received a non-finite image or h map")

    _, channels, height, width = image.shape
    patch_radius = patch // 2
    search_radius = search // 2
    origin = search_radius + patch_radius
    padded = F.pad(image, (origin, origin, origin, origin), mode="reflect")
    patches = F.unfold(padded, kernel_size=patch)
    patch_elems = patches.shape[1]
    grid_h = height + 2 * search_radius
    grid_w = width + 2 * search_radius
    patches = patches.view(image.shape[0], patch_elems, grid_h, grid_w)
    center = patches[:, :, search_radius : search_radius + height, search_radius : search_radius + width]

    numerator = torch.zeros_like(image)
    denominator = torch.zeros(
        image.shape[0], 1, height, width, device=image.device, dtype=image.dtype
    )
    h2 = h_map.square().clamp_min(1e-12)
    pixel_origin = origin
    offsets = [
        (dy, dx)
        for dy in range(-search_radius, search_radius + 1)
        for dx in range(-search_radius, search_radius + 1)
    ]
    # The sum does not depend on group size. 64 MiB keeps each neighbor
    # stack inside one contiguous block on an 8 GiB laptop GPU; a 256 MiB
    # stack failed there even while the driver reported several GiB free.
    budget = 64 * 1024 * 1024
    chunk = max(1, budget // max(1, patch_elems * height * width * 4 * image.shape[0]))

    for start in range(0, len(offsets), chunk):
        group = offsets[start : start + chunk]
        neighbors = torch.stack(
            [
                patches[
                    :,
                    :,
                    search_radius + dy : search_radius + dy + height,
                    search_radius + dx : search_radius + dx + width,
                ]
                for dy, dx in group
            ],
            dim=1,
        )
        values = torch.stack(
            [
                padded[
                    :,
                    :,
                    pixel_origin + dy : pixel_origin + dy + height,
                    pixel_origin + dx : pixel_origin + dx + width,
                ]
                for dy, dx in group
            ],
            dim=1,
        )
        # D does not depend on h. Keeping it out of the graph is the same
        # gradient and drops the patch tensors before backward.
        with torch.no_grad():
            distance = (center[:, None] - neighbors).square().sum(dim=2) / float(patch_elems)
        weight = torch.exp(-distance / h2)
        numerator = numerator + (weight[:, :, None] * values).sum(dim=1)
        denominator = denominator + weight.sum(dim=1, keepdim=True)

    restored = numerator / denominator
    if squeezed == 0:
        return restored
    if squeezed == 1:
        return restored[0]
    return restored[0, 0]


def _as_bchw(image: torch.Tensor) -> tuple[torch.Tensor, int]:
    if image.ndim == 2:
        return image[None, None], 2
    if image.ndim == 3:
        return image[None], 1
    if image.ndim == 4:
        return image, 0
    raise ValueError(f"image must be HxW, CxHxW, or BxCxHxW, got {tuple(image.shape)}")


def _as_hmap(h_map: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    if h_map.ndim == 0:
        h_map = h_map.view(1, 1, 1, 1)
    elif h_map.ndim == 2:
        h_map = h_map[None, None]
    elif h_map.ndim == 3:
        h_map = h_map[:, None] if h_map.shape[0] == image.shape[0] else h_map[None]
    elif h_map.ndim != 4:
        raise ValueError(f"h_map rank {h_map.ndim} is not supported")
    h_map = h_map.to(device=image.device, dtype=image.dtype)
    return torch.broadcast_to(h_map, (image.shape[0], 1, image.shape[2], image.shape[3])).contiguous()


def _naive_numpy(image, h_map, patch: int, search: int):
    """Reference used only by the self-test. Same reflect pad and mean distance."""
    import numpy as np

    height, width = image.shape
    patch_radius = patch // 2
    search_radius = search // 2
    pad = search_radius + patch_radius
    big = np.pad(image, pad, mode="reflect")
    restored = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            numerator = 0.0
            denominator = 0.0
            h2 = float(h_map[i, j]) ** 2
            center_i = i + pad
            center_j = j + pad
            patch_i = big[center_i - patch_radius : center_i + patch_radius + 1, center_j - patch_radius : center_j + patch_radius + 1]
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    other = big[
                        center_i + dy - patch_radius : center_i + dy + patch_radius + 1,
                        center_j + dx - patch_radius : center_j + dx + patch_radius + 1,
                    ]
                    distance = float(np.mean((patch_i - other) ** 2))
                    weight = float(np.exp(-distance / h2))
                    numerator += weight * float(big[center_i + dy, center_j + dx])
                    denominator += weight
            restored[i, j] = numerator / denominator
    return restored


def _self_test() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    image_np = rng.random((9, 11)).astype(np.float32)
    h_np = (0.04 + 0.2 * rng.random((9, 11))).astype(np.float32)
    reference = _naive_numpy(image_np, h_np, patch=3, search=5)
    got = nlm(
        torch.from_numpy(image_np),
        torch.from_numpy(h_np),
        patch=3,
        search=5,
    ).cpu().numpy()
    err = float(np.max(np.abs(reference - got)))
    if err > 1e-5:
        raise AssertionError(f"NLM does not match the direct formula, max abs {err}")

    flat = torch.ones(1, 1, 16, 16)
    noisy = flat + 0.2
    tiny = nlm(noisy, torch.full_like(noisy, 1e-4), patch=3, search=5)
    wide = nlm(flat, torch.full((1, 1, 16, 16), 50.0), patch=3, search=5)
    if float((tiny - noisy).abs().mean()) > 1e-3:
        raise AssertionError("vanishing h did not reproduce the input")
    if float((wide - 1).abs().mean()) > 1e-4:
        raise AssertionError("huge h on a constant image was not the constant")

    varying = torch.linspace(0.02, 0.4, 16).view(1, 1, 1, 16).expand(1, 1, 16, 16).contiguous()
    varying = varying.detach().requires_grad_(True)
    restored = nlm(torch.rand(1, 1, 16, 16), varying, patch=3, search=5)
    restored.square().mean().backward()
    if varying.grad is None or float(varying.grad.abs().sum()) == 0.0:
        raise AssertionError("h map received no gradient")
    print(f"nlm self-test ok  max_abs={err:.2e}")


if __name__ == "__main__":
    _self_test()
