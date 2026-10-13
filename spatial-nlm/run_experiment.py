"""Can a learned spatial h improve classical NLM without replacing it?

Validation chooses every constant: the fixed bandwidth, the hand-crafted
gain, and the training checkpoint. The Kodak test split is scored once,
after those choices are frozen.

Protocols
  awgn     Gaussian noise, sigma = 25/255, same everywhere
  spatial  linear sigma ramp from 15/255 to 45/255, random axis and direction

Methods, all using the same NLM operator (patch 7, search 21)
  fixed        one h, chosen on validation
  hand         h(i) = clip(k * local_sigma(noisy)), k chosen on validation
  scalar       H-Net that emits one h per image
  spatial      H-Net that emits h(i)
  oracle_h     best constant h for that observation, using the clean image
  oracle_sigma clip(k * true sigma map), k from validation; uses the noise field
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import (
    SIGMA_AWGN,
    TEST_IDS,
    TRAIN_IDS,
    VAL_IDS,
    batch_crops,
    load_split,
    realize,
)
from metrics import psnr, ssim
from models import HNet
from nlm import _self_test, nlm

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PATCH = 7
SEARCH = 21
CROP = 128
BATCH = 4
STEPS = 400
VAL_EVERY = 50
LR = 3e-3
H_LO_INIT = 0.02
H_HI_INIT = 0.50
H_GRID_N = 12
K_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
TEST_REALIZATIONS = 2
SEEDS = (0,)


def log(message: str) -> None:
    print(message, flush=True)


def h_grid(lo: float = H_LO_INIT, hi: float = H_HI_INIT, n: int = H_GRID_N) -> list[float]:
    return [float(v) for v in np.geomspace(lo, hi, n)]


def local_sigma(image: torch.Tensor) -> torch.Tensor:
    """Immerkaer estimate, averaged over a 21x21 window. image is (B, 1, H, W)."""
    kernel = torch.tensor(
        [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    lap = F.conv2d(F.pad(image, (1, 1, 1, 1), mode="reflect"), kernel)
    radius = 10
    pooled = F.avg_pool2d(F.pad(lap.abs(), (radius, radius, radius, radius), mode="reflect"), 21, stride=1)
    return pooled * (math.sqrt(math.pi / 2.0) / 6.0)


def grad_mag(image: torch.Tensor) -> torch.Tensor:
    """image (H, W) -> magnitude of forward differences, padded to the same shape."""
    gy = torch.zeros_like(image)
    gx = torch.zeros_like(image)
    gy[:-1] = image[1:] - image[:-1]
    gx[:, :-1] = image[:, 1:] - image[:, :-1]
    return torch.sqrt(gx.square() + gy.square() + 1e-12)


def _finite_mean(values: list[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).any():
        return None
    return float(np.nanmean(array))


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if float(denom) < 1e-8:
        return float("nan")
    return float((a * b).sum() / denom)


def freeze(
    images: list[torch.Tensor],
    image_ids: tuple[int, ...] | list[int],
    protocol: str,
    seed: int,
    realizations: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    frozen = []
    for image_id, clean in zip(image_ids, images):
        observations = []
        for _ in range(realizations):
            noisy, sigma = realize(clean, protocol, rng)
            observations.append({"noisy": noisy, "sigma": sigma})
        frozen.append({"id": int(image_id), "clean": clean, "observations": observations})
    return frozen


def _nlm_image(noisy: torch.Tensor, h_map: torch.Tensor, device: torch.device) -> torch.Tensor:
    batch = noisy[None, None].to(device)
    bandwidth = h_map.to(device)

    def _run() -> torch.Tensor:
        return nlm(batch, bandwidth, PATCH, SEARCH)[0, 0].detach().cpu()

    try:
        return _run()
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        torch.cuda.empty_cache()
        return _run()


def score_pair(restored: torch.Tensor, clean: torch.Tensor) -> tuple[float, float]:
    with torch.no_grad():
        return float(psnr(restored, clean)[0]), float(ssim(restored, clean)[0])


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0)), "n": int(array.size)}


def summarize_by_image(per_image: list[float]) -> dict[str, float]:
    return mean_std(per_image)


def apply_fixed(noisy: torch.Tensor, h: float, device: torch.device) -> torch.Tensor:
    return _nlm_image(noisy, torch.tensor(h, dtype=torch.float32), device)


def apply_scaled_sigma(noisy: torch.Tensor, sigma: torch.Tensor, k: float, h_lo: float, h_hi: float, device: torch.device) -> torch.Tensor:
    h_map = (k * sigma).clamp(h_lo, h_hi)
    return _nlm_image(noisy, h_map, device)


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def apply_model(model: HNet, noisy: torch.Tensor, h_lo: float, h_hi: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    batch = noisy[None, None].to(device)

    def _run() -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            h_map = model(batch, h_lo, h_hi)
            restored = nlm(batch, h_map, PATCH, SEARCH)[0, 0].detach().cpu()
            return restored, h_map[0, 0].detach().cpu()

    try:
        return _run()
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        torch.cuda.empty_cache()
        return _run()


def evaluate_constant(
    frozen: list[dict],
    h: float,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    per_image_psnr = []
    per_image_ssim = []
    for item in frozen:
        psnr_obs = []
        ssim_obs = []
        for obs in item["observations"]:
            restored = apply_fixed(obs["noisy"], h, device)
            p, s = score_pair(restored, item["clean"])
            psnr_obs.append(p)
            ssim_obs.append(s)
        per_image_psnr.append(float(np.mean(psnr_obs)))
        per_image_ssim.append(float(np.mean(ssim_obs)))
    return summarize_by_image(per_image_psnr), summarize_by_image(per_image_ssim)


def evaluate_sigma_rule(
    frozen: list[dict],
    k: float,
    h_lo: float,
    h_hi: float,
    device: torch.device,
    use_true_sigma: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    per_image_psnr = []
    per_image_ssim = []
    for item in frozen:
        psnr_obs = []
        ssim_obs = []
        for obs in item["observations"]:
            if use_true_sigma:
                sigma = obs["sigma"]
            else:
                sigma = local_sigma(obs["noisy"][None, None].to(device))[0, 0].detach().cpu()
            restored = apply_scaled_sigma(obs["noisy"], sigma, k, h_lo, h_hi, device)
            p, s = score_pair(restored, item["clean"])
            psnr_obs.append(p)
            ssim_obs.append(s)
        per_image_psnr.append(float(np.mean(psnr_obs)))
        per_image_ssim.append(float(np.mean(ssim_obs)))
    return summarize_by_image(per_image_psnr), summarize_by_image(per_image_ssim)


def evaluate_model(
    model: HNet,
    frozen: list[dict],
    h_lo: float,
    h_hi: float,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    per_image_psnr = []
    per_image_ssim = []
    stds = []
    corr_sigma = []
    corr_grad = []
    means = []
    mechanism_mins = []
    mechanism_maxs = []
    for item in frozen:
        psnr_obs = []
        ssim_obs = []
        for obs in item["observations"]:
            restored, h_map = apply_model(model, obs["noisy"], h_lo, h_hi, device)
            p, s = score_pair(restored, item["clean"])
            psnr_obs.append(p)
            ssim_obs.append(s)
            stds.append(float(h_map.std()))
            means.append(float(h_map.mean()))
            corr_sigma.append(pearson(h_map, obs["sigma"]))
            corr_grad.append(pearson(h_map, grad_mag(obs["noisy"])))
            mechanism_mins.append(float(h_map.min()))
            mechanism_maxs.append(float(h_map.max()))
        per_image_psnr.append(float(np.mean(psnr_obs)))
        per_image_ssim.append(float(np.mean(ssim_obs)))
    mechanism = {
        "h_mean": float(np.mean(means)),
        "h_std": float(np.mean(stds)),
        "h_min": float(np.min(mechanism_mins)),
        "h_max": float(np.max(mechanism_maxs)),
        "corr_true_sigma": _finite_mean(corr_sigma),
        "corr_noisy_gradient": _finite_mean(corr_grad),
    }
    return summarize_by_image(per_image_psnr), summarize_by_image(per_image_ssim), mechanism


def oracle_constant(frozen: list[dict], grid: list[float], device: torch.device) -> dict[str, float]:
    """Best constant h per observation, using the clean target. An upper bound."""
    per_image = []
    for item in frozen:
        obs_scores = []
        for obs in item["observations"]:
            best = -1e9
            for h in grid:
                restored = apply_fixed(obs["noisy"], h, device)
                value = score_pair(restored, item["clean"])[0]
                if value > best:
                    best = value
            obs_scores.append(best)
        per_image.append(float(np.mean(obs_scores)))
    return summarize_by_image(per_image)


def sweep_values(values: list[float], score_fn) -> tuple[float, list[dict[str, float]]]:
    curve = []
    best_value = values[0]
    best_score = -1e9
    for value in values:
        started = time.perf_counter()
        score = score_fn(value)
        elapsed = time.perf_counter() - started
        row = {"value": value, "psnr": score, "seconds": round(elapsed, 2)}
        curve.append(row)
        log(f"  sweep value={value:.4f}  val_psnr={score:.3f}  ({elapsed:.1f}s)")
        if score > best_score:
            best_score = score
            best_value = value
    return best_value, curve


def extend_grid_if_endpoint(grid: list[float], best: float) -> list[float]:
    if best == grid[0]:
        extra = [float(v) for v in np.geomspace(grid[0] / 2.0, grid[0], 5, endpoint=False)]
        return extra + grid
    if best == grid[-1]:
        extra = [float(v) for v in np.geomspace(grid[-1], grid[-1] * 2.0, 5)[1:]]
        return grid + extra
    return grid


def _center_initial_h(model: HNet, h_star: float, h_lo: float, h_hi: float, device: torch.device) -> None:
    """Keep the mean bandwidth at h* while leaving the small random head intact."""
    model.set_constant_h(h_star, h_lo, h_hi)
    span = h_hi - h_lo
    target = min(max((h_star - h_lo) / span, 1e-4), 1.0 - 1e-4)
    target_logit = math.log(target / (1.0 - target))
    with torch.no_grad():
        for _ in range(4):
            sample = torch.rand(4, 1, 128, 128, device=device)
            predicted = model(sample, h_lo, h_hi)
            fraction = ((predicted.mean() - h_lo) / span).clamp(1e-4, 1.0 - 1e-4)
            current = math.log(float(fraction) / (1.0 - float(fraction)))
            model.head.bias += target_logit - current


def train_model(
    spatial: bool,
    protocol: str,
    train_images: list[torch.Tensor],
    val_frozen: list[dict],
    h_star: float,
    h_lo: float,
    h_hi: float,
    device: torch.device,
    seed: int,
) -> tuple[HNet, list[dict[str, float]]]:
    torch.manual_seed(seed)
    model = HNet(spatial=spatial).to(device)
    _center_initial_h(model, h_star, h_lo, h_hi, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed)
    best_score = -1e9
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    curve: list[dict[str, float]] = []
    name = "spatial" if spatial else "scalar"

    init_psnr, _, _ = evaluate_model(model, val_frozen, h_lo, h_hi, device)
    fixed_psnr, _ = evaluate_constant(val_frozen, h_star, device)
    log(f"{protocol}/{name} init val {init_psnr['mean']:.3f}  fixed h* val {fixed_psnr['mean']:.3f}")
    if abs(init_psnr["mean"] - fixed_psnr["mean"]) > 0.3:
        raise RuntimeError(
            f"H-Net initialization is not the constant-h baseline "
            f"({init_psnr['mean']:.3f} vs {fixed_psnr['mean']:.3f})"
        )

    for step in range(1, STEPS + 1):
        model.train()
        clean, noisy, _sigma = batch_crops(train_images, BATCH, CROP, protocol, rng)
        clean = clean.to(device)
        noisy = noisy.to(device)
        restored = h_map = loss = None
        for attempt in (1, 2):
            try:
                h_map = model(noisy, h_lo, h_hi)
                restored = nlm(noisy, h_map, PATCH, SEARCH)
                loss = (restored - clean).abs().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                break
            except RuntimeError as exc:
                optimizer.zero_grad(set_to_none=True)
                restored = h_map = loss = None
                torch.cuda.empty_cache()
                if not _is_oom(exc) or attempt == 2:
                    raise
                log(f"{protocol}/{name} seed={seed} step={step} oom, retrying once")
        trunk_grad = 0.0
        for parameter in model.trunk.parameters():
            if parameter.grad is not None:
                trunk_grad += float(parameter.grad.detach().square().sum())
        loss_value = float(loss.detach())
        del restored, h_map, loss, clean, noisy
        if step % 10 == 0:
            torch.cuda.empty_cache()

        if step == STEPS // 2:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5

        if step == 1 or step % VAL_EVERY == 0 or step == STEPS:
            torch.cuda.empty_cache()
            val_psnr, _, mechanism = evaluate_model(model, val_frozen, h_lo, h_hi, device)
            row = {
                "step": step,
                "loss": loss_value,
                "val_psnr": val_psnr["mean"],
                "h_mean": mechanism["h_mean"],
                "h_std": mechanism["h_std"],
                "trunk_grad": trunk_grad,
            }
            curve.append(row)
            log(
                f"{protocol}/{name} seed={seed} step={step} loss={row['loss']:.4f} "
                f"val_psnr={row['val_psnr']:.3f} h_mean={row['h_mean']:.4f} "
                f"h_std={row['h_std']:.6f} trunk_grad={trunk_grad:.3e} "
                f"head_w={float(model.head.weight.detach().abs().mean()):.5f}"
            )
            if row["val_psnr"] > best_score:
                best_score = row["val_psnr"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, RESULTS / f"{protocol}_{name}.pt")

    model.load_state_dict(best_state)
    return model, curve


def opencv_reference(noisy: torch.Tensor, clean: torch.Tensor) -> dict[str, float]:
    import cv2

    source = np.clip(np.round(noisy.numpy() * 255.0), 0, 255).astype(np.uint8)
    best = {"h": None, "psnr": -1e9}
    for h in (3, 5, 7, 10, 12, 15, 20, 30):
        restored = cv2.fastNlMeansDenoising(source, None, h, PATCH, SEARCH)
        value = score_pair(torch.from_numpy(restored.astype(np.float32) / 255.0), clean)[0]
        if value > best["psnr"]:
            best = {"h": h, "psnr": value}
    return best


def save_figure(
    path: Path,
    top: list[tuple[str, torch.Tensor | None]],
    bottom: list[tuple[str, torch.Tensor | None]],
) -> None:
    """Top row is images in [0, 1]. Bottom row is bandwidth maps on one scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = len(top)
    figure, axes = plt.subplots(2, columns, figsize=(2.8 * columns, 6.4), dpi=110, layout="constrained")
    for column, (name, image) in enumerate(top):
        axes[0, column].axis("off")
        axes[0, column].set_title(name, fontsize=9)
        if image is not None:
            axes[0, column].imshow(image.numpy(), cmap="gray", vmin=0, vmax=1)
    finite = [panel.numpy() for _, panel in bottom if panel is not None]
    vmin = min(float(array.min()) for array in finite)
    vmax = max(float(array.max()) for array in finite)
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    last = None
    for column, (name, h_map) in enumerate(bottom):
        axes[1, column].axis("off")
        axes[1, column].set_title(name, fontsize=9)
        if h_map is not None:
            last = axes[1, column].imshow(h_map.numpy(), cmap="magma", vmin=vmin, vmax=vmax)
    if last is not None:
        figure.colorbar(last, ax=axes[1, :].ravel().tolist(), fraction=0.015, pad=0.01)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def bench(device: torch.device) -> None:
    image = torch.rand(1, 1, CROP, CROP, device=device)
    h_map = torch.full((1, 1, CROP, CROP), 0.1, device=device, requires_grad=True)
    for _ in range(2):
        nlm(image, h_map, PATCH, SEARCH)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(3):
        restored = nlm(image, h_map, PATCH, SEARCH)
        restored.square().mean().backward()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) / 3
    log(f"bench crop {CROP} batch implicit 1  forward+backward {elapsed:.3f}s")
    full = torch.rand(1, 1, 512, 768, device=device)
    h_full = torch.full((1, 1, 512, 768), 0.1, device=device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        nlm(full, h_full, PATCH, SEARCH)
    torch.cuda.synchronize()
    log(f"bench full 512x768 forward {(time.perf_counter() - started):.3f}s")


def run_protocol(
    protocol: str,
    train_images: list[torch.Tensor],
    val_images: list[torch.Tensor],
    test_images: list[torch.Tensor],
    device: torch.device,
) -> dict:
    val_seed = 1000 if protocol == "awgn" else 2000
    test_seed = 3000 if protocol == "awgn" else 4000
    val_frozen = freeze(val_images, VAL_IDS, protocol, val_seed, realizations=1)
    test_frozen = freeze(test_images, TEST_IDS, protocol, test_seed, realizations=TEST_REALIZATIONS)

    grid = h_grid()
    log(f"=== {protocol} fixed-h sweep on val ===")
    best_h, curve = sweep_values(
        grid,
        lambda h: evaluate_constant(val_frozen, h, device)[0]["mean"],
    )
    extended = extend_grid_if_endpoint(grid, best_h)
    if extended != grid:
        log(f"{protocol} best h sat on the grid edge; extending once")
        best_h, curve = sweep_values(
            extended,
            lambda h: evaluate_constant(val_frozen, h, device)[0]["mean"],
        )
        grid = extended
    h_lo = float(min(grid))
    h_hi = float(max(grid))
    if best_h <= h_lo:
        h_lo = best_h * 0.5
    if best_h >= h_hi:
        h_hi = best_h * 2.0
    log(f"{protocol} h* = {best_h:.4f}  bounds [{h_lo:.4f}, {h_hi:.4f}]")

    log(f"=== {protocol} hand-crafted k sweep on val ===")
    best_k, k_curve = sweep_values(
        list(K_GRID),
        lambda k: evaluate_sigma_rule(val_frozen, k, h_lo, h_hi, device, use_true_sigma=False)[0]["mean"],
    )
    log(f"=== {protocol} oracle-sigma k sweep on val ===")
    oracle_k, oracle_k_curve = sweep_values(
        list(K_GRID),
        lambda k: evaluate_sigma_rule(val_frozen, k, h_lo, h_hi, device, use_true_sigma=True)[0]["mean"],
    )

    reference = opencv_reference(val_frozen[0]["observations"][0]["noisy"], val_frozen[0]["clean"])
    ours_same = score_pair(apply_fixed(val_frozen[0]["observations"][0]["noisy"], best_h, device), val_frozen[0]["clean"])[0]
    log(f"{protocol} opencv best h={reference['h']} psnr={reference['psnr']:.3f}  our fixed on same image {ours_same:.3f}")

    models = {}
    curves = {}
    for spatial in (False, True):
        name = "spatial" if spatial else "scalar"
        train_seed = (10 if protocol == "awgn" else 20) + (1 if spatial else 0) + 100 * SEEDS[0]
        model, curve_rows = train_model(
            spatial, protocol, train_images, val_frozen, best_h, h_lo, h_hi, device, train_seed
        )
        models[name] = model
        curves[name] = curve_rows
        torch.save(model.state_dict(), RESULTS / f"{protocol}_{name}.pt")

    log(f"=== {protocol} test ===")
    report: dict = {
        "h_star": best_h,
        "h_lo": h_lo,
        "h_hi": h_hi,
        "h_curve_val": curve,
        "k_star": best_k,
        "k_curve_val": k_curve,
        "oracle_k": oracle_k,
        "oracle_k_curve_val": oracle_k_curve,
        "opencv_val0": reference,
        "our_fixed_val0": ours_same,
        "train": curves,
        "test": {},
        "mechanism": {},
    }
    fixed_p, fixed_s = evaluate_constant(test_frozen, best_h, device)
    hand_p, hand_s = evaluate_sigma_rule(test_frozen, best_k, h_lo, h_hi, device, use_true_sigma=False)
    oracle_s_p, oracle_s_s = evaluate_sigma_rule(test_frozen, oracle_k, h_lo, h_hi, device, use_true_sigma=True)
    log(f"{protocol} oracle constant-h sweep on test (upper bound)")
    oracle_h = oracle_constant(test_frozen, grid, device)
    noisy_psnr = []
    noisy_ssim = []
    for item in test_frozen:
        p_obs, s_obs = [], []
        for obs in item["observations"]:
            p, s = score_pair(obs["noisy"], item["clean"])
            p_obs.append(p)
            s_obs.append(s)
        noisy_psnr.append(float(np.mean(p_obs)))
        noisy_ssim.append(float(np.mean(s_obs)))
    scalar_p, scalar_s, scalar_m = evaluate_model(models["scalar"], test_frozen, h_lo, h_hi, device)
    spatial_p, spatial_s, spatial_m = evaluate_model(models["spatial"], test_frozen, h_lo, h_hi, device)
    report["test"] = {
        "noisy": {"psnr": summarize_by_image(noisy_psnr), "ssim": summarize_by_image(noisy_ssim)},
        "fixed": {"psnr": fixed_p, "ssim": fixed_s},
        "handcrafted": {"psnr": hand_p, "ssim": hand_s},
        "oracle_constant_h": {"psnr": oracle_h},
        "oracle_true_sigma": {"psnr": oracle_s_p, "ssim": oracle_s_s},
        "scalar": {"psnr": scalar_p, "ssim": scalar_s},
        "spatial": {"psnr": spatial_p, "ssim": spatial_s},
    }
    report["mechanism"] = {"scalar": scalar_m, "spatial": spatial_m}
    for name, block in report["test"].items():
        log(f"{protocol} test {name:20s} psnr {block['psnr']['mean']:.3f}")

    illustrate(protocol, models, best_h, best_k, oracle_k, h_lo, h_hi, device)
    return report


def illustrate(protocol, models, best_h, best_k, oracle_k, h_lo, h_hi, device) -> None:
    """kodim17, one precommitted ramp direction for the picture only."""
    clean = load_split((17,))[0]
    rng = np.random.default_rng(17 if protocol == "spatial" else 18)
    # realize() randomizes ramp direction. For the figure, build the ramp
    # explicitly so the left side is the low-noise side and the picture is readable.
    if protocol == "spatial":
        width = clean.shape[1]
        coord = np.linspace(0.0, 1.0, width, dtype=np.float32)
        sigma_np = (15.0 / 255.0) + ((45.0 - 15.0) / 255.0) * coord
        sigma_np = np.broadcast_to(sigma_np[None, :], clean.shape).copy()
        noise = rng.normal(0.0, 1.0, size=clean.shape).astype(np.float32) * sigma_np
        noisy = torch.from_numpy(np.clip(clean.numpy() + noise, 0.0, 1.0))
        sigma = torch.from_numpy(sigma_np)
    else:
        noisy, sigma = realize(clean, protocol, rng)

    fixed = apply_fixed(noisy, best_h, device)
    hand_sigma = local_sigma(noisy[None, None].to(device))[0, 0].detach().cpu()
    hand = apply_scaled_sigma(noisy, hand_sigma, best_k, h_lo, h_hi, device)
    scalar, scalar_h = apply_model(models["scalar"], noisy, h_lo, h_hi, device)
    spatial, spatial_h = apply_model(models["spatial"], noisy, h_lo, h_hi, device)
    fixed_h = torch.full_like(spatial_h, best_h)
    true_h = (oracle_k * sigma).clamp(h_lo, h_hi)

    hand_h = (best_k * hand_sigma).clamp(h_lo, h_hi)
    save_figure(
        RESULTS / f"figure_{protocol}_kodim17.png",
        [
            ("clean", clean),
            ("noisy", noisy),
            ("fixed h", fixed),
            ("hand", hand),
            ("scalar h", scalar),
            ("spatial h", spatial),
        ],
        [
            ("", None),
            ("k * true sigma", true_h),
            (f"h={best_h:.3f}", fixed_h),
            ("k * local sigma", hand_h),
            ("scalar h", scalar_h),
            ("learned h(x,y)", spatial_h),
        ],
    )


def _awgn_block_complete(path: Path) -> bool:
    """True only when the stationary protocol was scored and saved in full."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("steps") != STEPS or payload.get("seeds") != list(SEEDS):
            return False
        test = payload["protocols"]["awgn"]["test"]
        needed = (
            "noisy",
            "fixed",
            "handcrafted",
            "oracle_constant_h",
            "oracle_true_sigma",
            "scalar",
            "spatial",
        )
        if not all("mean" in test[name]["psnr"] for name in needed):
            return False
        if "h_std" not in payload["protocols"]["awgn"]["mechanism"]["spatial"]:
            return False
    except (KeyError, TypeError, json.JSONDecodeError, OSError):
        return False
    return (
        (RESULTS / "awgn_scalar.pt").is_file()
        and (RESULTS / "awgn_spatial.pt").is_file()
        and (RESULTS / "figure_awgn_kodim17.png").is_file()
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    log("nlm formula check")
    _self_test()
    if not torch.cuda.is_available():
        raise RuntimeError("This run expects the local CUDA device")
    device = torch.device("cuda")
    probe = HNet(spatial=True)
    log(f"device {torch.cuda.get_device_name(0)}")
    log(f"spatial H-Net parameters {sum(p.numel() for p in probe.parameters())}")
    bench(device)
    if len(sys.argv) > 1 and sys.argv[1] == "--bench":
        return

    train_images = load_split(TRAIN_IDS)
    val_images = load_split(VAL_IDS)
    test_images = load_split(TEST_IDS)
    payload = {
        "question": "Can learned spatial control improve NLM without replacing the classical algorithm?",
        "operator": {
            "patch": PATCH,
            "search": SEARCH,
            "distance": "mean squared difference over the patch",
            "weight": "exp(-D / h(i)^2)",
            "h_depends_on": "reference pixel only",
        },
        "split": {"train": list(TRAIN_IDS), "val": list(VAL_IDS), "test": list(TEST_IDS)},
        "noise": {"awgn_sigma": SIGMA_AWGN, "spatial_sigma_uint8": [15, 45]},
        "steps": STEPS,
        "crop": CROP,
        "seeds": list(SEEDS),
        "protocols": {},
    }
    protocols = ["awgn", "spatial"]
    metrics_path = RESULTS / "metrics.json"
    if metrics_path.exists() and _awgn_block_complete(metrics_path):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        protocols = ["spatial"]
        log("awgn block in metrics.json is complete; scoring spatial noise only")
    for protocol in protocols:
        payload["protocols"][protocol] = run_protocol(
            protocol, train_images, val_images, test_images, device
        )
        (RESULTS / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log(f"wrote partial metrics after {protocol}")
    (RESULTS / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FAILED {type(exc).__name__}: {exc}")
        raise
