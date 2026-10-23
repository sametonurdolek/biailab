"""CPU checks for the control package. This does not train and does not read the dataset."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.dataset import assert_allowed, letterbox
from shared.engine import load_measured_manifold
from shared.losses import LOSS_PRESETS, image_loss
from shared.metrics import page_metrics, recovery_ratio, structure_masks
from shared.models import ToneController
from shared.operator import (
    affine_from_endpoints,
    apply_bw,
    cell_costs,
    clamp_tone,
    compose_from_labels,
    endpoints_from_raw,
    gain_bias_from_raw,
    interp_manifold,
    normalize_s31,
    tone_affine_np,
    tone_bw_np,
    upsample_maps,
)
from shared.oracle import calibrate_tau, soft_targets
from shared.presets import BUILDERS
from shared.utils import read_json


def _check(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise SystemExit(f"SELFCHECK_FAIL {name} {detail}")
    print("PASS", name)


def main() -> None:
    torch.manual_seed(20261014)
    full = ToneController(1, 6, 128, lightweight=False, kind="spatial")
    count = sum(param.numel() for param in full.parameters())
    _check("fpn_parameter_count", count == 1_800_326, str(count))
    x = torch.rand(2, 1, 512, 512)
    mask = torch.ones(2, 1, 512, 512)
    geom = torch.tensor([[0, 0, 512, 512], [6, 8, 400, 480]], dtype=torch.float32)
    logits = full(x, mask, geom)
    _check("fpn_logits", tuple(logits.shape) == (2, 6, 128, 128), str(tuple(logits.shape)))

    global_net = ToneController(1, 2, 1, lightweight=False, kind="global")
    _check("global_shape", tuple(global_net(x, mask, geom).shape) == (2, 2))
    dual = ToneController(2, 2, 32, lightweight=True, kind="spatial")
    dual_out = dual(torch.rand(1, 2, 512, 512), mask[:1], geom[:1])
    _check("lightweight_dual", tuple(dual_out.shape) == (1, 2, 32, 32), str(tuple(dual_out.shape)))

    bounds = read_json(ROOT / "configs" / "bounds_fallback.json")
    raw = torch.randn(400, 2)
    black, white = endpoints_from_raw(raw, bounds, float(bounds["min_span"]))
    span = white - black
    _check(
        "endpoint_box",
        bool(torch.all(black >= -1e-6) and torch.all(black <= 0.30 + 1e-6)
             and torch.all(white <= 1 + 1e-5) and torch.all(white >= 0.70 - 1e-5)
             and torch.all(span >= float(bounds["min_span"]) - 1e-5)),
        f"black {black.min().item():.4f} {black.max().item():.4f} white {white.min().item():.4f} {white.max().item():.4f}",
    )
    gain, bias = gain_bias_from_raw(raw, bounds)
    _check(
        "gain_bias_box",
        bool(torch.all(gain >= 1 - 1e-6) and torch.all(gain <= 2.5 + 1e-6)
             and torch.all(bias >= -0.75 - 1e-6) and torch.all(bias <= 1e-6)),
    )
    z = np.linspace(0, 1, 21, dtype=np.float32)
    for black_np, white_np in ((0.10, 0.92), (0.22, 0.82), (0.02, 0.98)):
        gain_np, bias_np = affine_from_endpoints(black_np, white_np)
        same = np.allclose(
            tone_bw_np(z, black_np, white_np),
            tone_affine_np(z, gain_np, bias_np),
            atol=1e-6,
        )
        _check(f"affine_equivalence_{black_np}", same, f"a={gain_np:.6f} c={bias_np:.6f}")
    _check("v03_inside_gain_box", 1.0 <= affine_from_endpoints(0.10, 0.92)[0] <= 2.5)

    above = torch.tensor([2.0], requires_grad=True)
    clamp_tone(above, "straight_through").sum().backward()
    exact = torch.tensor([2.0], requires_grad=True)
    clamp_tone(exact, "exact").sum().backward()
    _check("straight_through_grad", above.grad.item() == 1.0, str(above.grad))
    _check("exact_clamp_grad", exact.grad.item() == 0.0, str(exact.grad))

    rng = np.random.default_rng(0)
    image = rng.random((40, 36), dtype=np.float32)
    normalized = normalize_s31(image)
    _check(
        "normalize_finite",
        normalized.shape == image.shape and np.isfinite(normalized).all()
        and normalized.min() >= -1e-6 and normalized.max() <= 1 + 1e-6
        and np.all(normalized + 1e-5 >= image),
    )
    target = np.clip(normalized * 0.9, 0, 1).astype(np.float32)
    metrics = page_metrics(normalized, target, structure_masks(target), grid=8)
    _check("metrics_finite", all(np.isfinite(metrics[key]) for key in ("psnr", "ssim", "mae", "gradient_error")))
    _check("recovery_half", recovery_ratio(27.5, 27.0, 28.0) == 0.5)
    _check("recovery_empty", recovery_ratio(27.0, 27.0, 27.0) is None)

    height = width = 32
    images = np.stack([np.full((height, width), 0.2, np.float32), np.full((height, width), 0.8, np.float32)])
    labels = np.zeros((4, 4), np.int64)
    labels[:, 2:] = 1
    dirty = np.full((height, width), 0.5, np.float32)
    hard = compose_from_labels(dirty, images, labels, "hard")
    hybrid = compose_from_labels(dirty, images, labels, "hybrid")
    _check("compose_interior", abs(float(hybrid[8, 2]) - 0.2) < 1e-5 and abs(float(hybrid[8, 30]) - 0.8) < 1e-5)
    _check("compose_band", float(np.max(np.abs(hybrid - hard))) > 1e-4, str(float(np.max(np.abs(hybrid - hard)))))
    costs = cell_costs(images[0], images[1], 4)
    _check("cell_costs_shape", costs.shape == (4, 4), str(costs.shape))
    regret, q = soft_targets(np.stack([costs, costs * 0.5]), 0.01)
    _check("soft_targets", q.shape == (2, 4, 4) and np.allclose(q.sum(0), 1, atol=1e-5))
    tau = calibrate_tau(np.array([0.0, 0.02, 0.04, 0.0], np.float32))
    _check("tau_median", abs(tau - 0.03) < 1e-8, str(tau))

    example = load_measured_manifold(ROOT / "configs" / "manifold.example.json")
    missing = load_measured_manifold(ROOT / "configs" / "manifold.json")
    _check("manifold_gate", example is None and missing is None)
    knots = [
        {"t": 0.0, "black": 0.02, "white": 0.98},
        {"t": 1.0, "black": 0.22, "white": 0.82},
    ]
    tone = torch.tensor([0.0, 0.5, 1.0])
    mb, mw = interp_manifold(tone, knots)
    _check("manifold_interp", abs(float(mb[1]) - 0.12) < 1e-5 and abs(float(mw[1]) - 0.90) < 1e-5)

    ids = []
    grids = []
    for index, builder in BUILDERS.items():
        cfg = builder(ROOT)
        ids.append(cfg["experiment_id"])
        grids.append(cfg["grid"])
        _check(f"preset_{index}_question", bool(cfg["research_question"]) and bool(cfg["stop_go"]))
    _check("ten_unique", len(ids) == 10 and len(set(ids)) == 10, str(ids))
    _check("grids", grids == [128, 128, 128, 128, 1, 32, 64, 128, 128, 128], str(grids))
    try:
        assert_allowed(["binarizeddataset::0000001::aaaaaaaaaa"], {"binarizeddataset::0000001::aaaaaaaaaa"})
        leaked = True
    except SystemExit:
        leaked = False
    _check("test_rejection", not leaked)

    bank = read_json(ROOT / "configs" / "tone_bank_bench01_s05.json")
    expected = [(0.02, 0.98), (0.05, 0.95), (0.10, 0.92), (0.15, 0.88), (0.22, 0.82)]
    got = [(item["black"], item["white"]) for item in bank["presets"]]
    _check("frozen_bank", got == expected and bank["bench03_present"] is False)
    _check("fallback_bounds", bounds["bench03_present"] is False and bounds["min_span"] == 0.05)
    _check(
        "loss_weights_explicit",
        LOSS_PRESETS["mse"] == {"mse": 1.0, "ssim": 0.0, "edge": 0.0, "smooth": 0.0}
        and set(LOSS_PRESETS) == {"mse", "mse_ssim", "mse_ssim_edge", "mse_structural_smooth"},
    )

    canvas, box, geom_np = letterbox(image)
    _check("letterbox", canvas.shape == (512, 512) and box.shape == (512, 512) and geom_np.shape == (4,))

    light = ToneController(1, 2, 16, lightweight=True, kind="spatial")
    light.train()
    light.set_stage("head")
    sample = torch.rand(1, 1, 512, 512)
    raw_maps = light(sample, torch.ones(1, 1, 512, 512), geom[:1]).permute(0, 2, 3, 1)
    pred_b, pred_w = endpoints_from_raw(raw_maps, bounds, 0.05)
    zed = torch.rand(48, 40)
    pred = apply_bw(
        zed,
        upsample_maps(pred_b[:, None], 48, 40)[0, 0],
        upsample_maps(pred_w[:, None], 48, 40)[0, 0],
        "straight_through",
        0.05,
    )
    loss = image_loss(pred[None], torch.rand(1, 48, 40), LOSS_PRESETS["mse"], pred_b, pred_w)
    loss["total"].backward()
    encoder_clean = all(param.grad is None for param in light.encoder.parameters())
    head_grad = light.classifier.weight.grad is not None and light.classifier.weight.grad.abs().sum().item() > 0
    _check("head_stage_grad", encoder_clean and head_grad)

    colabs = sorted((ROOT / "colabs").glob("*.py"))
    _check("colab_count", len(colabs) == 10, str(len(colabs)))
    for path in colabs:
        text = path.read_text(encoding="utf-8")
        ok = (
            'drive.mount("/content/drive")' in text
            and "ALLOW_TEST_SPLIT = False" in text
            and "DRIVE_ROOT" in text
            and "from shared.engine import run" in text
            and "Research question" in text
        )
        _check(f"colab_{path.name}", ok)
    print("SELFCHECK_OK")


if __name__ == "__main__":
    main()
