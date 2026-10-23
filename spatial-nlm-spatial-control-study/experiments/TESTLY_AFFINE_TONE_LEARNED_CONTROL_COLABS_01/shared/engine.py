"""Train and validate one tone-control experiment.

The network emits controls. Pixels come from the frozen s=31 operator and the
exact affine clip. The test split is a rejection list and is never opened.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .dataset import (
    PageStore,
    assert_allowed,
    calibration_development,
    find_split,
    index_allowed_pairs,
    letterbox_stack,
    load_split,
    spec_hash,
)
from .losses import LOSS_PRESETS, classification_loss, image_loss
from .metrics import mean_metrics, page_metrics, recovery_ratio
from .models import ToneController, load_encoder_warm_start
from .operator import (
    apply_bw,
    apply_gain_bias,
    cell_area_weights,
    cell_costs,
    clamp_tone,
    compose_from_labels,
    endpoints_from_raw,
    gain_bias_from_raw,
    interp_manifold,
    normalize_paper,
    tone_bw_np,
    upsample_maps,
)
from .oracle import bank_costs, best_cell_endpoints, best_page_endpoint, calibrate_tau, lattice, soft_targets
from .utils import (
    digest,
    file_sha256,
    fmt,
    gpu_vram_gb,
    read_json,
    recommend_batch,
    recovery_label,
    seed_all,
    write_history,
    write_json,
)

REFERENCE_VERSION = "affine_tone_colab_01"
CURRENT6 = (15, 21, 31, 45, 63, 91)
RECOVERY_CAVEAT = (
    "PSNR is logarithmic in MSE, so a recovery ratio is not a linear fraction of "
    "removed error or of visual quality."
)


def _tqdm(iterable, **kwargs):
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def load_measured_manifold(path: Path) -> dict | None:
    """Accept only a BENCH_03 measurement. The example file is not a manifold."""
    path = Path(path)
    if not path.is_file():
        return None
    data = read_json(path)
    if data.get("status") != "measured_from_bench03":
        return None
    knots = data.get("knots") or []
    if len(knots) < 2:
        return None
    return data


def _input_channels(mode: str) -> int:
    if mode in ("x", "z"):
        return 1
    if mode == "xz":
        return 2
    raise SystemExit(f"UNKNOWN_INPUT_MODE {mode}")


def _channels(page: dict, mode: str) -> list[np.ndarray]:
    if mode == "x":
        return [page["dirty"]]
    if mode == "z":
        return [page["Z"]]
    if mode == "xz":
        return [page["dirty"], page["Z"]]
    raise SystemExit(f"UNKNOWN_INPUT_MODE {mode}")


def _schedule_for(epoch: int, schedule: list[dict]) -> dict:
    for row in schedule:
        if epoch <= int(row["until_epoch"]):
            return row
    return schedule[-1]


def _allocate_dir(drive_root: Path, experiment_id: str, resume: bool) -> Path:
    root = drive_root / "Testly_Affine_Tone_DL"
    root.mkdir(parents=True, exist_ok=True)
    base = root / experiment_id

    def nonempty(path: Path) -> bool:
        return path.exists() and any(path.iterdir())

    if resume or not nonempty(base):
        return base
    for index in range(2, 100):
        candidate = root / f"{experiment_id}_{index}"
        if not nonempty(candidate):
            return candidate
    raise SystemExit("NO_FREE_EXPERIMENT_DIR")


def _reference(path: Path) -> dict:
    if path.is_file():
        data = read_json(path)
        if data.get("version") == REFERENCE_VERSION:
            data.setdefault("pages", {})
            return data
    return {"version": REFERENCE_VERSION, "pages": {}}


def _masks(page: dict) -> dict:
    return {"ink": page["ink"], "edge": page["edge"], "bg": page["bg"]}


def _tone_stack(z: np.ndarray, presets: list[dict]) -> np.ndarray:
    return np.stack([tone_bw_np(z, item["black"], item["white"]) for item in presets], 0)


def _render_fixed(page: dict, black: float, white: float) -> np.ndarray:
    return tone_bw_np(page["Z"], black, white)


def _render_bank_page(page: dict, presets: list[dict]) -> np.ndarray:
    best_image = None
    best_mse = None
    for preset in presets:
        image = tone_bw_np(page["Z"], preset["black"], preset["white"])
        mse = float(np.mean((image - page["T"]) ** 2, dtype=np.float64))
        if best_mse is None or mse < best_mse:
            best_mse = mse
            best_image = image
    return best_image


def _render_bank_hybrid(page: dict, presets: list[dict], grid: int) -> np.ndarray:
    images = _tone_stack(page["Z"], presets)
    costs = np.stack([cell_costs(images[i], page["T"], grid) for i in range(len(presets))], 0)
    labels = costs.argmin(0).astype(np.int64)
    return compose_from_labels(page["dirty"], images, labels, "hybrid")


def _render_current6(page: dict) -> np.ndarray:
    images = np.stack([normalize_paper(page["dirty"], scale) for scale in CURRENT6], 0)
    costs = np.stack([cell_costs(images[i], page["T"], 128) for i in range(len(CURRENT6))], 0)
    labels = costs.argmin(0).astype(np.int64)
    return compose_from_labels(page["dirty"], images, labels, "hybrid")


def _metric_row(image: np.ndarray, page: dict, grid: int) -> dict:
    return page_metrics(image, page["T"], _masks(page), grid=grid)


def _fill_reference(path: Path, ids: list[str], load_page, render, grid: int = 128) -> dict:
    data = _reference(path)
    changed = False
    for pid in _tqdm(ids, desc=path.stem):
        if pid in data["pages"]:
            continue
        page = load_page(pid)
        data["pages"][pid] = _metric_row(render(page), page, grid)
        del page
        changed = True
        if len(data["pages"]) % 10 == 0:
            write_json(path, data)
    if changed:
        write_json(path, data)
    return data


def _optimizer_step(optimizer, scaler, model, clip: float, use_amp: bool) -> None:
    if use_amp:
        scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def _mean_ids(data: dict, ids: list[str]) -> dict:
    rows = [data["pages"][pid] for pid in ids if pid in data["pages"]]
    if len(rows) != len(ids):
        missing = len(ids) - len(rows)
        return {"available": False, "missing": missing}
    out = mean_metrics(rows)
    out["available"] = True
    out["pages"] = len(rows)
    return out


class ControlDataset(Dataset):
    def __init__(self, ids, pairs, store, test_ids, cfg, presets, points, tau, augment):
        self.ids = list(ids)
        self.pairs = pairs
        self.store = store
        self.test_ids = test_ids
        self.cfg = cfg
        self.presets = presets
        self.points = points
        self.tau = tau
        self.augment = augment
        assert_allowed(self.ids, test_ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        pid = self.ids[index]
        pair = self.pairs[pid]
        page = self.store.ensure(pid, pair, self.test_ids)
        dirty = page["dirty"]
        zed = page["Z"]
        target = page["T"]
        masks = _masks(page)
        flip = self.augment and random.random() < 0.5
        if flip:
            dirty = np.ascontiguousarray(dirty[:, ::-1])
            zed = np.ascontiguousarray(zed[:, ::-1])
            target = np.ascontiguousarray(target[:, ::-1])
            masks = {key: np.ascontiguousarray(value[:, ::-1]) for key, value in masks.items()}
        view = {"dirty": dirty, "Z": zed, "T": target, **masks}
        canvas, mask, geom = letterbox_stack(_channels(view, self.cfg["input_mode"]))
        item = {
            "id": pid,
            "x": canvas,
            "mask": mask[None].astype(np.float32),
            "geom": geom,
            "page": view,
        }
        grid = int(self.cfg["grid"])
        if self.cfg["family"] == "discrete":
            if flip:
                costs = bank_costs(zed, target, self.presets, grid)
            else:
                costs = _load_costs(self.cfg, pid, pair["dirty_sha"])
            regret, q = soft_targets(costs, self.tau)
            item["q"] = q
            item["regret"] = regret
            item["weight"] = cell_area_weights(*zed.shape, grid)
            item["costs"] = costs
        if self.cfg["supervision"] == "oracle_param":
            black, white = _load_oracle_params(self.cfg, pid, pair["dirty_sha"], self.points, view, flip)
            item["black_star"] = black
            item["white_star"] = white
        return item


def _collate(items: list[dict]) -> dict:
    batch = {
        "id": [item["id"] for item in items],
        "x": torch.from_numpy(np.stack([item["x"] for item in items])),
        "mask": torch.from_numpy(np.stack([item["mask"] for item in items])),
        "geom": torch.from_numpy(np.stack([item["geom"] for item in items])),
        "pages": [item["page"] for item in items],
    }
    for key in ("q", "regret", "weight", "black_star", "white_star"):
        if key in items[0]:
            batch[key] = torch.from_numpy(np.stack([item[key] for item in items]))
    if "costs" in items[0]:
        batch["costs"] = np.stack([item["costs"] for item in items])
    return batch


def _cost_path(cfg: dict, pid: str, dirty_sha: str) -> Path:
    folder = Path(cfg["cost_dir"])
    name = digest({"id": pid, "dirty_sha": dirty_sha, "bank": cfg["bank_hash"], "grid": cfg["grid"]})
    return folder / f"{name}.npz"


def _npz_text(value) -> str:
    if isinstance(value, np.ndarray):
        return str(value.item())
    return str(value)


def _load_costs(cfg: dict, pid: str, dirty_sha: str) -> np.ndarray:
    path = _cost_path(cfg, pid, dirty_sha)
    with np.load(path, allow_pickle=False) as blob:
        if _npz_text(blob["operator_hash"]) != cfg["operator_hash"] or _npz_text(blob["bank_hash"]) != cfg["bank_hash"]:
            raise SystemExit(f"COST_CACHE_MISMATCH {pid}")
        return np.ascontiguousarray(blob["costs"], dtype=np.float32)


def _oracle_path(cfg: dict, pid: str, dirty_sha: str) -> Path:
    folder = Path(cfg["oracle_dir"])
    name = digest({
        "id": pid,
        "dirty_sha": dirty_sha,
        "lattice": cfg["lattice_hash"],
        "grid": cfg["grid"],
        "level": cfg["oracle_param_level"],
    })
    return folder / f"{name}.npz"


def _load_oracle_params(cfg, pid, dirty_sha, points, view, flip):
    if flip:
        return _oracle_arrays(cfg, view, points)
    path = _oracle_path(cfg, pid, dirty_sha)
    with np.load(path, allow_pickle=False) as blob:
        return (
            np.ascontiguousarray(blob["black"], dtype=np.float32),
            np.ascontiguousarray(blob["white"], dtype=np.float32),
        )


def _oracle_arrays(cfg: dict, view: dict, points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if cfg["oracle_param_level"] == "page":
        best = best_page_endpoint(view["Z"], view["T"], points)
        return np.float32(best["black"]), np.float32(best["white"])
    return best_cell_endpoints(view["Z"], view["T"], points, int(cfg["grid"]))


def _move(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=device.type == "cuda")
        else:
            out[key] = value
    return out


def _controls(logits: torch.Tensor, cfg: dict, bounds: dict, knots: list[dict]) -> dict:
    family = cfg["family"]
    if family == "discrete":
        return {"logits": logits}
    if family == "scalar":
        tone = torch.sigmoid(logits[:, 0])
        black, white = interp_manifold(tone, knots)
        return {"t": tone, "black": black, "white": white}
    raw = logits if logits.ndim == 2 else logits.permute(0, 2, 3, 1).contiguous()
    if family == "gain_bias":
        gain, bias = gain_bias_from_raw(raw, bounds)
        return {"gain": gain, "bias": bias}
    black, white = endpoints_from_raw(raw, bounds, float(cfg["min_span"]))
    return {"black": black, "white": white}


def _upsample_control(value: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if value.ndim == 0 or (value.ndim == 1 and value.numel() == 1):
        return value.reshape(())
    if tuple(value.shape[-2:]) == (height, width):
        return value
    return upsample_maps(value[None, None], height, width)[0, 0]


def _continuous_image_loss(controls: dict, batch: dict, cfg: dict) -> torch.Tensor:
    losses = []
    family = cfg["family"]
    lambdas = cfg["lambdas"]
    for index, page in enumerate(batch["pages"]):
        device = next(iter(controls.values())).device
        zed = torch.as_tensor(page["Z"], device=device, dtype=torch.float32)
        target = torch.as_tensor(page["T"], device=device, dtype=torch.float32)
        height, width = zed.shape
        if family == "gain_bias":
            gain = _upsample_control(controls["gain"][index], height, width)
            bias = _upsample_control(controls["bias"][index], height, width)
            image = apply_gain_bias(zed, gain, bias, cfg["clamp_backward"])
            smooth_a = controls["gain"][index][None] if controls["gain"][index].ndim == 2 else None
            smooth_c = controls["bias"][index][None] if controls["bias"][index].ndim == 2 else None
            terms = image_loss(image[None], target[None], lambdas, smooth_a, smooth_c)
        else:
            black = _upsample_control(controls["black"][index], height, width)
            white = _upsample_control(controls["white"][index], height, width)
            image = apply_bw(zed, black, white, cfg["clamp_backward"], float(cfg["min_span"]))
            if family == "scalar" and controls["t"][index].ndim == 2:
                smooth_b, smooth_w = controls["t"][index][None], None
            elif controls["black"][index].ndim == 2:
                smooth_b, smooth_w = controls["black"][index][None], controls["white"][index][None]
            else:
                smooth_b = smooth_w = None
            terms = image_loss(image[None], target[None], lambdas, smooth_b, smooth_w)
        losses.append(terms["total"])
    return torch.stack(losses).mean()


def _discrete_image_loss(logits: torch.Tensor, batch: dict, cfg: dict, presets: list[dict]) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=1)
    losses = []
    for index, page in enumerate(batch["pages"]):
        target = torch.as_tensor(page["T"], device=logits.device, dtype=torch.float32)
        height, width = target.shape
        local = F.interpolate(probs[index:index + 1], size=(height, width), mode="bilinear", align_corners=False)[0]
        local = local / local.sum(0, keepdim=True).clamp_min(1e-12)
        candidates = torch.as_tensor(_tone_stack(page["Z"], presets), device=logits.device, dtype=torch.float32)
        image = clamp_tone((local * candidates).sum(0), cfg["clamp_backward"])
        losses.append(image_loss(image[None], target[None], cfg["lambdas"])["total"])
    return torch.stack(losses).mean()


def _objective(logits, batch, cfg, schedule, bounds, presets, knots):
    controls = _controls(logits, cfg, bounds, knots)
    total = logits.sum() * 0
    parts = {}
    if cfg["family"] == "discrete":
        regret = batch["regret"].to(logits.device) / float(cfg["tau"])
        classified = classification_loss(
            logits,
            batch["q"].to(logits.device),
            regret,
            batch["weight"].to(logits.device),
            {"kl": schedule["lambda_kl"], "regret": schedule["lambda_regret"]},
        )
        total = total + classified["total"]
        parts["kl"] = float(classified["kl"].detach())
        parts["regret"] = float(classified["regret"].detach())
        if float(schedule["image_weight"]) > 0:
            image = _discrete_image_loss(logits, batch, cfg, presets)
            total = total + float(schedule["image_weight"]) * image
            parts["image"] = float(image.detach())
    else:
        if cfg["supervision"] == "oracle_param":
            black = controls["black"]
            white = controls["white"]
            stars_b = batch["black_star"].to(logits.device)
            stars_w = batch["white_star"].to(logits.device)
            param = (black - stars_b).square().mean() + (white - stars_w).square().mean()
            total = total + param
            parts["param"] = float(param.detach())
        if float(schedule["image_weight"]) > 0:
            image = _continuous_image_loss(controls, batch, cfg)
            total = total + float(schedule["image_weight"]) * image
            parts["image"] = float(image.detach())
    if not total.requires_grad:
        raise RuntimeError("objective is disconnected from the controller")
    return total, parts


def _forward(model, batch, device, use_amp: bool) -> torch.Tensor:
    moved = _move(batch, device)
    if use_amp:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            logits = model(moved["x"], moved["mask"], moved["geom"])
    else:
        logits = model(moved["x"], moved["mask"], moved["geom"])
    return logits.float(), moved


@torch.no_grad()
def _render_exact(logits: torch.Tensor, page: dict, cfg: dict, bounds: dict, presets, knots) -> tuple[np.ndarray, dict]:
    """Deployment operator: exact clip, or hard labels plus the guided boundary."""
    family = cfg["family"]
    aux = {}
    if family == "discrete":
        probs = torch.softmax(logits[0].float(), dim=0).cpu().numpy()
        labels = probs.argmax(0).astype(np.int64)
        images = _tone_stack(page["Z"], presets)
        image = compose_from_labels(page["dirty"], images, labels, "hybrid")
        oracle = None
        if "costs" in page:
            oracle = page["costs"].argmin(0)
        elif presets:
            oracle = bank_costs(page["Z"], page["T"], presets, int(cfg["grid"])).argmin(0)
        valid = np.ones(labels.shape, dtype=bool)
        logp = np.log(np.clip(probs, 1e-12, 1))
        entropy = float((-(probs * logp).sum(0)[valid].mean()) / np.log(2))
        agreement = within = regret = None
        if oracle is not None:
            agreement = float((labels[valid] == oracle[valid]).mean())
            within = float((np.abs(labels[valid] - oracle[valid]) <= 1).mean())
            # Regret is reported in cell-MSE units. Tau normalization is training-only.
            costs = page.get("costs")
            if costs is None:
                costs = bank_costs(page["Z"], page["T"], presets, int(cfg["grid"]))
            regret_map = costs - costs.min(0, keepdims=True)
            regret = float((probs * regret_map).sum(0)[valid].mean())
        aux.update({
            "entropy_bits": entropy,
            "class_agreement": agreement,
            "within_one_class": within,
            "expected_regret": regret,
            "pred_labels": labels,
            "oracle_labels": oracle,
        })
        return image, aux
    controls = _controls(logits, cfg, bounds, knots)
    height, width = page["Z"].shape
    if family == "gain_bias":
        gain = _upsample_control(controls["gain"][0], height, width)
        bias = _upsample_control(controls["bias"][0], height, width)
        zed = torch.as_tensor(page["Z"], dtype=torch.float32)
        image = apply_gain_bias(zed, gain, bias, "exact").cpu().numpy()
        gain_np = gain.detach().cpu().numpy()
        bias_np = bias.detach().cpu().numpy()
        aux.update(_continuous_stats(None, None, image, gain_np, bias_np, family))
        return image.astype(np.float32), aux
    black = _upsample_control(controls["black"][0], height, width)
    white = _upsample_control(controls["white"][0], height, width)
    zed = torch.as_tensor(page["Z"], dtype=torch.float32)
    image = apply_bw(zed, black, white, "exact", float(cfg["min_span"])).cpu().numpy()
    aux.update(_continuous_stats(
        black.detach().cpu().numpy(),
        white.detach().cpu().numpy(),
        image,
        None,
        None,
        family,
    ))
    return image.astype(np.float32), aux


def _continuous_stats(black, white, image, gain, bias, family: str) -> dict:
    flat_image = np.asarray(image, dtype=np.float32)
    saturated = float(((flat_image <= 1e-6) | (flat_image >= 1 - 1e-6)).mean())
    out = {"saturation_fraction": saturated, "family": family}
    if black is not None:
        black = np.asarray(black, dtype=np.float32)
        white = np.asarray(white, dtype=np.float32)
        out.update({
            "mean_black": float(np.mean(black)),
            "std_black": float(np.std(black)),
            "mean_white": float(np.mean(white)),
            "std_white": float(np.std(white)),
            "mean_span": float(np.mean(white - black)),
            "spatial_tv": _tv(black) + _tv(white),
            "global_collapse": bool(np.std(black) < 1e-3 and np.std(white) < 1e-3),
        })
    if gain is not None:
        gain = np.asarray(gain, dtype=np.float32)
        bias = np.asarray(bias, dtype=np.float32)
        out.update({
            "mean_gain": float(np.mean(gain)),
            "std_gain": float(np.std(gain)),
            "mean_bias": float(np.mean(bias)),
            "std_bias": float(np.std(bias)),
            "spatial_tv": _tv(gain) + _tv(bias),
            "global_collapse": bool(np.std(gain) < 1e-3 and np.std(bias) < 1e-3),
        })
    return out


def _tv(field: np.ndarray) -> float:
    if field.ndim < 2 or min(field.shape) < 2:
        return 0.0
    return float(np.mean(np.abs(field[:, 1:] - field[:, :-1])) + np.mean(np.abs(field[1:, :] - field[:-1, :])))


def _evaluate(model, dataset: ControlDataset, cfg, bounds, presets, knots, device) -> dict:
    model.eval()
    rows = []
    aux_rows = []
    panels = []
    for index in range(len(dataset)):
        item = dataset[index]
        batch = _collate([item])
        with torch.no_grad():
            logits, _moved = _forward(model, batch, device, use_amp=False)
        page = dict(item["page"])
        if "costs" in item:
            page["costs"] = item["costs"]
        image, aux = _render_exact(logits, page, cfg, bounds, presets, knots)
        metrics = page_metrics(image, page["T"], _masks(page), grid=128)
        rows.append(metrics)
        aux_rows.append(aux)
        if len(panels) < int(cfg["num_panels"]):
            panels.append({"id": item["id"], "page": page, "image": image, "aux": aux})
    summary = mean_metrics(rows)
    for key in ("entropy_bits", "class_agreement", "within_one_class", "expected_regret",
                "mean_black", "std_black", "mean_white", "std_white", "mean_span",
                "mean_gain", "std_gain", "mean_bias", "std_bias", "spatial_tv", "saturation_fraction"):
        vals = [row[key] for row in aux_rows if row.get(key) is not None and np.isfinite(row[key])]
        if vals:
            summary[key] = float(np.mean(vals))
    collapses = [bool(row.get("global_collapse")) for row in aux_rows if "global_collapse" in row]
    if collapses:
        summary["global_collapse_fraction"] = float(np.mean(collapses))
    summary["pages"] = len(rows)
    return {"mean": summary, "rows": rows, "panels": panels}


def _structural_pass(learned: dict, baseline: dict, ratio: float) -> bool:
    for key in ("ink_mae", "edge_mae"):
        base = baseline.get(key)
        got = learned.get(key)
        if base is None or got is None:
            continue
        if not np.isfinite(base) or not np.isfinite(got):
            return False
        if got > float(ratio) * base:
            return False
    return True


def _save_curves(history: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    series = (
        ("train_loss", "train loss"),
        ("cal_psnr", "calibration PSNR"),
        ("cal_ssim", "calibration SSIM"),
        ("cal_regret", "calibration expected regret"),
    )
    for axis, (key, title) in zip(axes.ravel(), series):
        values = [row.get(key) for row in history]
        if any(isinstance(value, (int, float)) and np.isfinite(value) for value in values):
            axis.plot(epochs, [np.nan if value is None else value for value in values])
        axis.set_title(title)
        axis.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_panels(panels: list[dict], cfg: dict, folder: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    for panel in panels:
        page = panel["page"]
        image = panel["image"]
        error = np.abs(image - page["T"])
        if cfg["family"] == "discrete":
            tiles = [
                page["dirty"], page["Z"], panel["aux"].get("pred_labels"),
                panel["aux"].get("oracle_labels"), image, page["T"], error,
            ]
            titles = ["dirty", "Z", "predicted class", "oracle class", "output", "target", "abs error"]
        else:
            tiles = [page["dirty"], page["Z"], image, page["T"], error]
            titles = ["dirty", "Z", "output", "target", "abs error"]
        fig, axes = plt.subplots(1, len(tiles), figsize=(3.2 * len(tiles), 3.4))
        for axis, tile, title in zip(axes, tiles, titles):
            if tile is None:
                axis.set_title(title + " n/a")
                axis.axis("off")
                continue
            axis.imshow(tile, cmap="gray", vmin=0, vmax=1 if tile.dtype != np.int64 and np.max(tile) <= 1.5 else None)
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(panel["id"])
        fig.tight_layout()
        fig.savefig(folder / f"{panel['id'].replace(':', '_')}.png", dpi=110)
        plt.close(fig)


def _print_block(block: dict) -> None:
    print("EXPERIMENT:", block["experiment"])
    print("BEST EPOCH:", block["best_epoch"])
    print("PSNR:", block["psnr"])
    print("SSIM:", block["ssim"])
    print("FIXED BASELINE:", block["fixed"])
    print("ORACLE:", block["oracle"])
    print("HEADROOM RECOVERED:", block["recovery"])
    print("STRUCTURAL GUARD:", block["guard"])
    print("CHECKPOINT:", block["checkpoint"])
    print("DRIVE OUTPUT:", block["output"])
    print("CALIBRATION_PSNR:", block.get("calibration_psnr", "n/a"))
    print("RECOVERY_CAVEAT:", RECOVERY_CAVEAT)
    print("STOP_GO:", block.get("stop_go", "n/a"))
    print("CURRENT6_HYBRID_ORACLE:", block.get("current6", "n/a"))
    print("GLOBAL_COLLAPSE:", block.get("collapse", "n/a"))


def _stop_manifold(cfg: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    write_json(output / "STOP.json", {
        "status": "MANIFOLD_CONFIG_REQUIRED",
        "expected_file": cfg["manifold_path"],
        "required_status": "measured_from_bench03",
        "bench03_present": False,
        "measured": None,
    })
    _print_block({
        "experiment": cfg["experiment_id"],
        "best_epoch": "n/a",
        "psnr": "n/a",
        "ssim": "n/a",
        "fixed": "n/a",
        "oracle": "n/a",
        "recovery": "n/a",
        "guard": "NOT_RUN",
        "checkpoint": "n/a",
        "output": str(output),
        "calibration_psnr": "n/a",
        "stop_go": "MANIFOLD_CONFIG_REQUIRED",
        "current6": "n/a",
        "collapse": "n/a",
    })
    raise SystemExit("MANIFOLD_CONFIG_REQUIRED")


def _prepare_caches(cfg, train_ids, cal_ids, dev_ids, pairs, store, presets, points):
    """Build the shared Z cache one page at a time so the train set is not held in RAM."""
    needed = list(train_ids) + list(cal_ids) + list(dev_ids)
    print(f"Ensuring Z cache for {len(needed)} pages. Test ids are not included.")
    for pid in _tqdm(needed, desc="Z s=31"):
        page = store.ensure(pid, pairs[pid], cfg["test_ids"])
        del page
    if cfg["family"] == "discrete":
        print("Precomputing frozen-bank cell costs. Tau is not stored in this cache.")
        for pid in _tqdm(needed, desc="tone costs"):
            path = _cost_path(cfg, pid, pairs[pid]["dirty_sha"])
            if path.exists():
                continue
            page = store.ensure(pid, pairs[pid], cfg["test_ids"])
            costs = bank_costs(page["Z"], page["T"], presets, int(cfg["grid"]))
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(
                tmp, costs=costs, operator_hash=cfg["operator_hash"], bank_hash=cfg["bank_hash"],
            )
            tmp.replace(path)
            del page
        samples = []
        for pid in cal_ids:
            costs = _load_costs(cfg, pid, pairs[pid]["dirty_sha"])
            regret = costs - costs.min(0, keepdims=True)
            samples.append(regret.reshape(-1))
        tau = calibrate_tau(np.concatenate(samples))
        cfg["tau"] = tau
        cfg["tau_source"] = "median_positive_calibration_cell_regret"
        print(f"Frozen calibration tau = {tau:.8e}. It is not the V2 scale-regret constant.")
    else:
        cfg["tau"] = None
    if cfg["supervision"] == "oracle_param" and not cfg.get("image_with_oracle", False):
        for row in cfg["schedule"]:
            row["image_weight"] = 0.0
    if cfg["supervision"] == "oracle_param":
        level = "page" if cfg["family"] == "global" else "cell"
        cfg["oracle_param_level"] = level
        print(f"Precomputing {level} lattice targets. The lattice is a search grid, not an optimum.")
        ids = list(train_ids) + list(cal_ids) + list(dev_ids)
        for pid in _tqdm(ids, desc="oracle parameters"):
            path = _oracle_path(cfg, pid, pairs[pid]["dirty_sha"])
            if path.exists():
                continue
            page = store.ensure(pid, pairs[pid], cfg["test_ids"])
            black, white = _oracle_arrays(cfg, page, points)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, black=black, white=white, lattice_hash=cfg["lattice_hash"])
            tmp.replace(path)
            del page


def _build_model(cfg, bounds) -> ToneController:
    kind = "global" if cfg["family"] == "global" else "spatial"
    grid = 1 if kind == "global" else int(cfg["grid"])
    model = ToneController(
        in_ch=_input_channels(cfg["input_mode"]),
        out_ch=int(cfg["out_ch"]),
        grid=grid,
        lightweight=bool(cfg["lightweight"]),
        kind=kind,
    )
    return model


def _class_prior(cfg, dataset: ControlDataset) -> torch.Tensor | None:
    if cfg["family"] != "discrete" or cfg["lightweight"]:
        return None
    counts = None
    for index in range(len(dataset)):
        costs = dataset[index]["costs"]
        labels = costs.argmin(0).reshape(-1)
        hist = np.bincount(labels, minlength=costs.shape[0]).astype(np.float64)
        counts = hist if counts is None else counts + hist
    prior = counts / max(counts.sum(), 1)
    return torch.log(torch.tensor(prior, dtype=torch.float32).clamp_min(1e-6))


def _find_encoder(cfg) -> Path | None:
    explicit = cfg.get("encoder_checkpoint")
    if explicit:
        return Path(explicit)
    root = Path(cfg["drive_root"])
    for relative in (
        Path("Testly_Learned_Paper_Scale_v21") / "checkpoints" / "best.pt",
        Path("checkpoints") / "v21_epoch54.pt",
    ):
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def _checkpoint(model, optimizer, scaler, epoch, history, state) -> dict:
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "history": history,
        "best_cal_psnr": state["best_cal_psnr"],
        "epochs_without_improvement": state["wait"],
        "patience_started": state["patience_started"],
        "tau": state["tau"],
        "experiment_id": state["experiment_id"],
    }


def _structural_baseline_ids(ids, load_page, black: float, white: float) -> dict:
    rows = []
    for pid in ids:
        page = load_page(pid)
        rows.append(_metric_row(_render_fixed(page, black, white), page, 128))
        del page
    return mean_metrics(rows)


def run(cfg: dict) -> None:
    if cfg.get("allow_test_split"):
        raise SystemExit("ALLOW_TEST_SPLIT must stay False")
    seed_all(int(cfg["seed"]))
    package = Path(cfg["package_root"])
    cfg_dir = package / "configs"
    bounds = read_json(cfg_dir / cfg["bounds_file"])
    fixed = read_json(cfg_dir / cfg["fixed_file"])
    bank = read_json(cfg_dir / cfg["tone_bank_file"])
    if bank.get("bench03_present") or bounds.get("bench03_present"):
        raise SystemExit("BENCH_03 flag was set in a config that this package treats as absent")
    presets = list(bank["presets"])
    if len(presets) < 2:
        raise SystemExit("TONE_BANK_CONFIG_REQUIRED")
    cfg["min_span"] = float(bounds["min_span"])
    cfg["bounds"] = bounds
    cfg["presets"] = presets
    cfg["fixed_black"] = float(fixed["s05_v03"]["black"])
    cfg["fixed_white"] = float(fixed["s05_v03"]["white"])
    cfg["split_sha"] = fixed["split_sha256"]
    cfg["split_seed"] = int(fixed["split_seed"])
    cfg["encoder_sha"] = fixed["encoder_sha256"]
    cfg["encoder_epoch"] = int(fixed["encoder_epoch"])
    cfg["structural_ratio"] = float(fixed["structural_ink_edge_ratio_max"])
    cfg["operator_hash"] = spec_hash()
    cfg["bank_hash"] = digest(presets)[:16]
    if cfg["loss_preset"] not in LOSS_PRESETS:
        raise SystemExit(f"UNKNOWN_LOSS_PRESET {cfg['loss_preset']}")
    cfg["lambdas"] = dict(LOSS_PRESETS[cfg["loss_preset"]])
    for key, value in cfg["lambdas"].items():
        cfg[f"lambda_{key}"] = value
    if cfg["family"] == "discrete":
        cfg["out_ch"] = len(presets)
    cfg["input_channels"] = _input_channels(cfg["input_mode"])
    points = lattice(bounds, float(cfg["lattice_step"]), float(cfg["min_span"]))
    cfg["lattice_hash"] = digest({"points": points, "step": cfg["lattice_step"]})[:16]
    cfg["oracle_param_level"] = "page" if cfg["family"] == "global" else "cell"

    drive_root = Path(cfg["drive_root"])
    output = _allocate_dir(drive_root, cfg["experiment_id"], bool(cfg["resume"]))
    cfg["output_dir"] = str(output)
    knots = []
    if cfg["family"] == "scalar":
        cfg["manifold_path"] = str(cfg_dir / cfg["manifold_file"])
        measured = load_measured_manifold(cfg["manifold_path"])
        if measured is None:
            _stop_manifold(cfg, output)
        knots = measured["knots"]
        # Illegal knots fail inside interp_manifold before any optimizer step.
        interp_manifold(torch.zeros(1), knots)

    for name in ("checkpoints", "metrics", "predictions", "logs"):
        (output / name).mkdir(parents=True, exist_ok=True)
    cache = drive_root / "Testly_Affine_Tone_DL" / "_shared_cache"
    cfg["cost_dir"] = str(cache / "tone_bank_costs" / f"{cfg['bank_hash']}_g{cfg['grid']}")
    cfg["oracle_dir"] = str(cache / "oracle_params" / f"{cfg['lattice_hash']}_{cfg['oracle_param_level']}_g{cfg['grid']}")
    Path(cfg["cost_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["oracle_dir"]).mkdir(parents=True, exist_ok=True)
    ref_dir = cache / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)

    print("EXPERIMENT:", cfg["experiment_id"])
    print("RESEARCH_QUESTION:", cfg["research_question"])
    print("EXPECTED:", cfg["stop_go"])
    print("MEASURED: none yet. This process has not trained a controller.")
    print("BENCH_03: absent. Tone bank source: bench01_s05_frozen. Bounds source:", bounds["source"])
    print("OpenCV", __import__("cv2").__version__, "operator hash", cfg["operator_hash"])

    split_path = find_split(drive_root)
    split = load_split(split_path, cfg["split_sha"])
    test_ids = set(split["test"])
    cfg["test_ids"] = test_ids
    calibration, development = calibration_development(split["val"], cfg["split_seed"])
    train_ids = list(split["train"])
    assert_allowed(train_ids + calibration + development, test_ids)
    if set(train_ids) & set(calibration) or set(calibration) & set(development):
        raise SystemExit("SPLIT_OVERLAP")
    pairs = index_allowed_pairs(drive_root, cache, train_ids + calibration + development, test_ids)
    store = PageStore(cache / "z_s31")
    _prepare_caches(cfg, train_ids, calibration, development, pairs, store, presets, points)
    write_json(output / "config.json", {k: v for k, v in cfg.items() if k != "test_ids"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vram = gpu_vram_gb()
    kind = "global" if cfg["family"] == "global" else "spatial"
    auto_batch, auto_accum = recommend_batch(kind, int(cfg["grid"]), vram)
    batch_size = int(cfg["batch_size"] or auto_batch)
    accumulation = int(cfg["accumulation"] or auto_accum)
    cfg["batch_size_resolved"] = batch_size
    cfg["accumulation_resolved"] = accumulation
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"Device {gpu_name} VRAM_GB {vram:.2f} batch {batch_size} accumulation {accumulation}")
    write_json(output / "config.json", {k: v for k, v in cfg.items() if k != "test_ids"})

    model = _build_model(cfg, bounds).to(device)
    if cfg["warm_start"]:
        checkpoint = _find_encoder(cfg)
        if checkpoint is None:
            raise SystemExit("WARM_START_CHECKPOINT_NOT_FOUND")
        note = load_encoder_warm_start(
            model, str(checkpoint), cfg["encoder_sha"], cfg["encoder_epoch"],
            strict_encoder=(not cfg["lightweight"] and cfg["input_channels"] == 1),
        )
        print(note)
        (output / "logs" / "warm_start.txt").write_text(note + "\n", encoding="utf-8")
    elif cfg["input_channels"] == 1 and not cfg["lightweight"]:
        print("Scratch initialization. Set WARM_START = True to require the epoch-54 encoder.")

    train_set = ControlDataset(
        train_ids, pairs, store, test_ids, cfg, presets, points, cfg["tau"], bool(cfg["augment"]),
    )
    cal_set = ControlDataset(calibration, pairs, store, test_ids, cfg, presets, points, cfg["tau"], False)
    latest = output / "checkpoints" / "latest.pt"
    best_path = output / "checkpoints" / "best.pt"
    resuming = bool(cfg["resume"] and latest.is_file())
    if (not resuming) and cfg["family"] == "discrete" and not cfg["lightweight"] and hasattr(model, "classifier"):
        prior = _class_prior(cfg, train_set)
        if prior is not None and prior.numel() == model.classifier.bias.numel():
            with torch.no_grad():
                model.classifier.bias.copy_(prior.to(model.classifier.bias.device))
            print("Classifier bias initialized from the train-set oracle class prior.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]),
    )
    use_amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = []
    state = {
        "best_cal_psnr": None,
        "wait": 0,
        "patience_started": False,
        "tau": cfg["tau"],
        "experiment_id": cfg["experiment_id"],
    }
    start_epoch = 1
    if resuming:
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved.get("experiment_id") != cfg["experiment_id"]:
            raise SystemExit("CHECKPOINT_EXPERIMENT_MISMATCH")
        if cfg["tau"] is not None and abs(float(saved.get("tau")) - float(cfg["tau"])) > 1e-12:
            raise SystemExit("TAU_MISMATCH")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        history = saved.get("history", [])
        state["best_cal_psnr"] = saved.get("best_cal_psnr")
        state["wait"] = int(saved.get("epochs_without_improvement", 0))
        state["patience_started"] = bool(saved.get("patience_started", False))
        start_epoch = int(saved["epoch"]) + 1
        print(f"Resumed latest checkpoint at epoch {saved['epoch']}.")

    # Fixed S05 V03 on the calibration pages. Cheap, and it defines the structural guard.
    # Pages were already materialized in the Z cache; reload through the store via the dataset pages.
    def load_page(pid):
        return store.ensure(pid, pairs[pid], test_ids)

    fixed_cal = _structural_baseline_ids(
        calibration, load_page, cfg["fixed_black"], cfg["fixed_white"],
    )
    write_json(output / "metrics" / "fixed_s05_v03_calibration.json", fixed_cal)
    print(
        "Calibration fixed S05 V03 PSNR",
        fmt(fixed_cal.get("psnr")),
        "ink", fmt(fixed_cal.get("ink_mae")),
        "edge", fmt(fixed_cal.get("edge_mae")),
    )

    def save_latest(epoch):
        torch.save(_checkpoint(model, optimizer, scaler, epoch, history, state), latest)

    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        schedule = _schedule_for(epoch, cfg["schedule"])
        model.set_stage(schedule["stage"])
        model.train()
        generator = torch.Generator()
        generator.manual_seed(int(cfg["seed"]) + epoch)
        loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=_collate,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        totals = []
        step = 0
        for batch in _tqdm(loader, desc=f"epoch {epoch} {schedule['stage']}"):
            try:
                logits, moved = _forward(model, batch, device, use_amp)
                loss, _parts = _objective(logits, moved, cfg, schedule, bounds, presets, knots)
                scaler.scale(loss / accumulation).backward()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise SystemExit(
                        "CUDA_OOM. Set BATCH_SIZE = 1 and run again. latest.pt remains resumable."
                    ) from exc
                raise
            totals.append(float(loss.detach()))
            step += 1
            if step % accumulation == 0:
                _optimizer_step(optimizer, scaler, model, float(cfg["grad_clip"]), use_amp)
        if step % accumulation:
            _optimizer_step(optimizer, scaler, model, float(cfg["grad_clip"]), use_amp)

        evaluated = _evaluate(model, cal_set, cfg, bounds, presets, knots, device)
        learned = evaluated["mean"]
        passed = _structural_pass(learned, fixed_cal, cfg["structural_ratio"])
        promoted = False
        image_on = float(schedule["image_weight"]) > 0 or cfg["supervision"] == "oracle_param"
        if cfg["delay_patience"] and image_on and not state["patience_started"]:
            state["patience_started"] = True
            state["wait"] = 0
        if not cfg["delay_patience"]:
            state["patience_started"] = True
        if passed and (state["best_cal_psnr"] is None or learned["psnr"] > state["best_cal_psnr"] + 1e-6):
            state["best_cal_psnr"] = learned["psnr"]
            state["wait"] = 0
            promoted = True
            torch.save(_checkpoint(model, optimizer, scaler, epoch, history, state), best_path)
            write_json(output / "best_metrics.json", {
                "status": "calibration_selected_development_pending",
                "best_epoch": epoch,
                "calibration": learned,
                "fixed_s05_v03_calibration": fixed_cal,
                "structural_guard": "PASS",
                "selection": "max calibration PSNR among epochs whose ink and edge MAE stay within 1.25x S05 V03",
            })
        elif state["patience_started"]:
            state["wait"] += 1
        row = {
            "epoch": epoch,
            "stage": schedule["stage"],
            "train_loss": float(np.mean(totals)) if totals else None,
            "cal_psnr": learned.get("psnr"),
            "cal_ssim": learned.get("ssim"),
            "cal_mae": learned.get("mae"),
            "cal_regret": learned.get("expected_regret"),
            "structural_pass": int(passed),
            "promoted": int(promoted),
            "global_collapse_fraction": learned.get("global_collapse_fraction"),
        }
        history.append(row)
        write_history(output / "training_history.csv", history)
        save_latest(epoch)
        print(
            f"epoch {epoch} stage {schedule['stage']} loss {fmt(row['train_loss'])} "
            f"cal_psnr {fmt(row['cal_psnr'])} guard {'PASS' if passed else 'FAIL'} "
            f"promoted {int(promoted)} wait {state['wait']}"
        )
        if state["patience_started"] and state["wait"] >= int(cfg["patience"]):
            print(f"Early stop at epoch {epoch}. Patience {cfg['patience']} on calibration PSNR.")
            break

    _save_curves(history, output / "metrics" / "curves.png")
    if not best_path.is_file():
        write_json(output / "best_metrics.json", {
            "status": "NO_PROMOTABLE_CHECKPOINT",
            "calibration_history_epochs": len(history),
            "structural_guard": "FAIL",
            "note": "No calibration epoch passed the ink/edge guard. best.pt was not fabricated.",
        })
        _print_block({
            "experiment": cfg["experiment_id"],
            "best_epoch": "n/a",
            "psnr": "n/a",
            "ssim": "n/a",
            "fixed": fmt(fixed_cal.get("psnr")),
            "oracle": "n/a",
            "recovery": "n/a",
            "guard": "NO_PROMOTABLE_CHECKPOINT",
            "checkpoint": str(latest),
            "output": str(output),
            "calibration_psnr": "n/a",
            "stop_go": "UNMEASURED",
            "current6": "n/a",
            "collapse": "n/a",
        })
        return

    saved = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    best_epoch = int(saved["epoch"])
    final_cal = _evaluate(model, cal_set, cfg, bounds, presets, knots, device)
    dev_set = ControlDataset(development, pairs, store, test_ids, cfg, presets, points, cfg["tau"], False)
    final_dev = _evaluate(model, dev_set, cfg, bounds, presets, knots, device)
    _save_panels(final_cal["panels"], cfg, output / "predictions")

    report_ids = {"calibration": calibration, "development": development}
    fixed_rows = {}
    bank_rows = {}
    for split_name, ids in report_ids.items():
        fixed_path = ref_dir / "s05_v03.json"
        bank_name = "bank_page.json" if cfg["oracle_kind"] == "bank_page" else "bank_hybrid_g128.json"
        bank_path = ref_dir / bank_name
        fixed_data = _fill_reference(
            fixed_path, ids, load_page,
            lambda page: _render_fixed(page, cfg["fixed_black"], cfg["fixed_white"]),
        )
        if cfg["oracle_kind"] == "bank_page":
            render = lambda page: _render_bank_page(page, presets)
        else:
            render = lambda page: _render_bank_hybrid(page, presets, 128)
        bank_data = _fill_reference(bank_path, ids, load_page, render)
        fixed_rows[split_name] = _mean_ids(fixed_data, ids)
        bank_rows[split_name] = _mean_ids(bank_data, ids)

    current6_dev = {"available": False, "label": "current6_g128_hybrid_oracle"}
    if cfg["compute_current6"]:
        try:
            current = _fill_reference(
                ref_dir / "current6_g128_hybrid_oracle.json",
                development, load_page, _render_current6,
            )
            current6_dev = _mean_ids(current, development)
            current6_dev["label"] = "current6_g128_hybrid_oracle"
        except Exception as exc:
            current6_dev = {"available": False, "error": str(exc), "label": "current6_g128_hybrid_oracle"}
            print("CURRENT6 reference unavailable:", exc)

    learned_dev = final_dev["mean"]
    fixed_dev = fixed_rows["development"]
    oracle_dev = bank_rows["development"]
    learned_psnr = learned_dev.get("psnr")
    fixed_psnr = fixed_dev.get("psnr") if fixed_dev.get("available") else None
    oracle_psnr = oracle_dev.get("psnr") if oracle_dev.get("available") else None
    recovered = None
    if learned_psnr is not None and fixed_psnr is not None:
        recovered = recovery_ratio(learned_psnr, fixed_psnr, oracle_psnr)
    label = recovery_label(recovered, 0.20, 0.70)
    collapse = learned_dev.get("global_collapse_fraction")
    if cfg["family"] == "global":
        collapse_text = "expected_for_global_architecture"
    elif collapse is not None and collapse >= 0.9:
        collapse_text = f"GLOBAL_COLLAPSE fraction {collapse:.3f} (flag only, not an automatic failure)"
    else:
        collapse_text = fmt(collapse) if collapse is not None else "n/a"
    if fixed_dev.get("available"):
        dev_guard = _structural_pass(learned_dev, fixed_dev, cfg["structural_ratio"])
        guard_text = "PASS" if dev_guard else "FAIL"
    else:
        guard_text = "UNMEASURED"
    report = {
        "status": "measured",
        "experiment_id": cfg["experiment_id"],
        "research_question": cfg["research_question"],
        "expected_stop_go": cfg["stop_go"],
        "measured_stop_go": label,
        "best_epoch": best_epoch,
        "selection": "calibration PSNR among structurally eligible epochs; development is a report",
        "calibration": final_cal["mean"],
        "development": learned_dev,
        "fixed_s05_v03_development": fixed_dev,
        "oracle_development": oracle_dev,
        "oracle_definition": cfg["oracle_kind"],
        "oracle_note": (
            "bank_page is the best frozen S05 preset on the whole page. "
            "bank_hybrid is the target-chosen G128 hybrid of those same five presets. "
            "Neither is a BENCH_03 optimum. A continuous model can legally exceed this reference."
        ),
        "recovery": recovered,
        "recovery_caveat": RECOVERY_CAVEAT,
        "current6_g128_hybrid_oracle_development": current6_dev,
        "structural_guard_development": guard_text,
        "global_collapse": collapse_text,
        "tau": cfg["tau"],
        "bench03_present": False,
    }
    write_json(output / "best_metrics.json", report)
    write_json(output / "metrics" / "development.json", learned_dev)
    _print_block({
        "experiment": cfg["experiment_id"],
        "best_epoch": best_epoch,
        "psnr": fmt(learned_psnr),
        "ssim": fmt(learned_dev.get("ssim")),
        "fixed": fmt(fixed_psnr),
        "oracle": fmt(oracle_psnr),
        "recovery": fmt(recovered),
        "guard": guard_text,
        "checkpoint": str(best_path),
        "output": str(output),
        "calibration_psnr": fmt(final_cal["mean"].get("psnr")),
        "stop_go": f"{label} exploratory. Expected was: {cfg['stop_go']}",
        "current6": fmt(current6_dev.get("psnr")) if current6_dev.get("available") else "unavailable",
        "collapse": collapse_text,
    })
