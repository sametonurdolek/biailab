"""Learned paper normalization: ScaleNet V2.1 chooses s, classical closing does the rest.

The network emits class logits. It does not emit pixels. Deployment is hard argmax
over S_VALUES. The image operator is the frozen local branch of document_pipeline.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from document_pipeline import preprocess

S_VALUES = (15, 21, 31, 45, 63, 91)
CANONICAL_LONG_SIDE = 1024
MODEL_INPUT = 512
EXPECTED_PARAMS = 1_710_918
_GROUP_CANDIDATES = (16, 8, 4, 2, 1)


def make_group_norm(channels: int) -> nn.GroupNorm:
    for groups in _GROUP_CANDIDATES:
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    raise RuntimeError(f"no GroupNorm divisor for {channels} channels")


class ConvGNAct(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = make_group_norm(cout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = make_group_norm(channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = make_group_norm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


def masked_pool(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = F.interpolate(mask, size=feature.shape[-2:], mode="nearest")
    numer = (feature * mask).sum(dim=(2, 3))
    denom = mask.sum(dim=(2, 3)).clamp_min(1)
    return numer / denom


class ScaleNetV21(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvGNAct(1, 32, stride=2)
        self.stage1 = nn.Sequential(ResBlock(32), ResBlock(32))
        self.down2 = ConvGNAct(32, 64, stride=2)
        self.stage2 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.down3 = ConvGNAct(64, 96, stride=2)
        self.stage3 = nn.Sequential(ResBlock(96), ResBlock(96))
        self.down4 = ConvGNAct(96, 160, stride=2)
        self.stage4 = nn.Sequential(ResBlock(160), ResBlock(160))
        self.head = nn.Sequential(
            nn.Linear(352, 160),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(160, 6),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        s1 = self.stage1(x)
        x = self.down2(s1)
        s2 = self.stage2(x)
        x = self.down3(s2)
        s3 = self.stage3(x)
        x = self.down4(s3)
        s4 = self.stage4(x)
        desc = torch.cat(
            [masked_pool(s1, mask), masked_pool(s2, mask), masked_pool(s3, mask), masked_pool(s4, mask)],
            dim=1,
        )
        return self.head(desc)


@dataclass
class ScaleEstimate:
    selected_s: int
    probabilities: np.ndarray
    logits: np.ndarray
    index: int


@dataclass
class ModelBundle:
    model: ScaleNetV21
    device: torch.device
    s_values: tuple[int, ...]
    model_input: int
    epoch: int | None
    best_global_s: int | None
    parameter_count: int
    checkpoint_path: Path
    sha256: str
    metadata: dict


@dataclass
class CleaningResult:
    canonical_dirty: np.ndarray
    cleaned_canonical: np.ndarray
    selected_s: int
    probabilities: np.ndarray
    logits: np.ndarray
    original_hw: tuple[int, int]
    canonical_hw: tuple[int, int]
    restored: np.ndarray | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(checkpoint_path: str | Path) -> Path:
    raw = Path(checkpoint_path)
    if raw.is_file():
        return raw.resolve()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(Path(__file__).resolve().parent / raw)
        candidates.append(Path(__file__).resolve().parents[2] / raw.name)
    seen: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.append(candidate)
        if candidate.is_file():
            return candidate.resolve()
    looked = "\n".join(str(item) for item in seen)
    raise FileNotFoundError(f"FAIL: s-estimator checkpoint not found: {checkpoint_path}\nLooked in:\n{looked}")


def load_s_estimator(checkpoint_path: str | Path = "s-estimator.pt", device: torch.device | str | None = None) -> ModelBundle:
    path = resolve_checkpoint(checkpoint_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise RuntimeError(f"FAIL: checkpoint is {type(state).__name__}, expected a dict")
    missing = [key for key in ("model", "s_values", "model_input") if key not in state]
    if missing:
        raise RuntimeError(f"FAIL: checkpoint missing {missing}; keys={list(state)}")
    s_values = tuple(int(value) for value in state["s_values"])
    model_input = int(state["model_input"])
    if s_values != S_VALUES:
        raise RuntimeError(f"FAIL: checkpoint s_values {list(s_values)} != {list(S_VALUES)}")
    if model_input != MODEL_INPUT:
        raise RuntimeError(f"FAIL: checkpoint model_input {model_input} != {MODEL_INPUT}")
    weights = state["model"]
    if not isinstance(weights, dict):
        raise RuntimeError("FAIL: state['model'] is not a state_dict")
    model = ScaleNetV21()
    model.load_state_dict(weights, strict=True)
    count = sum(parameter.numel() for parameter in model.parameters())
    print(f"s-estimator parameter count: {count}", flush=True)
    if count != EXPECTED_PARAMS:
        raise RuntimeError(f"FAIL: parameter count {count} != {EXPECTED_PARAMS}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model.to(device)
    model.eval()
    if model.training:
        raise RuntimeError("FAIL: model.train() is set after eval()")
    metadata = {key: value for key, value in state.items() if key != "model"}
    return ModelBundle(
        model=model,
        device=device,
        s_values=s_values,
        model_input=model_input,
        epoch=None if state.get("epoch") is None else int(state["epoch"]),
        best_global_s=None if state.get("best_global_s") is None else int(state["best_global_s"]),
        parameter_count=count,
        checkpoint_path=path,
        sha256=file_sha256(path),
        metadata=metadata,
    )


def read_gray01(image_or_path: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(image_or_path, (str, Path)):
        gray = cv2.imread(str(image_or_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"FAIL: cannot read grayscale image: {image_or_path}")
        if gray.dtype != np.uint8:
            raise RuntimeError(f"FAIL: expected uint8 grayscale, got {gray.dtype}")
        return gray.astype(np.float32) / 255.0
    array = np.asarray(image_or_path)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"FAIL: expected one grayscale plane, got shape {array.shape}")
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    values = array.astype(np.float32, copy=False)
    if values.size and float(np.nanmax(values)) > 1.5:
        values = values / 255.0
    return np.ascontiguousarray(values, dtype=np.float32)


def resize_long_side(image: np.ndarray, long_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = long_side / max(height, width)
    if height >= width:
        new_h = long_side
        new_w = max(1, int(round(width * scale)))
    else:
        new_w = long_side
        new_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    if max(resized.shape[:2]) != long_side:
        raise RuntimeError(f"FAIL: long side {max(resized.shape[:2])} != {long_side}")
    return np.ascontiguousarray(resized, dtype=np.float32)


def prepare_canonical_image(image_or_path: np.ndarray | str | Path) -> np.ndarray:
    return resize_long_side(read_gray01(image_or_path), CANONICAL_LONG_SIDE)


def prepare_estimator_input(canonical_image: np.ndarray, model_input: int = MODEL_INPUT) -> tuple[np.ndarray, np.ndarray]:
    page = resize_long_side(np.asarray(canonical_image, dtype=np.float32), model_input)
    height, width = page.shape[:2]
    canvas = np.ones((model_input, model_input), dtype=np.float32)
    mask = np.zeros((model_input, model_input), dtype=np.float32)
    top = (model_input - height) // 2
    left = (model_input - width) // 2
    canvas[top : top + height, left : left + width] = page
    mask[top : top + height, left : left + width] = 1.0
    return canvas, mask


def estimate_s(model_bundle: ModelBundle, canonical_image: np.ndarray) -> ScaleEstimate:
    image = np.asarray(canonical_image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"FAIL: canonical image must be HxW, got {image.shape}")
    if max(image.shape) != CANONICAL_LONG_SIDE:
        raise RuntimeError(
            f"FAIL: s is defined on long side {CANONICAL_LONG_SIDE}; got shape {image.shape}"
        )
    canvas, mask = prepare_estimator_input(image, model_bundle.model_input)
    if canvas.shape != (model_bundle.model_input, model_bundle.model_input):
        raise RuntimeError(f"FAIL: estimator canvas {canvas.shape}")
    if mask.shape != canvas.shape:
        raise RuntimeError(f"FAIL: estimator mask {mask.shape}")
    x = torch.from_numpy(canvas)[None, None].to(model_bundle.device)
    m = torch.from_numpy(mask)[None, None].to(model_bundle.device)
    with torch.inference_mode():
        logits = model_bundle.model(x, m)
        probabilities = torch.softmax(logits, dim=1)
    if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
        raise RuntimeError("FAIL: non-finite estimator output")
    total = float(probabilities.sum())
    if abs(total - 1.0) > 1e-4:
        raise RuntimeError(f"FAIL: probability sum {total}")
    index = int(torch.argmax(probabilities, dim=1).item())
    selected = int(model_bundle.s_values[index])
    if selected not in S_VALUES:
        raise RuntimeError(f"FAIL: selected s {selected} is outside {list(S_VALUES)}")
    return ScaleEstimate(
        selected_s=selected,
        probabilities=probabilities[0].detach().cpu().numpy().astype(np.float64),
        logits=logits[0].detach().cpu().numpy().astype(np.float64),
        index=index,
    )


def frozen_local_config(size: int) -> dict:
    return {"name": f"local_{int(size)}_0.0", "mode": "local", "size": int(size), "white": 0.98, "black": 0.0}


def historical_local_31_config() -> dict:
    path = Path(__file__).resolve().parent / "manifests" / "calibration.json"
    config = json.loads(path.read_text(encoding="utf-8"))["pipeline"]
    if config.get("name") != "local_31_0.0" or config.get("mode") != "local":
        raise RuntimeError(f"FAIL: frozen pipeline is not local_31_0.0: {config}")
    if int(config.get("size")) != 31 or float(config.get("black")) != 0.0:
        raise RuntimeError(f"FAIL: frozen local_31_0.0 fields changed: {config}")
    return config


def normalize_paper(canonical_image: np.ndarray, s: int) -> np.ndarray:
    size = int(s)
    if size < 1:
        raise ValueError(f"FAIL: kernel size {s}")
    image = np.asarray(canonical_image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"FAIL: normalize_paper expects HxW, got {image.shape}")
    return preprocess(image, frozen_local_config(size))


def historical_local_31(image: np.ndarray) -> np.ndarray:
    return preprocess(np.asarray(image, dtype=np.float32), historical_local_31_config())


def regression_against_historical(image: np.ndarray) -> dict:
    historical = historical_local_31(image)
    current = normalize_paper(image, 31)
    if historical.shape != current.shape:
        raise RuntimeError(f"FAIL: regression shape {historical.shape} vs {current.shape}")
    delta = np.abs(historical.astype(np.float64) - current.astype(np.float64))
    return {
        "max_abs_difference": float(delta.max()) if delta.size else 0.0,
        "mean_abs_difference": float(delta.mean()) if delta.size else 0.0,
        "shape": list(historical.shape),
    }


def resize_to_hw(image: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    height, width = hw
    source_long = max(image.shape[:2])
    target_long = max(height, width)
    interpolation = cv2.INTER_AREA if target_long < source_long else cv2.INTER_LINEAR
    resized = cv2.resize(np.asarray(image, dtype=np.float32), (width, height), interpolation=interpolation)
    return np.ascontiguousarray(resized, dtype=np.float32)


def clean_document(
    image_or_path: np.ndarray | str | Path,
    checkpoint_path: str | Path = "s-estimator.pt",
    restore_original_size: bool = False,
    bundle: ModelBundle | None = None,
) -> CleaningResult:
    gray = read_gray01(image_or_path)
    canonical = prepare_canonical_image(gray)
    if bundle is None:
        bundle = load_s_estimator(checkpoint_path)
    elif bundle.model.training:
        raise RuntimeError("FAIL: estimator bundle is not in eval mode")
    estimate = estimate_s(bundle, canonical)
    cleaned = normalize_paper(canonical, estimate.selected_s)
    restored = resize_to_hw(cleaned, gray.shape) if restore_original_size else None
    return CleaningResult(
        canonical_dirty=canonical,
        cleaned_canonical=cleaned,
        selected_s=estimate.selected_s,
        probabilities=estimate.probabilities,
        logits=estimate.logits,
        original_hw=gray.shape,
        canonical_hw=canonical.shape,
        restored=restored,
    )


def probability_lines(probabilities: np.ndarray) -> str:
    return "\n".join(f"    s{scale} {float(prob):.8f}" for scale, prob in zip(S_VALUES, probabilities))


def save_gray_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.clip(np.round(np.asarray(image, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), output):
        raise RuntimeError(f"FAIL: could not write {path}")


def _print_result(result: CleaningResult, checkpoint: Path, device: torch.device, output: Path) -> None:
    print(f"model checkpoint: {checkpoint}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"input resolution: {result.original_hw[0]}x{result.original_hw[1]}", flush=True)
    print(f"canonical resolution: {result.canonical_hw[0]}x{result.canonical_hw[1]}", flush=True)
    print(f"selected s: {result.selected_s}", flush=True)
    print("probabilities:", flush=True)
    print(probability_lines(result.probabilities), flush=True)
    print(f"output path: {output}", flush=True)


def _clean_one(path: Path, bundle: ModelBundle, output: Path, save_canonical: bool, restore_original_size: bool) -> None:
    result = clean_document(path, bundle=bundle, restore_original_size=restore_original_size)
    image = result.restored if restore_original_size else result.cleaned_canonical
    save_gray_png(output, image)
    if save_canonical:
        canonical_path = output.with_name(f"{output.stem}_canonical{output.suffix}")
        save_gray_png(canonical_path, result.cleaned_canonical)
    _print_result(result, bundle.checkpoint_path, bundle.device, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="ScaleNet V2.1 hard-argmax paper normalization")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", default="s-estimator.pt")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--save-canonical", action="store_true")
    parser.add_argument("--restore-original-size", action="store_true")
    args = parser.parse_args()
    bundle = load_s_estimator(args.model)
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        paths = sorted(item for item in args.input.iterdir() if item.suffix.lower() in suffixes)
        if not paths:
            raise RuntimeError(f"FAIL: no images in {args.input}")
        for path in paths:
            _clean_one(path, bundle, args.output / f"{path.stem}_learned.png", args.save_canonical, args.restore_original_size)
        return
    if not args.input.is_file():
        raise FileNotFoundError(f"FAIL: input not found: {args.input}")
    _clean_one(args.input, bundle, args.output, args.save_canonical, args.restore_original_size)


if __name__ == "__main__":
    main()
