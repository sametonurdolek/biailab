"""Three-pair benchmark: dirty, fixed s=31, learned s, PSNR oracle, clean target.

Imports the production operator. No NLM, H-Net, binarizer, dewarp, or retraining.
"""

from __future__ import annotations

import csv
import json
import traceback
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from learned_paper_normalization import (
    CANONICAL_LONG_SIDE,
    S_VALUES,
    clean_document,
    estimate_s,
    load_s_estimator,
    normalize_paper,
    prepare_canonical_image,
    read_gray01,
    regression_against_historical,
)
from metrics import psnr, ssim

ROOT = Path(__file__).resolve().parent
BETTER = ROOT / "results" / "bettertest"
REPORT = ROOT / "reports" / "learned_paper_normalization"
PAIR_IDS = ("0000015", "0000025", "0000035")
REGRESSION_LIMIT = 1e-5


def score_pair(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    if pred.shape != target.shape:
        raise RuntimeError(f"FAIL: score shape {pred.shape} vs {target.shape}")
    restored = torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float32))
    clean = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32))
    return float(psnr(restored, clean).item()), float(ssim(restored, clean).item())


def align_clean(clean: np.ndarray, hw: tuple[int, int], source_max_side: int) -> np.ndarray:
    height, width = hw
    interpolation = cv2.INTER_AREA if source_max_side > CANONICAL_LONG_SIDE else cv2.INTER_LINEAR
    resized = cv2.resize(np.asarray(clean, dtype=np.float32), (width, height), interpolation=interpolation)
    return np.ascontiguousarray(resized, dtype=np.float32)


def aspect_ratio(image: np.ndarray) -> float:
    height, width = image.shape[:2]
    return width / height


def load_pairs() -> list[dict]:
    document = json.loads((BETTER / "pairs.json").read_text(encoding="utf-8"))
    pairs = document["pairs"]
    ids = tuple(pair["id"] for pair in pairs)
    if ids != PAIR_IDS:
        raise RuntimeError(f"FAIL: Bettertest pair ids {ids} != {PAIR_IDS}")
    return pairs


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.clip(np.round(np.asarray(image, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), output):
        raise RuntimeError(f"FAIL: could not write {path}")


def render(path: Path, row: dict, panels: list[tuple[str, np.ndarray]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(panels), figsize=(20, 8), constrained_layout=True)
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    figure.suptitle(row["id"] + " | canonical long side 1024 | no NLM")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def mean_of(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def verdict(learned: float, fixed: float) -> str:
    if learned > fixed:
        return "bu frozen üçlüde image-conditioned s selection faydalı"
    return "bu frozen üçlüde learned s fixed31'i geçmedi"


def run(folder: Path) -> dict:
    pairs = load_pairs()
    print("benchmark pairs: results/bettertest/pairs.json", flush=True)
    print("historical operator: document_pipeline.preprocess local branch via manifests/calibration.json", flush=True)
    regressions = []
    for pair in pairs:
        dirty_path = BETTER / pair["prepared"]["dirty"]["path"]
        clean_path = BETTER / pair["prepared"]["clean"]["path"]
        print(f"pair {pair['id']} dirty {dirty_path}", flush=True)
        print(f"pair {pair['id']} clean {clean_path}", flush=True)
        dirty = read_gray01(dirty_path)
        report = regression_against_historical(dirty)
        report["id"] = pair["id"]
        regressions.append(report)
        print(
            f"regression {pair['id']} max {report['max_abs_difference']:.3e} mean {report['mean_abs_difference']:.3e}",
            flush=True,
        )
        if report["max_abs_difference"] > REGRESSION_LIMIT:
            raise RuntimeError(f"FAIL: local_31 regression on {pair['id']}: {report}")

    bundle = load_s_estimator("s-estimator.pt")
    print(f"checkpoint: {bundle.checkpoint_path}", flush=True)
    print(f"device: {bundle.device}", flush=True)
    print(f"epoch: {bundle.epoch}", flush=True)
    print(f"s_values: {list(bundle.s_values)}", flush=True)
    print(f"model_input: {bundle.model_input}", flush=True)
    print(f"strict load: PASS", flush=True)
    print(f"eval mode: {not bundle.model.training}", flush=True)
    if bundle.model.training:
        raise RuntimeError("FAIL: model is training")

    rows = []
    for pair in pairs:
        dirty_path = BETTER / pair["prepared"]["dirty"]["path"]
        clean_path = BETTER / pair["prepared"]["clean"]["path"]
        dirty = read_gray01(dirty_path)
        clean = read_gray01(clean_path)
        relative = abs(aspect_ratio(dirty) - aspect_ratio(clean)) / max(aspect_ratio(dirty), aspect_ratio(clean))
        if relative > 0.03:
            raise RuntimeError(f"FAIL: aspect mismatch {pair['id']} relative {relative}")
        canonical_dirty = prepare_canonical_image(dirty)
        if max(canonical_dirty.shape) != CANONICAL_LONG_SIDE:
            raise RuntimeError(f"FAIL: canonical long side {canonical_dirty.shape}")
        canonical_clean = align_clean(clean, canonical_dirty.shape, max(clean.shape))
        wrapped = clean_document(dirty_path, bundle=bundle)
        if wrapped.selected_s != estimate_s(bundle, canonical_dirty).selected_s:
            raise RuntimeError("FAIL: clean_document drifted from estimate_s")
        if not np.array_equal(wrapped.logits, estimate_s(bundle, canonical_dirty).logits):
            raise RuntimeError("FAIL: repeated logits differ")
        estimate = estimate_s(bundle, canonical_dirty)
        learned = normalize_paper(canonical_dirty, estimate.selected_s)
        if float(np.max(np.abs(learned - wrapped.cleaned_canonical))) != 0.0:
            raise RuntimeError("FAIL: clean_document output differs from normalize_paper")
        fixed = normalize_paper(canonical_dirty, 31)
        oracle_s = None
        oracle_image = None
        oracle_psnr = None
        oracle_ssim = None
        psnr_by_s = []
        for scale in S_VALUES:
            candidate = normalize_paper(canonical_dirty, scale)
            candidate_psnr, candidate_ssim = score_pair(candidate, canonical_clean)
            psnr_by_s.append(candidate_psnr)
            if oracle_psnr is None or candidate_psnr > oracle_psnr:
                oracle_psnr = candidate_psnr
                oracle_ssim = candidate_ssim
                oracle_s = int(scale)
                oracle_image = candidate
        if int(np.argmax(psnr_by_s)) != list(S_VALUES).index(oracle_s):
            raise RuntimeError("FAIL: oracle tie break drifted")
        dirty_psnr, dirty_ssim = score_pair(canonical_dirty, canonical_clean)
        fixed_psnr, fixed_ssim = score_pair(fixed, canonical_clean)
        learned_psnr, learned_ssim = score_pair(learned, canonical_clean)
        probs = [float(value) for value in estimate.probabilities]
        row = {
            "id": pair["id"],
            "selected_s": estimate.selected_s,
            "p_s15": probs[0],
            "p_s21": probs[1],
            "p_s31": probs[2],
            "p_s45": probs[3],
            "p_s63": probs[4],
            "p_s91": probs[5],
            "dirty_psnr": dirty_psnr,
            "fixed31_psnr": fixed_psnr,
            "learned_psnr": learned_psnr,
            "oracle_psnr": oracle_psnr,
            "dirty_ssim": dirty_ssim,
            "fixed31_ssim": fixed_ssim,
            "learned_ssim": learned_ssim,
            "oracle_ssim": oracle_ssim,
            "learned_minus_fixed_psnr": learned_psnr - fixed_psnr,
            "learned_minus_fixed_ssim": learned_ssim - fixed_ssim,
            "oracle_minus_learned_psnr": oracle_psnr - learned_psnr,
            "oracle_s": oracle_s,
            "original_hw": list(dirty.shape),
            "canonical_hw": list(canonical_dirty.shape),
            "aspect_relative_difference": relative,
            "logits": [float(value) for value in estimate.logits],
        }
        rows.append(row)
        print(json.dumps({key: row[key] for key in row if key != "logits"}), flush=True)
        save_png(folder / "outputs" / f"{pair['id']}_learned.png", learned)
        render(
            folder / "comparisons" / f"{pair['id']}_comparison.png",
            row,
            [
                (f"Dirty\n{dirty_psnr:.4f} dB / {dirty_ssim:.6f}", canonical_dirty),
                (f"Fixed s=31\n{fixed_psnr:.4f} dB / {fixed_ssim:.6f}", fixed),
                (f"Learned s={estimate.selected_s}\n{learned_psnr:.4f} dB / {learned_ssim:.6f}", learned),
                (f"Oracle s={oracle_s}\n{oracle_psnr:.4f} dB / {oracle_ssim:.6f}", oracle_image),
                ("Clean target\nreference", canonical_clean),
            ],
        )
        (folder / "probabilities").mkdir(parents=True, exist_ok=True)
        (folder / "probabilities" / f"{pair['id']}.json").write_text(
            json.dumps(
                {"id": pair["id"], "selected_s": estimate.selected_s, "probabilities": probs, "logits": row["logits"]},
                indent=2,
            ),
            encoding="utf-8",
        )

    metric_keys = [
        "dirty_psnr",
        "fixed31_psnr",
        "learned_psnr",
        "oracle_psnr",
        "dirty_ssim",
        "fixed31_ssim",
        "learned_ssim",
        "oracle_ssim",
        "learned_minus_fixed_psnr",
        "learned_minus_fixed_ssim",
        "oracle_minus_learned_psnr",
        "p_s15",
        "p_s21",
        "p_s31",
        "p_s45",
        "p_s63",
        "p_s91",
    ]
    means = {key: mean_of(rows, key) for key in metric_keys}
    headroom = means["oracle_psnr"] - means["fixed31_psnr"]
    captured = None if abs(headroom) < 1e-12 else (means["learned_psnr"] - means["fixed31_psnr"]) / headroom
    histogram = {str(scale): int(Counter(row["selected_s"] for row in rows).get(scale, 0)) for scale in S_VALUES}
    n = len(rows)
    psnr_wins = sum(row["learned_psnr"] > row["fixed31_psnr"] for row in rows)
    ssim_wins = sum(row["learned_ssim"] > row["fixed31_ssim"] for row in rows)
    summary = {
        "n_images": n,
        "mean_psnr_fixed31": means["fixed31_psnr"],
        "mean_psnr_learned": means["learned_psnr"],
        "delta_psnr": means["learned_minus_fixed_psnr"],
        "mean_ssim_fixed31": means["fixed31_ssim"],
        "mean_ssim_learned": means["learned_ssim"],
        "delta_ssim": means["learned_minus_fixed_ssim"],
        "images_learned_psnr_gt_fixed31": f"{psnr_wins}/{n}",
        "images_learned_ssim_gt_fixed31": f"{ssim_wins}/{n}",
        "selected_s_histogram": histogram,
        "mean_oracle_psnr": means["oracle_psnr"],
        "mean_oracle_headroom_psnr": headroom,
        "captured_oracle_headroom_psnr": captured,
        "psnr_verdict": verdict(means["learned_psnr"], means["fixed31_psnr"]),
        "ssim_verdict": verdict(means["learned_ssim"], means["fixed31_ssim"]),
    }
    payload = {
        "status": "completed",
        "checkpoint": {
            "path": str(bundle.checkpoint_path),
            "sha256": bundle.sha256,
            "epoch": bundle.epoch,
            "s_values": list(bundle.s_values),
            "model_input": bundle.model_input,
            "parameter_count": bundle.parameter_count,
            "strict_load": True,
            "best_global_s": bundle.best_global_s,
            "device": str(bundle.device),
            "eval": not bundle.model.training,
        },
        "regression": regressions,
        "coordinate": "s is a pixel scale on the canonical long-side-1024 grid",
        "oracle": "offline argmax of PSNR over S_VALUES; clean is never an estimator input",
        "means": means,
        "summary": summary,
        "per_image": rows,
    }
    (folder / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    fieldnames = [
        "id",
        "selected_s",
        "p_s15",
        "p_s21",
        "p_s31",
        "p_s45",
        "p_s63",
        "p_s91",
        "dirty_psnr",
        "fixed31_psnr",
        "learned_psnr",
        "oracle_psnr",
        "dirty_ssim",
        "fixed31_ssim",
        "learned_ssim",
        "oracle_ssim",
        "learned_minus_fixed_psnr",
        "learned_minus_fixed_ssim",
        "oracle_minus_learned_psnr",
        "oracle_s",
    ]
    with (folder / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        mean_row = {"id": "MEAN", "selected_s": "", "oracle_s": ""}
        mean_row.update(means)
        writer.writerow(mean_row)
    text = [
        "dirty -> canonical long side 1024 -> ScaleNet 512 letterbox -> argmax s",
        "-> Closing(s) -> Gaussian sigma s/6 -> x/max(B,0.1) -> clip[0,1]",
        "",
        f"checkpoint: {bundle.checkpoint_path}",
        f"epoch: {bundle.epoch}",
        f"parameter count: {bundle.parameter_count}",
        f"device: {bundle.device}",
        "",
        "Historical regression local_31_0.0 vs normalize_paper(s=31)",
    ]
    for item in regressions:
        text.append(
            f"  {item['id']}: max {item['max_abs_difference']:.3e}  mean {item['mean_abs_difference']:.3e}"
        )
    text.append("")
    text.append("image | predicted s | probabilities")
    for row in rows:
        probs = " ".join(f"s{scale}:{row[f'p_s{scale}']:.4f}" for scale in S_VALUES)
        text.append(f"  {row['id']} | {row['selected_s']} | {probs}")
    text.extend(
        [
            "",
            f"mean PSNR fixed31: {means['fixed31_psnr']:.6f}",
            f"mean PSNR learned: {means['learned_psnr']:.6f}",
            f"delta PSNR: {means['learned_minus_fixed_psnr']:+.6f}",
            f"mean SSIM fixed31: {means['fixed31_ssim']:.6f}",
            f"mean SSIM learned: {means['learned_ssim']:.6f}",
            f"delta SSIM: {means['learned_minus_fixed_ssim']:+.6f}",
            f"mean PSNR oracle: {means['oracle_psnr']:.6f}",
            f"mean oracle headroom PSNR: {headroom:.6f}",
            f"captured oracle headroom PSNR: {captured}",
            f"learned PSNR > fixed31: {psnr_wins}/{n}",
            f"learned SSIM > fixed31: {ssim_wins}/{n}",
            f"selected-s histogram: {histogram}",
            "",
            f"PSNR: {summary['psnr_verdict']}",
            f"SSIM: {summary['ssim_verdict']}",
            "n=3. Bu üç çiftten belge temizleme görevine genelleme yapılmaz.",
        ]
    )
    (folder / "summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text), flush=True)
    return payload


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    try:
        run(REPORT)
    except Exception:
        (REPORT / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
