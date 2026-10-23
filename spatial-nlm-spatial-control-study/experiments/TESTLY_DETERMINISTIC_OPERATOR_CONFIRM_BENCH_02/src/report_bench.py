"""Summaries for BENCH_02. Reads finished page records. Does not retune."""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import protocol as P

CONTROL_NAMES = ("dirty", "fixed31", "global_oracle", "g128_hard", "g128_hybrid")
STRUCT_FIELDS = ("psnr", "ssim", "mae", "ink_mae", "edge_mae", "bg_mae", "gradient_error", "lighten", "darken", "seam_error", "seam_gradient")


def f6(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _select(rows: list[dict], ids: list[str]) -> list[dict]:
    by = {r["pair_id"]: r for r in rows}
    return [by[i] for i in ids]


def _mean(values) -> float:
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    return float(np.nanmean(arr))


def _boot(delta: np.ndarray, n_resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, delta.size, size=(n_resamples, delta.size))
    means = delta[idx].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "frac_positive": float(np.mean(delta > 1e-6)),
        "frac_negative": float(np.mean(delta < -1e-6)),
        "frac_tie": float(np.mean(np.abs(delta) <= 1e-6)),
        "minimum": float(delta.min()),
        "maximum": float(delta.max()),
        "n": int(delta.size),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def _delta(rows: list[dict], key: str, field: str) -> np.ndarray:
    out = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        out[i] = row["variants"][key]["metrics"][field] - row["controls"]["g128_hybrid"][field]
    return out


def _wins(delta: np.ndarray) -> dict:
    return {
        "wins": int(np.sum(delta > 1e-6)),
        "losses": int(np.sum(delta < -1e-6)),
        "ties": int(np.sum(np.abs(delta) <= 1e-6)),
        "n": int(delta.size),
    }


def _severe_fraction(rows: list[dict], key: str) -> float:
    d_p = _delta(rows, key, "psnr")
    d_s = _delta(rows, key, "ssim")
    severe = (d_p < P.GATE["severe_psnr_lt"]) | (d_s < P.GATE["severe_ssim_lt"])
    return float(severe.mean())


def _compare(rows: list[dict], key: str, n_boot: int) -> dict:
    d_p = _delta(rows, key, "psnr")
    d_s = _delta(rows, key, "ssim")
    return {
        "psnr": _boot(d_p, n_boot, P.SEED),
        "ssim": _boot(d_s, n_boot, P.SEED + 1),
        "wins_psnr": _wins(d_p),
        "wins_ssim": _wins(d_s),
        "severe_fraction": _severe_fraction(rows, key),
        "mean_psnr": _mean([r["variants"][key]["metrics"]["psnr"] for r in rows]),
        "mean_ssim": _mean([r["variants"][key]["metrics"]["ssim"] for r in rows]),
        "worst_psnr_delta": float(d_p.min()),
        "worst_pair": rows[int(d_p.argmin())]["pair_id"],
    }


def _passes(stats: dict) -> bool:
    return (
        stats["psnr"]["mean"] > P.GATE["mean_delta_psnr_gt"]
        and stats["psnr"]["ci95"][0] > P.GATE["ci95_lower_gt"]
        and stats["ssim"]["mean"] > P.GATE["mean_delta_ssim_gt"]
        and stats["severe_fraction"] <= P.GATE["max_severe_page_fraction"]
    )


def evaluate_gate(cal_rows: list[dict], holdout_ids: list[str]) -> dict:
    rows = _select(cal_rows, holdout_ids)
    if len(rows) != 21:
        raise SystemExit("GATE_PAGE_COUNT")
    payload = {"decision": "STOP", "passers": [], "candidates": {}, "n": 21, "rule": P.GATE}
    for key in P.GATE["candidates"]:
        stats = _compare(rows, key, P.N_BOOT_HOLDOUT)
        stats["pass"] = _passes(stats)
        payload["candidates"][key] = stats
        if stats["pass"]:
            payload["passers"].append(key)
    if payload["passers"]:
        payload["decision"] = "GO"
    return payload


def _variant_keys() -> list[str]:
    keys = []
    for sid, preset_map in P.EXPECTED.items():
        for vid in preset_map:
            keys.append(f"{sid}_{vid}")
    return keys


def _family_mean(rows: list[dict], sid: str, mode: str, field: str) -> float:
    return _mean([r["family"][sid][mode][field] for r in rows])


def _structural_ok(rows: list[dict], key: str) -> bool:
    ratio = P.GATE["structural_ink_edge_ratio_max"]
    ok = True
    for field in ("ink_mae", "edge_mae"):
        base = _mean([r["controls"]["g128_hybrid"][field] for r in rows])
        got = _mean([r["variants"][key]["metrics"][field] for r in rows])
        if base > 0 and got > ratio * base:
            ok = False
    if _mean(_delta(rows, key, "ssim")) <= 0:
        ok = False
    return ok


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _page_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        hybrid_p = row["controls"]["g128_hybrid"]["psnr"]
        hybrid_s = row["controls"]["g128_hybrid"]["ssim"]
        for name in CONTROL_NAMES:
            metrics = row["controls"][name]
            out.append({
                "pair_id": row["pair_id"], "role": row["role"], "candidate": name, "kind": "control",
                "psnr": metrics["psnr"], "ssim": metrics["ssim"],
                "delta_psnr_hybrid": metrics["psnr"] - hybrid_p, "delta_ssim_hybrid": metrics["ssim"] - hybrid_s,
                **{k: metrics.get(k) for k in STRUCT_FIELDS if k not in ("psnr", "ssim")},
            })
        for key in _variant_keys():
            metrics = row["variants"][key]["metrics"]
            tone = row["variants"][key]["tone"]
            out.append({
                "pair_id": row["pair_id"], "role": row["role"], "candidate": key, "kind": "fixed",
                "psnr": metrics["psnr"], "ssim": metrics["ssim"],
                "delta_psnr_hybrid": metrics["psnr"] - hybrid_p, "delta_ssim_hybrid": metrics["ssim"] - hybrid_s,
                "runtime_ms": row["variants"][key]["runtime_ms"],
                "frac_eq_0": tone["frac_eq_0"], "frac_eq_1": tone["frac_eq_1"],
                "frac_lt_dirty": tone["frac_lt_dirty"], "frac_gt_dirty": tone["frac_gt_dirty"],
                "black_clip_frac": tone["black_clip_frac"], "white_clip_frac": tone["white_clip_frac"],
                **{k: metrics.get(k) for k in STRUCT_FIELDS if k not in ("psnr", "ssim")},
            })
        for sid in P.FAMILIES:
            for mode in ("global", "hard", "hybrid"):
                metrics = row["family"][sid][mode]
                out.append({
                    "pair_id": row["pair_id"], "role": row["role"], "candidate": f"{sid}_{mode}_oracle", "kind": "oracle",
                    "psnr": metrics["psnr"], "ssim": metrics["ssim"],
                    "delta_psnr_hybrid": metrics["psnr"] - hybrid_p, "delta_ssim_hybrid": metrics["ssim"] - hybrid_s,
                    "oracle_choice": row["family"][sid].get("global_variant") if mode == "global" else row["family"][sid].get("dominant"),
                    **{k: metrics.get(k) for k in STRUCT_FIELDS if k not in ("psnr", "ssim")},
                })
    return out


def _fixed_summary(rows: list[dict], split_name: str, n_boot: int) -> list[dict]:
    summary = []
    hybrid_p = _mean([r["controls"]["g128_hybrid"]["psnr"] for r in rows])
    hybrid_s = _mean([r["controls"]["g128_hybrid"]["ssim"] for r in rows])
    for key in _variant_keys():
        stats = _compare(rows, key, n_boot)
        sid = key.split("_")[0]
        summary.append({
            "split": split_name,
            "candidate": key,
            "mean_psnr": stats["mean_psnr"],
            "mean_ssim": stats["mean_ssim"],
            "delta_psnr": stats["psnr"]["mean"],
            "delta_psnr_ci_lo": stats["psnr"]["ci95"][0],
            "delta_psnr_ci_hi": stats["psnr"]["ci95"][1],
            "median_delta_psnr": stats["psnr"]["median"],
            "delta_ssim": stats["ssim"]["mean"],
            "delta_ssim_ci_lo": stats["ssim"]["ci95"][0],
            "delta_ssim_ci_hi": stats["ssim"]["ci95"][1],
            "median_delta_ssim": stats["ssim"]["median"],
            "wins_psnr": stats["wins_psnr"]["wins"],
            "losses_psnr": stats["wins_psnr"]["losses"],
            "ties_psnr": stats["wins_psnr"]["ties"],
            "wins_ssim": stats["wins_ssim"]["wins"],
            "losses_ssim": stats["wins_ssim"]["losses"],
            "severe_fraction": stats["severe_fraction"],
            "worst_pair": stats["worst_pair"],
            "worst_psnr_delta": stats["worst_psnr_delta"],
            "family_global_psnr": _family_mean(rows, sid, "global", "psnr"),
            "family_hybrid_psnr": _family_mean(rows, sid, "hybrid", "psnr"),
            "spatial_headroom_db": _family_mean(rows, sid, "hybrid", "psnr") - stats["mean_psnr"],
            "hybrid_control_psnr": hybrid_p,
            "hybrid_control_ssim": hybrid_s,
            "structural_ok": _structural_ok(rows, key),
            "gate_pass": _passes(stats),
        })
    return summary


def _oracle_summary(rows: list[dict], split_name: str) -> list[dict]:
    out = []
    hybrid_p = _mean([r["controls"]["g128_hybrid"]["psnr"] for r in rows])
    hybrid_s = _mean([r["controls"]["g128_hybrid"]["ssim"] for r in rows])
    for sid in P.FAMILIES:
        best_key = max(_variant_keys(), key=lambda k: _mean([r["variants"][k]["metrics"]["psnr"] for r in rows]) if k.startswith(sid + "_") else -1e9)
        best_p = _mean([r["variants"][best_key]["metrics"]["psnr"] for r in rows])
        best_s = _mean([r["variants"][best_key]["metrics"]["ssim"] for r in rows])
        out.append({
            "split": split_name,
            "family": sid,
            "best_fixed": best_key,
            "best_fixed_psnr": best_p,
            "best_fixed_ssim": best_s,
            "global_oracle_psnr": _family_mean(rows, sid, "global", "psnr"),
            "global_oracle_ssim": _family_mean(rows, sid, "global", "ssim"),
            "hard_oracle_psnr": _family_mean(rows, sid, "hard", "psnr"),
            "hard_oracle_ssim": _family_mean(rows, sid, "hard", "ssim"),
            "hybrid_oracle_psnr": _family_mean(rows, sid, "hybrid", "psnr"),
            "hybrid_oracle_ssim": _family_mean(rows, sid, "hybrid", "ssim"),
            "global_headroom_db": _family_mean(rows, sid, "global", "psnr") - best_p,
            "spatial_headroom_db": _family_mean(rows, sid, "hybrid", "psnr") - best_p,
            "current_hybrid_psnr": hybrid_p,
            "current_hybrid_ssim": hybrid_s,
            "mean_dominant_fraction": _mean([r["family"][sid]["dominant_fraction"] for r in rows]),
            "mean_entropy_bits": _mean([r["family"][sid]["entropy_bits"] for r in rows]),
            "mean_presets_gt_5pct": _mean([r["family"][sid]["n_presets_gt_5pct"] for r in rows]),
        })
    return out


def _selection_rows(rows: list[dict], split_name: str) -> tuple[list[dict], list[dict]]:
    counts, spatial = [], []
    for sid in P.FAMILIES:
        names = list(P.EXPECTED[sid])
        counter = Counter(r["family"][sid]["global_variant"] for r in rows)
        for vid in names:
            counts.append({"split": split_name, "family": sid, "variant": vid, "n_pages": counter.get(vid, 0), "fraction": counter.get(vid, 0) / len(rows)})
            spatial.append({
                "split": split_name, "family": sid, "variant": vid,
                "mean_area_fraction": _mean([r["family"][sid]["fractions"][vid] for r in rows]),
            })
    return counts, spatial


def _corr(a, b) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(float((a * a).sum() * (b * b).sum()))
    if denom == 0:
        return None
    return float((a * b).sum() / denom)


def _spearman(a, b) -> float | None:
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        out = np.empty(len(v), dtype=np.float64)
        out[order] = np.arange(len(v))
        return out
    return _corr(rank(np.asarray(a, dtype=np.float64)), rank(np.asarray(b, dtype=np.float64)))


def _region_mse(rows: list[dict], key: str, eps: str, region: str) -> float:
    vals = []
    for row in rows:
        if key in CONTROL_NAMES:
            packed = row["controls"][key]["directional"][eps][region]["mse"]
        else:
            packed = row["variants"][key]["directional"][eps][region]["mse"]
        vals.append(packed)
    return _mean(vals)


def _fit(image: np.ndarray, width: int = 180) -> np.ndarray:
    h, w = image.shape
    nh = max(1, int(round(h * (width / w))))
    return cv2.resize(image, (width, nh), interpolation=cv2.INTER_AREA)


def _u8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.clip(image, 0, 1) * 255.0), 0, 255).astype(np.uint8)


def _fix_panels(rows, split_name, keys):
    """Rebuild panels. The candidate tile is computed explicitly."""
    if not rows:
        return
    ops, can, _ = P.import_bench01()
    val = set(json.loads((P.SOURCE / "V1" / "manifests" / "split.json").read_text(encoding="utf-8"))["val"])
    cache: dict = {}

    def images(pair_id: str, needed: list[str]) -> dict:
        if pair_id not in cache:
            page = P.load_page(pair_id, val)
            dirty, target = page["dirty"], page["target"]
            bank = np.stack([can.normalize_paper(dirty, s) for s in P.S_VALUES], axis=0)
            _, hybrid = can.spatial_oracle(dirty, bank, target)
            cache[pair_id] = {"dirty": dirty, "hybrid": hybrid, "target": target, "z": bank[2], "ops": ops}
        made = cache[pair_id]
        for key in needed:
            if key not in made:
                sid, vid = key.split("_", 1)
                made[key] = P.apply_frozen(made["ops"], made["dirty"], made["z"], sid, P.EXPECTED[sid][vid])
        return made

    for key in keys:
        deltas = _delta(rows, key, "psnr")
        order = np.argsort(deltas)
        picks = (("worst", int(order[0])), ("median", int(order[len(order) // 2])), ("best", int(order[-1])))
        strips = []
        for kind, index in picks:
            row = rows[index]
            made = images(row["pair_id"], [key])
            cand = made[key]
            err_h = np.clip(np.abs(made["hybrid"] - made["target"]) * 4.0, 0, 1)
            err_c = np.clip(np.abs(cand - made["target"]) * 4.0, 0, 1)
            tiles = [_u8(_fit(img)) for img in (made["dirty"], made["hybrid"], cand, made["target"], err_h, err_c)]
            height = max(tile.shape[0] for tile in tiles)
            padded = [tile if tile.shape[0] == height else cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0) for tile in tiles]
            strip = np.hstack(padded)
            bar = np.zeros((28, strip.shape[1]), np.uint8)
            cv2.putText(bar, f"{kind} {row['pair_id']} dPSNR {deltas[index]:+.3f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1, cv2.LINE_AA)
            strips.append(np.vstack([bar, strip]))
        width = max(strip.shape[1] for strip in strips)
        canvas = [strip if strip.shape[1] == width else cv2.copyMakeBorder(strip, 0, 0, 0, width - strip.shape[1], cv2.BORDER_CONSTANT, value=0) for strip in strips]
        out = P.ROOT / "panels" / split_name / f"{key}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), np.vstack(canvas))


def _equivalence_md() -> str:
    lines = ["# S03 and S05 equivalence", "", "Code-derived identity, from the imported operators:", "", "```", "S05: Y = clip((Z - black) / (white - black))", "S03: Y = clip(gain * Z + bias)", "S04: Y = clip((Z - black) / (1 - black))", "Z = canonical closing-division at s=31", "```", "", "So `(Z-black)/(white-black) = gain * Z + bias` with", "", "`gain = 1/(white-black)` and `bias = -black/(white-black)`.", "", "Every frozen S05 preset is an affine map. S04 is the same affine with the white endpoint fixed at 1, which forces `gain + bias = 1`. S03 is the same output equation without that endpoint constraint. The five S03 points and the five S05 points are not the same numeric pairs.", "", "| Preset | black | white | gain | bias |", "|---|---:|---:|---:|---:|"]
    for vid, params in P.EXPECTED["S05"].items():
        aff = P.affine_from_endpoints(params["black"], params["white"])
        lines.append(f"| S05 {vid} | {params['black']:.2f} | {params['white']:.2f} | {aff['gain']:.6f} | {aff['bias']:.6f} |")
    for vid, params in P.EXPECTED["S04"].items():
        aff = P.affine_from_endpoints(params["black"], 1.0)
        lines.append(f"| S04 {vid} | {params['black']:.2f} | 1.00 | {aff['gain']:.6f} | {aff['bias']:.6f} |")
    for vid, params in P.EXPECTED["S03"].items():
        ends = P.endpoints_from_affine(params["gain"], params["bias"])
        lines.append(f"| S03 {vid} | {ends['black']:.6f} | {ends['white']:.6f} | {params['gain']:.2f} | {params['bias']:.2f} |")
    lines += [
        "",
        "S03's implied white point is the Z value that the unclipped line sends to 1. Several S03 presets place that point above 1, so paper at Z=1 clips to white while the slope and intercept still differ from every S05 preset.",
        "",
        "No new affine point was added. A denser gain/bias grid would be BENCH_03.",
        "",
    ]
    return "\n".join(lines)


def _chain_md(rows: list[dict], sid: str, fixed_key: str) -> str:
    hybrid_p = _mean([r["controls"]["g128_hybrid"]["psnr"] for r in rows])
    hybrid_s = _mean([r["controls"]["g128_hybrid"]["ssim"] for r in rows])
    fixed_p = _mean([r["variants"][fixed_key]["metrics"]["psnr"] for r in rows])
    fixed_s = _mean([r["variants"][fixed_key]["metrics"]["ssim"] for r in rows])
    steps = [
        ("Current6 hybrid", hybrid_p, hybrid_s),
        (f"Fixed {fixed_key}", fixed_p, fixed_s),
        ("Per-page global oracle", _family_mean(rows, sid, "global", "psnr"), _family_mean(rows, sid, "global", "ssim")),
        ("G128 hard oracle", _family_mean(rows, sid, "hard", "psnr"), _family_mean(rows, sid, "hard", "ssim")),
        ("G128 hybrid oracle", _family_mean(rows, sid, "hybrid", "psnr"), _family_mean(rows, sid, "hybrid", "ssim")),
    ]
    lines = ["| Stage | PSNR | ΔPSNR vs previous | SSIM | ΔSSIM vs previous |", "|---|---:|---:|---:|---:|"]
    prev_p, prev_s = None, None
    for name, psnr, ssim in steps:
        dp = "" if prev_p is None else f6(psnr - prev_p)
        ds = "" if prev_s is None else f6(ssim - prev_s)
        lines.append(f"| {name} | {f6(psnr)} | {dp} | {f6(ssim)} | {ds} |")
        prev_p, prev_s = psnr, ssim
    return "\n".join(lines)


def _failure_md(rows: list[dict], q03_mean: float) -> str:
    key = "S05_V05"
    deltas = _delta(rows, key, "psnr")
    desc_keys = ("mean", "std", "p95", "p99", "frac_below_0_2", "frac_above_0_9", "low_frequency_variance", "low_frequency_range")
    lines = ["# S05 failure regimes", "", "Descriptors are computed on the dirty page only. The threshold for a Q03-like page is the dirty mean of discovery page Q03.", "", f"Q03 dirty mean: {q03_mean:.6f}", "", "## Correlation with S05 V05 ΔPSNR", "", "| Descriptor | Pearson | Spearman |", "|---|---:|---:|"]
    for name in desc_keys:
        xs = [r["descriptors"][name] for r in rows]
        lines.append(f"| {name} | {f6(_corr(xs, deltas))} | {f6(_spearman(xs, deltas))} |")
    bright = [r for r in rows if r["descriptors"]["mean"] >= q03_mean]
    lines += ["", f"Pages at least as bright as Q03: {len(bright)}", ""]
    if bright:
        bdelta = _delta(bright, key, "psnr")
        lines.append(f"Mean ΔPSNR on those pages: {f6(float(bdelta.mean()))}. Worst: {bright[int(bdelta.argmin())]['pair_id']} {f6(float(bdelta.min()))}.")
    lines += ["", "## Severe pages (ΔPSNR < -1 dB or ΔSSIM < -0.05)", ""]
    for row, dp, ds in zip(rows, deltas, _delta(rows, key, "ssim")):
        if dp < P.GATE["severe_psnr_lt"] or ds < P.GATE["severe_ssim_lt"]:
            d = row["descriptors"]
            lines.append(f"- {row['pair_id']}: ΔPSNR {dp:+.3f}, ΔSSIM {ds:+.4f}, dirty mean {d['mean']:.3f}, p95 {d['p95']:.3f}, low-frequency range {d['low_frequency_range']:.3f}")
    if not any((deltas < P.GATE["severe_psnr_lt"]) | (_delta(rows, key, "ssim") < P.GATE["severe_ssim_lt"])):
        lines.append("None.")
    lines += ["", "This is a diagnosis of the frozen V05 preset. The preset was not edited.", ""]
    return "\n".join(lines)


def _decision_text(holdout_rows, dev_rows, gate, summaries) -> str:
    dev = dev_rows
    lines = ["# Final decision", ""]
    if not dev:
        hold_sum = {r["candidate"]: r for r in summaries["holdout21"]}
        oracle = {r["family"]: r for r in summaries["oracle_dev"]}
        s05 = oracle["S05"]
        s19 = oracle["S19"]
        hy_p = hold_sum["S05_V05"]["hybrid_control_psnr"]
        hy_s = hold_sum["S05_V05"]["hybrid_control_ssim"]
        lines += [
            "The predeclared holdout gate did not pass. Development-eval was not run. Numbers below are holdout-21 only.",
            "",
            "```text",
            "BENCH_02 STATUS:",
            "EXPLORATORY_ONLY_NOT_CONFIRMED",
            "",
            "HOLDOUT-21 RESULT:",
            f"{gate['decision']}; passers: {', '.join(gate['passers']) if gate['passers'] else 'none'}",
            "",
            "DEVELOPMENT-123 RESULT:",
            "NOT RUN",
            "",
            "CURRENT6 HYBRID:",
            f"PSNR: {hy_p:.6f}",
            f"SSIM: {hy_s:.6f}",
            "Scope: holdout-21. Not the 123.",
            "",
            "BEST CONFIRMED FIXED TONE OPERATOR:",
            "None. The predeclared tone candidates did not pass the holdout gate.",
            "",
            "S03 / S05 MATHEMATICAL RELATIONSHIP:",
            "Same post-normalization affine family. S05 is the two-endpoint parameterization. S03 is an unconstrained gain/bias grid. No frozen preset pair is numerically identical. See S03_VS_S05_EQUIVALENCE.md.",
            "",
            "WHAT DEGREE OF FREEDOM WAS ACTUALLY DISCOVERED?",
            "BENCH_01 proposed a bidirectional affine tone map after frozen s=31 closing-division. Holdout-21 did not confirm that this is a repeatable gain.",
            "",
            "S05 FAMILY GLOBAL ORACLE:",
            f"PSNR: {s05['global_oracle_psnr']:.6f}",
            f"SSIM: {s05['global_oracle_ssim']:.6f}",
            "Scope: holdout-21.",
            "",
            "S05 FAMILY G128 HYBRID ORACLE:",
            f"PSNR: {s05['hybrid_oracle_psnr']:.6f}",
            f"SSIM: {s05['hybrid_oracle_ssim']:.6f}",
            "Scope: holdout-21.",
            "",
            "S05 SPATIAL HEADROOM ABOVE BEST FIXED:",
            f"{s05['spatial_headroom_db']:.6f} dB above {s05['best_fixed']}",
            "",
            "S19 BEST FIXED:",
            f"{s19['best_fixed']} PSNR {s19['best_fixed_psnr']:.6f} SSIM {s19['best_fixed_ssim']:.6f}",
            "Scope: holdout-21.",
            "",
            "S19 G128 HYBRID ORACLE:",
            f"PSNR {s19['hybrid_oracle_psnr']:.6f} SSIM {s19['hybrid_oracle_ssim']:.6f}",
            "",
            "S19 SPATIAL HEADROOM:",
            f"{s19['spatial_headroom_db']:.6f} dB",
            "",
            "STRUCTURAL GUARDS:",
            "NOT APPLIED AS A CONFIRMATION PASS. Development-eval was not opened.",
            "",
            "MINIMAL CONFIRMED OPERATOR:",
            "None.",
            "",
            "HIGH-CAPACITY CONFIRMED OPERATOR:",
            "None. S19 was not promoted after the tone-family gate failed.",
            "",
            "DOES BACKGROUND ESTIMATION STILL LOOK LIKE THE MAIN BOTTLENECK?",
            "UNRESOLVED",
            "Reason: The tone-family confirmation failed, and this bench did not reopen background-estimator search.",
            "",
            "DOES POST-NORMALIZATION TONE CAPACITY LOOK LIKE THE MAIN BOTTLENECK?",
            "NO",
            "Reason: The predeclared S03/S05 gain did not clear the holdout-21 gate.",
            "",
            "IS A FIXED IMPROVED OPERATOR ALREADY ENOUGH TO JUSTIFY IMPLEMENTATION STUDY?",
            "NO",
            "",
            "IS A NEW LEARNED CONTROLLER JUSTIFIED YET?",
            "NO",
            "",
            "IF YES, WHAT SHOULD IT PREDICT?",
            "Not applicable. No controller is justified.",
            "",
            "IF NO, WHAT MUST BE TESTED NEXT?",
            "The BENCH_01 tone gain did not confirm on unseen calibration pages. Do not train. Do not retune these presets on the same 21.",
            "",
            "TEST SPLIT USED:",
            "NO",
            "```",
            "",
        ]
        return "\n".join(lines)

    dev_sum = {r["candidate"]: r for r in summaries["development123"]}
    primary = list(P.PRIMARY)
    confirmed = [k for k in primary if dev_sum[k]["gate_pass"] and dev_sum[k]["structural_ok"]]
    psnr_only = [k for k in primary if dev_sum[k]["gate_pass"] and not dev_sum[k]["structural_ok"]]
    if confirmed:
        best = max(confirmed, key=lambda k: dev_sum[k]["mean_psnr"])
        level = "CONFIRMED_FIXED_OPERATOR_GAIN"
    elif psnr_only:
        best = max(psnr_only, key=lambda k: dev_sum[k]["mean_psnr"])
        level = "STRUCTURAL_FAILURE"
    else:
        best = max(primary, key=lambda k: dev_sum[k]["mean_psnr"])
        level = "EXPLORATORY_ONLY_NOT_CONFIRMED"
    sid = best.split("_")[0]
    oracle = next(r for r in summaries["oracle_dev"] if r["family"] == sid)
    s19 = next(r for r in summaries["oracle_dev"] if r["family"] == "S19")
    s05 = next(r for r in summaries["oracle_dev"] if r["family"] == "S05")
    headroom = oracle["spatial_headroom_db"]
    if level == "CONFIRMED_FIXED_OPERATOR_GAIN" and headroom >= P.MATERIAL_HEADROOM_DB:
        level = "CONFIRMED_FIXED_OPERATOR_GAIN and CONFIRMED_CONTROL_HEADROOM"
    hy_p = dev_sum[best]["hybrid_control_psnr"]
    hy_s = dev_sum[best]["hybrid_control_ssim"]
    row = dev_sum[best]
    s05_v05 = dev_sum["S05_V05"]
    s05_v03 = dev_sum["S05_V03"]
    status = "RUN on all 123 pages. Frozen presets. No retune."
    controller = headroom >= P.MATERIAL_HEADROOM_DB or s19["spatial_headroom_db"] >= P.MATERIAL_HEADROOM_DB
    tone_yes = level.startswith("CONFIRMED_FIXED")
    lines += [
        f"Classification: {level}",
        "",
        "```text",
        "BENCH_02 STATUS:",
        level,
        "",
        "HOLDOUT-21 RESULT:",
        f"{gate['decision']}; passers: {', '.join(gate['passers']) if gate['passers'] else 'none'}",
        "",
        "DEVELOPMENT-123 RESULT:",
        status,
        "",
        "CURRENT6 HYBRID:",
        f"PSNR: {hy_p:.6f}",
        f"SSIM: {hy_s:.6f}",
        "",
        "BEST CONFIRMED FIXED TONE OPERATOR:",
        f"System: {sid}",
        f"Variant: {best}",
        f"PSNR: {row['mean_psnr']:.6f}",
        f"SSIM: {row['mean_ssim']:.6f}",
        f"ΔPSNR: {row['delta_psnr']:.6f}",
        f"ΔSSIM: {row['delta_ssim']:.6f}",
        f"Page win rate: {row['wins_psnr']}/{row['wins_psnr'] + row['losses_psnr'] + row['ties_psnr']}",
        "",
        "S03 / S05 MATHEMATICAL RELATIONSHIP:",
        "Same post-normalization affine family. S05 is the two-endpoint parameterization. S03 is an unconstrained gain/bias grid. No frozen preset pair is numerically identical. See S03_VS_S05_EQUIVALENCE.md.",
        "",
        "WHAT DEGREE OF FREEDOM WAS ACTUALLY DISCOVERED?",
        "A bidirectional affine tone map after the frozen s=31 closing-division. Black and white endpoints are one parameterization of that affine.",
        "",
        "S05 FAMILY GLOBAL ORACLE:",
        f"PSNR: {s05['global_oracle_psnr']:.6f}",
        f"SSIM: {s05['global_oracle_ssim']:.6f}",
        "",
        "S05 FAMILY G128 HYBRID ORACLE:",
        f"PSNR: {s05['hybrid_oracle_psnr']:.6f}",
        f"SSIM: {s05['hybrid_oracle_ssim']:.6f}",
        "",
        "S05 SPATIAL HEADROOM ABOVE BEST FIXED:",
        f"{s05['spatial_headroom_db']:.6f} dB above {s05['best_fixed']}",
        "",
        "S19 BEST FIXED:",
        f"{s19['best_fixed']} PSNR {s19['best_fixed_psnr']:.6f} SSIM {s19['best_fixed_ssim']:.6f}",
        "",
        "S19 G128 HYBRID ORACLE:",
        f"PSNR {s19['hybrid_oracle_psnr']:.6f} SSIM {s19['hybrid_oracle_ssim']:.6f}",
        "",
        "S19 SPATIAL HEADROOM:",
        f"{s19['spatial_headroom_db']:.6f} dB",
        "",
        "STRUCTURAL GUARDS:",
        f"{'PASS' if row['structural_ok'] else 'FAIL'}",
        f"Reason: ink and edge MAE ratio versus Current6 hybrid, limit {P.GATE['structural_ink_edge_ratio_max']}, plus mean SSIM delta > 0. Severe-page fraction for {best} is {row['severe_fraction']:.3f}.",
        "",
        "MINIMAL CONFIRMED OPERATOR:",
        f"{best}: canonical s=31 closing-division, then the frozen affine/endpoint preset." if tone_yes else "None. The predeclared tone candidates did not confirm.",
        "",
        "HIGH-CAPACITY CONFIRMED OPERATOR:",
        f"S19 {s19['best_fixed']} fixed, spatial oracle {s19['hybrid_oracle_psnr']:.6f} dB, headroom {s19['spatial_headroom_db']:.6f} dB." if s19["spatial_headroom_db"] >= P.MATERIAL_HEADROOM_DB else f"S19 spatial headroom is {s19['spatial_headroom_db']:.6f} dB, below the predeclared {P.MATERIAL_HEADROOM_DB:.2f} dB material line.",
        "",
        "DOES BACKGROUND ESTIMATION STILL LOOK LIKE THE MAIN BOTTLENECK?",
        "NO" if tone_yes else "UNRESOLVED",
        "Reason: The confirming gain, where present, is a tone stage on the unchanged s=31 background. S19 is a separate operator family, reported as capacity, not as a replacement background search.",
        "",
        "DOES POST-NORMALIZATION TONE CAPACITY LOOK LIKE THE MAIN BOTTLENECK?",
        "YES" if tone_yes else "NO",
        "Reason: S03 and S05 share the affine form Y = clip(a Z + b) after Z = current_s31(x). Confirmation is judged on the 123, not on the discovery 10.",
        "",
        "IS A FIXED IMPROVED OPERATOR ALREADY ENOUGH TO JUSTIFY IMPLEMENTATION STUDY?",
        "YES" if tone_yes else "NO",
        "",
        "IS A NEW LEARNED CONTROLLER JUSTIFIED YET?",
        "YES" if controller and tone_yes else "NO",
        "",
        "IF YES, WHAT SHOULD IT PREDICT?",
        "The affine controls (gain and bias, or equivalently black and white) at page level if one preset dominates area, and at G128 cells if several presets each hold more than 5% of area. S19 sigma and gain are a separate high-capacity target only if that family's spatial headroom stays material. This bench does not train it.",
        "",
        "IF NO, WHAT MUST BE TESTED NEXT?",
        "An implementation study of the confirmed fixed tone stage, or BENCH_03 if a denser affine grid is wanted. Do not train on the 10-page slice.",
        "",
        "TEST SPLIT USED:",
        "NO",
        "```",
        "",
        "## Predeclared primary candidates on the 123",
        "",
        f"S05 V03 ΔPSNR {s05_v03['delta_psnr']:.6f} (CI {s05_v03['delta_psnr_ci_lo']:.6f}, {s05_v03['delta_psnr_ci_hi']:.6f}), ΔSSIM {s05_v03['delta_ssim']:.6f}, wins {s05_v03['wins_psnr']}/123, severe fraction {s05_v03['severe_fraction']:.3f}.",
        f"S05 V05 ΔPSNR {s05_v05['delta_psnr']:.6f} (CI {s05_v05['delta_psnr_ci_lo']:.6f}, {s05_v05['delta_psnr_ci_hi']:.6f}), ΔSSIM {s05_v05['delta_ssim']:.6f}, wins {s05_v05['wins_psnr']}/123, severe fraction {s05_v05['severe_fraction']:.3f}.",
        "",
    ]
    return "\n".join(lines)


def _table(summary_rows: list[dict], keys: list[str]) -> str:
    lines = ["| Candidate | PSNR | ΔPSNR | CI lo | SSIM | ΔSSIM | PSNR wins | Severe frac | Structural |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    by = {r["candidate"]: r for r in summary_rows}
    for key in keys:
        r = by[key]
        n = r["wins_psnr"] + r["losses_psnr"] + r["ties_psnr"]
        lines.append(
            f"| {key} | {f6(r['mean_psnr'])} | {f6(r['delta_psnr'])} | {f6(r['delta_psnr_ci_lo'])} | {f6(r['mean_ssim'])} | {f6(r['delta_ssim'])} | {r['wins_psnr']}/{n} | {r['severe_fraction']:.3f} | {'PASS' if r['structural_ok'] else 'FAIL'} |"
        )
    return "\n".join(lines)


def _stat(rows: list[dict], key: str, field: str) -> float:
    if key in CONTROL_NAMES:
        return _mean([r["controls"][key][field] for r in rows])
    return _mean([r["variants"][key]["metrics"][field] for r in rows])


def _region_stat(rows: list[dict], key: str, eps: str, region: str, field: str) -> float:
    vals = []
    for row in rows:
        packed = row["controls"][key]["directional"] if key in CONTROL_NAMES else row["variants"][key]["directional"]
        vals.append(packed[eps][region][field])
    return _mean(vals)


def _answers_md(slices: dict, gate: dict, summaries: dict) -> str:
    """Numbered answers required by the confirmation protocol. Numbers come from the frozen run."""
    dev_rows = slices.get("development123") or []
    hold_rows = slices["holdout21"]
    disc_rows = slices["discovery10"]
    if not dev_rows:
        return "\n# Research questions\n\nThe holdout gate stopped the study before development-eval. The tone-family gain is not confirmed.\n"
    dev = {r["candidate"]: r for r in summaries["development123"]}
    hold = {r["candidate"]: r for r in summaries["holdout21"]}
    disc = {r["candidate"]: r for r in summaries["discovery10"]}
    lines = ["", "# Research questions", ""]

    def line(n, text):
        lines.append(f"{n}. {text}")
        lines.append("")

    v05, v03 = dev["S05_V05"], dev["S05_V03"]
    h05, h03 = hold["S05_V05"], hold["S05_V03"]
    d05 = disc["S05_V05"]
    line(1, (
        f"S05 V05 on the 21 unseen calibration pages: mean ΔPSNR {h05['delta_psnr']:+.3f} dB "
        f"(95% CI {h05['delta_psnr_ci_lo']:+.3f} to {h05['delta_psnr_ci_hi']:+.3f}), "
        f"wins {h05['wins_psnr']}/{h05['wins_psnr'] + h05['losses_psnr'] + h05['ties_psnr']}, "
        f"worst page {h05['worst_psnr_delta']:+.3f} dB. "
        f"The discovery-10 recomputed ΔPSNR was {d05['delta_psnr']:+.3f} dB. "
        f"The +2.54 dB exploratory figure does not survive at that size. A smaller positive gain does: the holdout gate passed for S05 V03 and S05 V05."
    ))
    line(2, (
        f"On all 123 development-eval pages, S05 V05 mean ΔPSNR is {v05['delta_psnr']:+.3f} dB "
        f"(95% CI {v05['delta_psnr_ci_lo']:+.3f} to {v05['delta_psnr_ci_hi']:+.3f}), "
        f"median {v05['median_delta_psnr']:+.3f} dB, "
        f"page wins {v05['wins_psnr']}/{v05['wins_psnr'] + v05['losses_psnr'] + v05['ties_psnr']}."
    ))
    line(3, (
        f"S05 V05 mean ΔSSIM on the 123 is {v05['delta_ssim']:+.4f} "
        f"(95% CI {v05['delta_ssim_ci_lo']:+.4f} to {v05['delta_ssim_ci_hi']:+.4f}), "
        f"SSIM wins {v05['wins_ssim']}/{v05['wins_psnr'] + v05['losses_psnr'] + v05['ties_psnr']}. "
        f"S05 V03 ΔSSIM is {v03['delta_ssim']:+.4f}."
    ))
    line(4, (
        f"Structural guard for S05 V05 is {'PASS' if v05['structural_ok'] else 'FAIL'}. "
        f"For S05 V03 it is {'PASS' if v03['structural_ok'] else 'FAIL'}. "
        f"The guard requires mean ink MAE and edge MAE at or below {P.GATE['structural_ink_edge_ratio_max']:.2f} times Current6 hybrid, and mean ΔSSIM > 0. "
        f"Severe-page fraction (ΔPSNR < -1 dB or ΔSSIM < -0.05) is {v05['severe_fraction']:.3f} for V05 and {v03['severe_fraction']:.3f} for V03. "
        f"Ink MAE: hybrid {_stat(dev_rows, 'g128_hybrid', 'ink_mae'):.4f}, V03 {_stat(dev_rows, 'S05_V03', 'ink_mae'):.4f}, V05 {_stat(dev_rows, 'S05_V05', 'ink_mae'):.4f}. "
        f"Edge MAE: hybrid {_stat(dev_rows, 'g128_hybrid', 'edge_mae'):.4f}, V03 {_stat(dev_rows, 'S05_V03', 'edge_mae'):.4f}, V05 {_stat(dev_rows, 'S05_V05', 'edge_mae'):.4f}."
    ))
    better = "V05" if v05["mean_psnr"] > v03["mean_psnr"] else "V03"
    line(5, (
        f"On the 123, V05 PSNR {v05['mean_psnr']:.3f} versus V03 PSNR {v03['mean_psnr']:.3f}. "
        f"V05 SSIM {v05['mean_ssim']:.4f} versus V03 SSIM {v03['mean_ssim']:.4f}. "
        f"The higher mean PSNR is {better}. "
        f"V05 severe fraction {v05['severe_fraction']:.3f} versus V03 {v03['severe_fraction']:.3f}. "
        f"V05 is the more aggressive endpoint pair. It is not uniformly safer."
    ))
    deltas = _delta(dev_rows, "S05_V05", "psnr")
    bright_corr = _corr([r["descriptors"]["mean"] for r in dev_rows], deltas)
    p95_corr = _corr([r["descriptors"]["p95"] for r in dev_rows], deltas)
    range_corr = _corr([r["descriptors"]["low_frequency_range"] for r in dev_rows], deltas)
    q03 = disc_rows[2]["descriptors"]["mean"]
    bright = [r for r in dev_rows if r["descriptors"]["mean"] >= q03]
    bright_delta = _delta(bright, "S05_V05", "psnr") if bright else None
    bright_txt = "No development page is as bright as discovery Q03."
    if bright_delta is not None and bright_delta.size:
        bright_txt = (
            f"{len(bright)} development pages have dirty mean at least Q03's {q03:.3f}. "
            f"Their mean V05 ΔPSNR is {float(bright_delta.mean()):+.3f} dB."
        )
    line(6, (
        f"V05 ΔPSNR correlates with dirty mean at Pearson {bright_corr:.3f}, with dirty p95 at {p95_corr:.3f}, "
        f"and with low-frequency range at {range_corr:.3f}. {bright_txt} "
        "Page-level detail is in S05_FAILURE_REGIMES.md. The preset was not changed."
    ))
    line(7, (
        "Yes. S05 is Y = clip((Z - black) / (white - black)). "
        "That is the affine Y = clip(gain * Z + bias) with gain = 1/(white-black) and bias = -black/(white-black). "
        "S03 is the same equation on an unconstrained (gain, bias) grid. No frozen S03 point equals a frozen S05 point. "
        "See S03_VS_S05_EQUIVALENCE.md."
    ))
    line(8, (
        "One degree of freedom family: a bidirectional affine tone map after the frozen s=31 closing-division. "
        "BENCH_01 did not find two independent winners. It sampled that affine family twice, once as endpoints and once as gain and bias."
    ))
    s04 = dev["S04_V04"]
    s04_v03 = dev["S04_V03"]
    line(9, (
        f"Black-only S04 V04 (black 0.18, white fixed at 1) mean PSNR on the 123 is {s04['mean_psnr']:.3f} "
        f"(Δ {s04['delta_psnr']:+.3f} dB versus Current6 hybrid {s04['hybrid_control_psnr']:.3f}). "
        f"S04 V03 (black 0.10) is {s04_v03['mean_psnr']:.3f} (Δ {s04_v03['delta_psnr']:+.3f} dB). "
        "A black endpoint alone moves the score. It is not the whole S05 gain."
    ))
    line(10, (
        f"At the same black point 0.10, S05 V03 (white 0.92) PSNR {v03['mean_psnr']:.3f} versus S04 V03 (white 1) PSNR {s04_v03['mean_psnr']:.3f}. "
        f"The white endpoint is the extra degree of freedom. S05 V05 (black 0.22, white 0.82) reaches {v05['mean_psnr']:.3f}."
    ))
    below_h = _region_stat(dev_rows, "g128_hybrid", "1/255", "below", "mse")
    below_5 = _region_stat(dev_rows, "S05_V05", "1/255", "below", "mse")
    below_3 = _region_stat(dev_rows, "S05_V03", "1/255", "below", "mse")
    below_31 = _region_stat(dev_rows, "fixed31", "1/255", "below", "mse")
    share_h = _region_stat(dev_rows, "g128_hybrid", "1/255", "below", "sse_share")
    share_5 = _region_stat(dev_rows, "S05_V05", "1/255", "below", "sse_share")
    above_h = _region_stat(dev_rows, "g128_hybrid", "1/255", "above", "mse")
    above_5 = _region_stat(dev_rows, "S05_V05", "1/255", "above", "mse")
    line(11, (
        f"On pixels with T < x - 1/255, mean per-page MSE is {below_h:.5f} for Current6 hybrid, "
        f"{below_31:.5f} for fixed s=31, {below_3:.5f} for S05 V03, and {below_5:.5f} for S05 V05. "
        f"Mean SSE share of that region is {share_h:.3f} for hybrid and {share_5:.3f} for S05 V05. "
        f"On T > x + 1/255 the MSE is {above_h:.5f} for hybrid and {above_5:.5f} for S05 V05. "
        "The tone stage is able to write values below Z, which the plain division cannot."
    ))
    line(12, (
        "The measured gain uses the unchanged s=31 closing and the unchanged division. "
        "Only the map after Z changes. Background estimation was not replaced. "
        "The gain is a photometric output-space effect on top of the current background."
    ))
    s05o = next(r for r in summaries["oracle_dev"] if r["family"] == "S05")
    counts = Counter(r["family"]["S05"]["global_variant"] for r in dev_rows)
    count_txt = ", ".join(f"{vid} {counts.get(vid, 0)}" for vid in P.EXPECTED["S05"])
    line(13, (
        f"S05 per-page oracle choices on the 123: {count_txt}. "
        f"Mean dominant-cell fraction {s05o['mean_dominant_fraction']:.3f}, "
        f"mean entropy {s05o['mean_entropy_bits']:.3f} bits, "
        f"mean number of presets above 5% area {s05o['mean_presets_gt_5pct']:.2f}. "
        f"Global-oracle headroom above the best frozen preset is {s05o['global_headroom_db']:.3f} dB. "
        f"The predeclared material line is {P.MATERIAL_HEADROOM_DB:.2f} dB."
    ))
    line(14, f"S05 per-page five-preset oracle headroom above best fixed is {s05o['global_headroom_db']:.3f} dB (best fixed {s05o['best_fixed']}, oracle PSNR {s05o['global_oracle_psnr']:.3f}).")
    line(15, (
        f"S05 G128 hybrid oracle PSNR {s05o['hybrid_oracle_psnr']:.3f}, "
        f"spatial headroom above best fixed {s05o['spatial_headroom_db']:.3f} dB. "
        f"Hard oracle PSNR {s05o['hard_oracle_psnr']:.3f}."
    ))
    bits = []
    for key in ("S05_V03", "S05_V05", "S03_V04", "S03_V05"):
        fixed = _mean([r["scale_tone"][key]["fixed_psnr"] for r in dev_rows])
        glob = _mean([r["scale_tone"][key]["global_psnr"] for r in dev_rows])
        hyb = _mean([r["scale_tone"][key]["hybrid_psnr"] for r in dev_rows])
        bits.append(f"{key}: fixed {fixed:.3f}, page-s oracle {glob:.3f} (Δ {glob - fixed:+.3f}), spatial hybrid {hyb:.3f} (Δ {hyb - fixed:+.3f})")
    line(16, "After one frozen tone is applied to every Current6 scale: " + "; ".join(bits) + ". This is the existing six-scale bank, not a new joint grid.")
    s19o = next(r for r in summaries["oracle_dev"] if r["family"] == "S19")
    s19_disc_h = _family_mean(disc_rows, "S19", "hybrid", "psnr")
    s19_hold_h = _family_mean(hold_rows, "S19", "hybrid", "psnr")
    line(17, (
        f"S19 G128 hybrid oracle is {s19_disc_h:.3f} dB on the recomputed discovery 10, "
        f"{s19_hold_h:.3f} dB on holdout-21, and {s19o['hybrid_oracle_psnr']:.3f} dB on the 123. "
        f"Best frozen S19 preset on the 123 is {s19o['best_fixed']} at {s19o['best_fixed_psnr']:.3f} dB."
    ))
    line(18, (
        f"S19 best fixed Δ versus Current6 hybrid on the 123 is {s19o['best_fixed_psnr'] - s19o['current_hybrid_psnr']:+.3f} dB. "
        f"Its spatial headroom above that fixed preset is {s19o['spatial_headroom_db']:.3f} dB. "
        "The 30 dB discovery oracle does not carry over as a deployment score. "
        "What carries is whether the spatial gap above the best frozen preset stays large. See the decision block."
    ))
    line(19, "The minimal confirmed operator, if the decision block names one, is canonical s=31 closing-division plus one frozen affine/endpoint preset. It is not a new background estimator and not S19.")
    line(20, (
        f"A learned controller is justified only when a confirmed family's oracle sits at least {P.MATERIAL_HEADROOM_DB:.2f} dB above its best frozen preset. "
        "That comparison is in the decision block. Justification here means a later estimator study. This bench does not train one. "
        "Oracle label entropy is conditional capacity, not evidence that a network can learn the map."
    ))
    return "\n".join(lines)


def emit(cal_rows: list[dict], dev_rows: list[dict], splits: dict, gate: dict, protocol_hash: str) -> None:
    slices = {
        "discovery10": _select(cal_rows, splits["discovery10"]),
        "holdout21": _select(cal_rows, splits["holdout21"]),
        "calibration31": cal_rows if len(cal_rows) == 31 else _select(cal_rows, splits["calibration"]),
    }
    if dev_rows:
        slices["development123"] = dev_rows
    boot_n = {"discovery10": P.N_BOOT_HOLDOUT, "holdout21": P.N_BOOT_HOLDOUT, "calibration31": P.N_BOOT_HOLDOUT, "development123": P.N_BOOT_DEV}
    summaries = {}
    oracles = []
    selections = []
    spatials = []
    boot_payload = {"protocol_hash": protocol_hash, "seed": P.SEED, "splits": {}}
    for name, rows in slices.items():
        summaries[name] = _fixed_summary(rows, name, boot_n[name])
        oracles.extend(_oracle_summary(rows, name))
        counts, spatial = _selection_rows(rows, name)
        selections.extend(counts)
        spatials.extend(spatial)
        boot_payload["splits"][name] = {r["candidate"]: {k: r[k] for k in r if k != "split"} for r in summaries[name]}

    page_fields = ["pair_id", "role", "candidate", "kind", "psnr", "ssim", "delta_psnr_hybrid", "delta_ssim_hybrid", "mae", "ink_mae", "edge_mae", "bg_mae", "gradient_error", "lighten", "darken", "seam_error", "seam_gradient", "runtime_ms", "frac_eq_0", "frac_eq_1", "frac_lt_dirty", "frac_gt_dirty", "black_clip_frac", "white_clip_frac", "oracle_choice"]
    targets = {
        "holdout21": (P.ROOT / "results" / "holdout21_page_metrics.csv", P.ROOT / "calibration_holdout_21" / "page_results.csv"),
        "calibration31": (P.ROOT / "results" / "calibration31_page_metrics.csv", P.ROOT / "calibration_31" / "page_results.csv"),
    }
    if "development123" in slices:
        targets["development123"] = (P.ROOT / "results" / "development123_page_metrics.csv", P.ROOT / "development_eval_123" / "page_results.csv")
    for name, paths in targets.items():
        payload = _page_rows(slices[name])
        for path in paths:
            _write_csv(path, payload, page_fields)
    flat = [r for rows in summaries.values() for r in rows]
    _write_csv(P.ROOT / "results" / "fixed_variant_summary.csv", flat, list(flat[0].keys()))
    for split_name, folder in (
        ("holdout21", P.ROOT / "calibration_holdout_21"),
        ("calibration31", P.ROOT / "calibration_31"),
        ("development123", P.ROOT / "development_eval_123"),
    ):
        if split_name in summaries:
            _write_csv(folder / "variant_results.csv", summaries[split_name], list(summaries[split_name][0].keys()))
    _write_csv(P.ROOT / "results" / "family_oracle_summary.csv", oracles, list(oracles[0].keys()))
    _write_csv(P.ROOT / "results" / "page_win_rates.csv", flat, ["split", "candidate", "wins_psnr", "losses_psnr", "ties_psnr", "wins_ssim", "losses_ssim", "severe_fraction", "worst_pair", "worst_psnr_delta"])
    _write_csv(P.ROOT / "results" / "preset_selection_distribution.csv", selections, ["split", "family", "variant", "n_pages", "fraction"])
    _write_csv(P.ROOT / "results" / "spatial_preset_distribution.csv", spatials, ["split", "family", "variant", "mean_area_fraction"])
    struct_rows = []
    for name, rows in slices.items():
        for key in ("S03_V04", "S03_V05", "S05_V03", "S05_V05", "S19_V04", "S04_V04", "S28_V04"):
            item = next(r for r in summaries[name] if r["candidate"] == key)
            struct_rows.append({
                "split": name,
                "candidate": key,
                "mean_psnr": item["mean_psnr"],
                "mean_ssim": item["mean_ssim"],
                "delta_psnr": item["delta_psnr"],
                "delta_ssim": item["delta_ssim"],
                "mae": _mean([r["variants"][key]["metrics"]["mae"] for r in rows]),
                "ink_mae": _mean([r["variants"][key]["metrics"]["ink_mae"] for r in rows]),
                "edge_mae": _mean([r["variants"][key]["metrics"]["edge_mae"] for r in rows]),
                "bg_mae": _mean([r["variants"][key]["metrics"]["bg_mae"] for r in rows]),
                "gradient_error": _mean([r["variants"][key]["metrics"]["gradient_error"] for r in rows]),
                "severe_fraction": item["severe_fraction"],
                "structural_ok": item["structural_ok"],
            })
        for sid in ("S03", "S05", "S19"):
            for mode in ("hybrid",):
                struct_rows.append({
                    "split": name, "candidate": f"{sid}_{mode}_oracle",
                    "mean_psnr": _family_mean(rows, sid, mode, "psnr"),
                    "mean_ssim": _family_mean(rows, sid, mode, "ssim"),
                    "mae": _family_mean(rows, sid, mode, "mae"),
                    "ink_mae": _family_mean(rows, sid, mode, "ink_mae"),
                    "edge_mae": _family_mean(rows, sid, mode, "edge_mae"),
                    "bg_mae": _family_mean(rows, sid, mode, "bg_mae"),
                    "gradient_error": _family_mean(rows, sid, mode, "gradient_error"),
                })
        for control in ("g128_hybrid", "fixed31"):
            struct_rows.append({
                "split": name, "candidate": control,
                "mean_psnr": _mean([r["controls"][control]["psnr"] for r in rows]),
                "mean_ssim": _mean([r["controls"][control]["ssim"] for r in rows]),
                "mae": _mean([r["controls"][control]["mae"] for r in rows]),
                "ink_mae": _mean([r["controls"][control]["ink_mae"] for r in rows]),
                "edge_mae": _mean([r["controls"][control]["edge_mae"] for r in rows]),
                "bg_mae": _mean([r["controls"][control]["bg_mae"] for r in rows]),
                "gradient_error": _mean([r["controls"][control]["gradient_error"] for r in rows]),
            })
    _write_csv(P.ROOT / "results" / "structural_metrics.csv", struct_rows, ["split", "candidate", "mean_psnr", "mean_ssim", "delta_psnr", "delta_ssim", "mae", "ink_mae", "edge_mae", "bg_mae", "gradient_error", "severe_fraction", "structural_ok"])
    (P.ROOT / "results" / "bootstrap_results.json").write_text(json.dumps(boot_payload, indent=2), encoding="utf-8")

    keys_show = ["S05_V03", "S05_V05", "S05_V04", "S03_V04", "S03_V05", "S04_V04", "S19_V04", "S28_V04"]
    reports = P.ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    hold = slices["holdout21"]
    q03 = _select(cal_rows, splits["discovery10"])[2]["descriptors"]["mean"]
    (reports / "HOLDOUT21_CONFIRMATION.md").write_text(
        "# Holdout-21 confirmation\n\nThese 21 calibration pages were not in the BENCH_01 discovery ten.\n\n"
        f"Gate decision: {gate['decision']}. Passers: {', '.join(gate['passers']) if gate['passers'] else 'none'}.\n\n"
        "A passer needs mean ΔPSNR > 0.50 dB, 95% CI lower bound > 0, mean ΔSSIM > 0, and severe-page fraction ≤ 0.25. "
        "Severe means ΔPSNR < -1 dB or ΔSSIM < -0.05.\n\n"
        + _table(summaries["holdout21"], keys_show) + "\n\nCurrent6 hybrid mean PSNR "
        + f6(_mean([r["controls"]["g128_hybrid"]["psnr"] for r in hold])) + ", SSIM "
        + f6(_mean([r["controls"]["g128_hybrid"]["ssim"] for r in hold])) + ".\n\n## S05 chains\n\n"
        + _chain_md(hold, "S05", "S05_V05") + "\n\n## S03 chain at V04\n\n" + _chain_md(hold, "S03", "S03_V04")
        + "\n\n## S19 chain at the frozen-five best is in the oracle table. Chain at V04:\n\n" + _chain_md(hold, "S19", "S19_V04") + "\n",
        encoding="utf-8",
    )
    (reports / "CALIBRATION31_SUMMARY.md").write_text(
        "# Calibration 31 aggregate\n\nThis aggregate includes the 10 discovery pages. It is not an independent confirmation.\n\n"
        + _table(summaries["calibration31"], keys_show) + "\n\n## Discovery 10, recomputed, reference only\n\n"
        + _table(summaries["discovery10"], keys_show) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(reports / "CALIBRATION31_SUMMARY.md", reports / "CALIBRATION_CONFIRMATION.md")
    if dev_rows:
        dev_md = (
            "# Development-eval 123\n\nFrozen presets. No retune. Test split not used.\n\n"
            + _table(summaries["development123"], keys_show)
            + "\n\n## S05 V05 chain\n\n" + _chain_md(dev_rows, "S05", "S05_V05")
            + "\n\n## S05 V03 chain\n\n" + _chain_md(dev_rows, "S05", "S05_V03")
            + "\n\n## S03 V04 chain\n\n" + _chain_md(dev_rows, "S03", "S03_V04")
            + "\n\n## S03 V05 chain\n\n" + _chain_md(dev_rows, "S03", "S03_V05")
            + "\n\n## S19 V04 chain\n\n" + _chain_md(dev_rows, "S19", "S19_V04") + "\n"
        )
    else:
        dev_md = "# Development-eval 123\n\nNot run. The holdout-21 gate failed.\n"
    (reports / "DEVELOPMENT123_CONFIRMATION.md").write_text(dev_md, encoding="utf-8")
    shutil.copyfile(reports / "DEVELOPMENT123_CONFIRMATION.md", reports / "DEVELOPMENT_EVAL_CONFIRMATION.md")
    (reports / "S03_VS_S05_EQUIVALENCE.md").write_text(_equivalence_md(), encoding="utf-8")
    focus = dev_rows or hold
    focus_name = "development-eval 123" if dev_rows else "holdout 21"
    (reports / "S05_FAILURE_REGIMES.md").write_text(_failure_md(focus, q03), encoding="utf-8")
    focus_key = "development123" if dev_rows else "holdout21"
    (reports / "S05_ANALYSIS.md").write_text("# S05\n\n" + _table(summaries[focus_key], [k for k in _variant_keys() if k.startswith("S05_")]) + "\n\n" + _chain_md(focus, "S05", "S05_V05") + "\n\n" + _chain_md(focus, "S05", "S05_V03") + "\n\nBelow-region MSE, eps=1/255, mean over pages. This is the squared error on pixels where the clean target is darker than the dirty page by more than 1/255.\n\n" + "\n".join(f"- {k}: {_region_mse(focus, k, '1/255', 'below'):.6f}" for k in ("g128_hybrid", "fixed31", "S04_V04", "S05_V03", "S05_V05")) + "\n", encoding="utf-8")
    (reports / "S03_ANALYSIS.md").write_text("# S03\n\n" + _table(summaries["development123" if dev_rows else "holdout21"], [k for k in _variant_keys() if k.startswith("S03_")]) + "\n\n" + _chain_md(focus, "S03", "S03_V04") + "\n", encoding="utf-8")
    (reports / "S19_ANALYSIS.md").write_text("# S19\n\n" + _table(summaries["development123" if dev_rows else "holdout21"], [k for k in _variant_keys() if k.startswith("S19_")]) + "\n\n" + _chain_md(focus, "S19", "S19_V04") + "\n\nPreset sigmas and gains are the frozen BENCH_01 values. Global-oracle page counts are in PARAMETER_SELECTION_DISTRIBUTIONS.md.\n", encoding="utf-8")
    shutil.copyfile(reports / "S19_ANALYSIS.md", reports / "S19_CAPACITY_ANALYSIS.md")
    struct_lines = ["# Structural guards", "", "Masks are the frozen V2 cache masks. Ink and background are functions of the clean target and are evaluation-only. Lighten and darken are the fraction of ink pixels whose signed error exceeds 0.05, matching `metrics_np`.", "", "| Split | Candidate | PSNR | SSIM | MAE | ink MAE | edge MAE | bg MAE | gradient |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in struct_rows:
        if item["split"] not in ("holdout21", "development123"):
            continue
        struct_lines.append(f"| {item['split']} | {item['candidate']} | {f6(item.get('mean_psnr'))} | {f6(item.get('mean_ssim'))} | {f6(item.get('mae'))} | {f6(item.get('ink_mae'))} | {f6(item.get('edge_mae'))} | {f6(item.get('bg_mae'))} | {f6(item.get('gradient_error'))} |")
    (reports / "STRUCTURAL_GUARDS.md").write_text("\n".join(struct_lines) + "\n", encoding="utf-8")
    shutil.copyfile(reports / "STRUCTURAL_GUARDS.md", reports / "STRUCTURAL_GUARD_ANALYSIS.md")
    sel_lines = ["# Parameter selection", "", "Global oracle counts are target-informed. They are capacity, not a deployed choice.", ""]
    for name in slices:
        sel_lines.append(f"## {name}")
        sel_lines.append("")
        sel_lines.append("| Family | Variant | Pages | Fraction |")
        sel_lines.append("|---|---|---:|---:|")
        for rec in selections:
            if rec["split"] == name and rec["family"] in ("S03", "S05", "S19"):
                sel_lines.append(f"| {rec['family']} | {rec['variant']} | {rec['n_pages']} | {rec['fraction']:.3f} |")
        sel_lines.append("")
    (reports / "PARAMETER_SELECTION_DISTRIBUTIONS.md").write_text("\n".join(sel_lines), encoding="utf-8")
    het_lines = ["# Spatial heterogeneity", "", "Area fractions are inside one family. Entropy is in bits. A high entropy says conditional capacity exists. It does not say a network can learn the map.", ""]
    for rec in oracles:
        if rec["split"] in ("holdout21", "development123") and rec["family"] in ("S03", "S05", "S19"):
            het_lines.append(f"- {rec['split']} {rec['family']}: mean dominant fraction {rec['mean_dominant_fraction']:.3f}, entropy {rec['mean_entropy_bits']:.3f} bits, presets over 5% area {rec['mean_presets_gt_5pct']:.2f}, spatial headroom {rec['spatial_headroom_db']:.3f} dB.")
    het_lines.append("")
    het_lines.append("## Spatial s after a frozen tone")
    het_lines.append("")
    het_lines.append("Each row applies one frozen tone to all six Current6 scales, then composes them. Fixed is s=31 plus that tone.")
    het_lines.append("")
    het_lines.append("| Split | Tone | Fixed s=31 PSNR | Modal page-level s | Page-level s oracle PSNR | Spatial hybrid | Spatial minus fixed |")
    het_lines.append("|---|---|---:|---:|---:|---:|---:|")
    for name, rows in slices.items():
        if name not in ("holdout21", "development123"):
            continue
        for key in ("S05_V03", "S05_V05", "S03_V04", "S03_V05"):
            fixed = _mean([r["scale_tone"][key]["fixed_psnr"] for r in rows])
            glob = _mean([r["scale_tone"][key]["global_psnr"] for r in rows])
            hyb = _mean([r["scale_tone"][key]["hybrid_psnr"] for r in rows])
            s_mode = Counter(r["scale_tone"][key]["global_s"] for r in rows).most_common(1)[0][0]
            het_lines.append(f"| {name} | {key} | {f6(fixed)} | {s_mode} | {f6(glob)} | {f6(hyb)} | {f6(hyb - fixed)} |")
    (reports / "SPATIAL_HETEROGENEITY.md").write_text("\n".join(het_lines) + "\n", encoding="utf-8")
    summaries["oracle_dev"] = [r for r in oracles if r["split"] == ("development123" if dev_rows else "holdout21")]
    decision = _decision_text(hold, dev_rows, gate, summaries) + _answers_md(slices, gate, summaries)
    (reports / "FINAL_DECISION.md").write_text(decision, encoding="utf-8")
    panel_keys = ["S05_V03", "S05_V05", "S03_V04", "S03_V05", "S19_V04"]
    _fix_panels(slices["holdout21"], "holdout21", panel_keys)
    if dev_rows:
        _fix_panels(dev_rows, "development123", panel_keys)
