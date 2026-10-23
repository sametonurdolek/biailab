"""BENCH_02 runner. Holdout-21 is scored before any development-eval page is opened."""
from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as P  # noqa: E402

_STATE: dict = {}


def _log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    path = P.ROOT / "logs" / "master.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def _safe(pair_id: str) -> str:
    return pair_id.replace("::", "__")


def _init_worker(val_ids: list[str]) -> None:
    cv2.setNumThreads(1)
    ops, can, _ = P.import_bench01()
    _STATE["ops"] = ops
    _STATE["can"] = can
    _STATE["val"] = set(val_ids)


def _metrics(y: np.ndarray, page: dict) -> dict:
    return P.structural(y, page)


def _family_block(dirty, target, page, images: list[np.ndarray], names: list[str]) -> dict:
    stack = np.stack(images, axis=0)
    hard, hybrid, fractions = P.spatial_detail(dirty, stack, target)
    named = {names[i]: fractions[i] for i in range(len(names))}
    ps = np.array([named[n] for n in names], dtype=np.float64)
    order = np.argsort(-ps)
    return {
        "hard": _metrics(hard, page),
        "hybrid": _metrics(hybrid, page),
        "fractions": named,
        "dominant": names[int(order[0])],
        "dominant_fraction": float(ps[order[0]]),
        "entropy_bits": P.entropy_bits(named),
        "n_presets_gt_5pct": int(np.sum(ps > 0.05)),
    }


def _scale_tone(ops, dirty, target, page, bank, sid: str, vid: str, params: dict) -> dict:
    toned = [P.apply_frozen(ops, dirty, bank[i], sid, params) for i in range(len(P.S_VALUES))]
    mse = np.array([np.mean((im - target) ** 2, dtype=np.float64) for im in toned])
    best = int(mse.argmin())
    hard, hybrid, fractions = P.spatial_detail(dirty, np.stack(toned, axis=0), target)
    named = {str(P.S_VALUES[i]): fractions[i] for i in range(len(P.S_VALUES))}
    fixed = _metrics(toned[2], page)
    return {
        "fixed_psnr": fixed["psnr"],
        "fixed_ssim": fixed["ssim"],
        "global_s": int(P.S_VALUES[best]),
        "global_psnr": _metrics(toned[best], page)["psnr"],
        "global_ssim": _metrics(toned[best], page)["ssim"],
        "hard_psnr": _metrics(hard, page)["psnr"],
        "hard_ssim": _metrics(hard, page)["ssim"],
        "hybrid_psnr": _metrics(hybrid, page)["psnr"],
        "hybrid_ssim": _metrics(hybrid, page)["ssim"],
        "fractions": named,
        "dominant_s": max(named, key=named.get),
        "dominant_fraction": max(named.values()),
        "entropy_bits": P.entropy_bits(named),
        "n_scales_gt_5pct": int(sum(v > 0.05 for v in named.values())),
    }


def process_page(task: dict) -> dict:
    cv2.setNumThreads(1)
    pair_id = task["pair_id"]
    path = P.ROOT / "results" / "pages" / f"{_safe(pair_id)}.json"
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("status") == "DONE" and old.get("protocol_hash") == task["protocol_hash"]:
                return {"pair_id": pair_id, "status": "DONE", "skipped": True}
        except json.JSONDecodeError:
            pass
    try:
        ops, can = _STATE["ops"], _STATE["can"]
        page = P.load_page(pair_id, _STATE["val"])
        dirty, target = page["dirty"], page["target"]
        t_bank = time.perf_counter()
        bank = np.stack([can.normalize_paper(dirty, s) for s in P.S_VALUES], axis=0)
        z = bank[2]
        controls = {}
        control_images = {
            "dirty": dirty,
            "fixed31": z,
        }
        mse = np.array([np.mean((bank[i] - target) ** 2, dtype=np.float64) for i in range(6)])
        control_images["global_oracle"] = bank[int(mse.argmin())]
        hard, hybrid = can.spatial_oracle(dirty, bank, target)
        control_images["g128_hard"] = hard
        control_images["g128_hybrid"] = hybrid
        for name, image in control_images.items():
            controls[name] = _metrics(image, page)
        controls["global_oracle"]["selected_s"] = int(P.S_VALUES[int(mse.argmin())])
        controls["bank_ms"] = (time.perf_counter() - t_bank) * 1000.0

        variants = {}
        store = {sid: [] for sid in P.FAMILIES}
        names = {sid: [] for sid in P.FAMILIES}
        for sid, preset_map in P.EXPECTED.items():
            for vid, params in preset_map.items():
                t0 = time.perf_counter()
                y = P.apply_frozen(ops, dirty, z, sid, params)
                key = f"{sid}_{vid}"
                black = params.get("black")
                white = params.get("white", 1.0 if sid == "S04" else None)
                if sid not in {"S04", "S05"}:
                    black, white = None, None
                variants[key] = {
                    "metrics": _metrics(y, page),
                    "tone": P.tone_diag(y, dirty, z if sid in {"S03", "S04", "S05", "S28"} else None, black, white),
                    "directional": P.region_pack(y, dirty, target),
                    "runtime_ms": (time.perf_counter() - t0) * 1000.0,
                    "params": params,
                }
                store[sid].append(y)
                names[sid].append(vid)
        for name, image in (("dirty", dirty), ("fixed31", z), ("g128_hybrid", hybrid)):
            controls[name]["directional"] = P.region_pack(image, dirty, target)

        family = {}
        for sid in P.FAMILIES:
            block = _family_block(dirty, target, page, store[sid], names[sid])
            page_psnr = [variants[f"{sid}_{vid}"]["metrics"]["psnr"] for vid in names[sid]]
            winner = names[sid][int(np.argmax(page_psnr))]
            block["global_variant"] = winner
            block["global"] = variants[f"{sid}_{winner}"]["metrics"]
            family[sid] = block

        scale_tone = {}
        for sid in P.TONE_ON_SCALES:
            for vid, params in P.EXPECTED[sid].items():
                scale_tone[f"{sid}_{vid}"] = _scale_tone(ops, dirty, target, page, bank, sid, vid, params)

        rec = {
            "status": "DONE",
            "protocol_hash": task["protocol_hash"],
            "protocol_version": P.PROTOCOL_VERSION,
            "pair_id": pair_id,
            "role": task["role"],
            "subset": task["subset"],
            "shape": [int(dirty.shape[0]), int(dirty.shape[1])],
            "descriptors": P.dirty_descriptors(dirty),
            "controls": controls,
            "variants": variants,
            "family": family,
            "scale_tone": scale_tone,
        }
        _atomic(path, rec)
        return {"pair_id": pair_id, "status": "DONE", "skipped": False}
    except Exception as exc:
        err = {"pair_id": pair_id, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=12)}
        _atomic(P.ROOT / "logs" / "failures" / f"{_safe(pair_id)}.json", err)
        return err


def _protocol_hash() -> str:
    blob = json.dumps(
        {"version": P.PROTOCOL_VERSION, "expected": P.EXPECTED, "gate": P.GATE, "seed": P.SEED, "boot_holdout": P.N_BOOT_HOLDOUT, "boot_dev": P.N_BOOT_DEV},
        sort_keys=True,
    ).encode()
    return P.sha256_bytes(blob)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _phase0(splits: dict, protocol_hash: str) -> None:
    gate_path = P.ROOT / "gate.json"
    if gate_path.exists():
        loaded = json.loads(gate_path.read_text(encoding="utf-8"))
        if loaded != P.GATE:
            raise SystemExit("GATE_TAMPER")
    else:
        _write_json(gate_path, P.GATE)
    ops, can, cat = P.import_bench01()
    val = set(json.loads((P.SOURCE / "V1" / "manifests" / "split.json").read_text(encoding="utf-8"))["val"])
    probe_id = splits["discovery10"][0]
    page = P.load_page(probe_id, val)
    dirty, target = page["dirty"], page["target"]
    bank = np.stack([can.normalize_paper(dirty, s) for s in P.S_VALUES], axis=0)
    z = bank[2]
    for sid, preset_map in P.EXPECTED.items():
        for vid, params in preset_map.items():
            fast = P.apply_frozen(ops, dirty, z, sid, params)
            exact = ops.apply_operator(dirty, {"system_id": sid, **params})
            delta = float(np.max(np.abs(fast - exact)))
            if delta > 1e-6:
                raise SystemExit(f"OPERATOR_DRIFT {sid} {vid} {delta}")
            a = P.psnr_ssim(fast, target)
            b = can.psnr_ssim(fast, target)
            if abs(a[0] - b[0]) > 1e-6 or abs(a[1] - b[1]) > 1e-6:
                raise SystemExit("PROTOCOL_CHANGE metrics")
    hard, hybrid, _ = P.spatial_detail(dirty, bank, target)
    hard2, hybrid2 = can.spatial_oracle(dirty, bank, target)
    if float(np.max(np.abs(hybrid - hybrid2))) > 1e-5 or float(np.max(np.abs(hard - hard2))) > 1e-5:
        raise SystemExit("PROTOCOL_CHANGE spatial oracle")
    manifest_samples = json.loads((P.BENCH01 / "benchmark_10" / "benchmark_manifest.json").read_text(encoding="utf-8"))["samples"][:10]
    matrix_path = P.BENCH01 / "experiment_matrix.json"
    # BENCH_01's sidecar is the newline-normalized text hash, not the raw Windows bytes.
    matrix_text_sha = P.sha256_bytes(matrix_path.read_text(encoding="utf-8").encode("utf-8"))
    hashes = {
        "operators.py": P.sha256_file(P.BENCH01 / "src" / "operators.py"),
        "canonical.py": P.sha256_file(P.BENCH01 / "src" / "canonical.py"),
        "catalog.py": P.sha256_file(P.BENCH01 / "src" / "catalog.py"),
        "experiment_matrix.json": matrix_text_sha,
        "experiment_matrix.json_raw_bytes": P.sha256_file(matrix_path),
        "experiment_matrix.sha256": (P.BENCH01 / "experiment_matrix.sha256").read_text(encoding="utf-8").strip(),
        "benchmark_manifest.json": P.sha256_file(P.BENCH01 / "benchmark_10" / "benchmark_manifest.json"),
        "implementation_version": cat.IMPLEMENTATION_VERSION,
        "protocol_version": P.PROTOCOL_VERSION,
        "protocol_hash": protocol_hash,
        "bench02_protocol.py": P.sha256_file(P.ROOT / "src" / "protocol.py"),
        "bench02_run_confirm.py": P.sha256_file(P.ROOT / "src" / "run_confirm.py"),
        "bench02_report_bench.py": P.sha256_file(P.ROOT / "src" / "report_bench.py"),
    }
    if hashes["experiment_matrix.json"] != hashes["experiment_matrix.sha256"]:
        raise SystemExit("BENCH01_IMPORT_MISMATCH_STOP")
    _write_json(P.ROOT / "BENCH_01_IMPORT_MANIFEST.json", hashes)
    frozen = {}
    for sid, preset_map in P.EXPECTED.items():
        system = next(s for s in cat.SYSTEMS if s["id"] == sid)
        frozen[sid] = {
            "name": system["name"],
            "math": system["math"],
            "stages": system["stages"],
            "presets": [{"variant": vid, "params": params} for vid, params in preset_map.items()],
        }
    _write_json(P.ROOT / "frozen_candidates.json", frozen)
    _write_json(P.ROOT / "splits.json", {k: splits[k] for k in ("discovery10", "holdout21", "calibration", "development_eval")})
    env = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scipy": None,
        "scikit_image": None,
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cv2_threads": 1,
        "workers": P.WORKERS,
        "bootstrap_seed": P.SEED,
        "bootstrap_holdout": P.N_BOOT_HOLDOUT,
        "bootstrap_dev": P.N_BOOT_DEV,
    }
    _write_json(P.ROOT / "environment.json", env)
    (P.ROOT / "README.md").write_text(
        "# BENCH_02 confirmation\n\n"
        "Frozen import of BENCH_01 S03, S04, S05, S19, and S28. "
        "No preset was changed. The test split is not read. "
        "Holdout-21 is the first confirmation set. Development-eval runs only if the predeclared gate passes.\n",
        encoding="utf-8",
    )
    for sid, folder in (("S03", "S03"), ("S04", "S04_diagnostic"), ("S05", "S05"), ("S19", "S19"), ("S28", "S28_optional_control")):
        text = (
            f"# {sid}\n\n"
            "Imported from BENCH_01. Parameters are in `frozen_candidates.json`. "
            "This folder does not contain a second implementation.\n\n"
            f"```\n{frozen[sid]['math']}\n```\n"
        )
        dest = P.ROOT / "systems" / folder / "README.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
    _log("PHASE0 import and probe passed")
    _log("BENCH_02 PHASE 0 COMPLETE")
    _log("BENCH_01 hashes verified.")
    _log("10 discovery pages identified.")
    _log("21 unseen calibration pages frozen.")
    _log("S03/S05/S19 implementations and parameters match BENCH_01 exactly.")
    _log("Ready for holdout-21 confirmation.")
    return manifest_samples


def _run_ids(pair_ids: list[str], roles: dict[str, str], subsets: dict[str, str], protocol_hash: str, val_ids: list[str]) -> None:
    pending = []
    for pid in pair_ids:
        path = P.ROOT / "results" / "pages" / f"{_safe(pid)}.json"
        if path.exists():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
                if old.get("status") == "DONE" and old.get("protocol_hash") == protocol_hash:
                    continue
            except json.JSONDecodeError:
                pass
        pending.append({"pair_id": pid, "role": roles[pid], "subset": subsets[pid], "protocol_hash": protocol_hash})
    if not pending:
        _log(f"RUN skip {len(pair_ids)} already done")
        return
    _log(f"RUN start {len(pending)} pages")
    failed = 0
    done = 0
    with ProcessPoolExecutor(max_workers=P.WORKERS, initializer=_init_worker, initargs=(val_ids,)) as pool:
        futures = [pool.submit(process_page, task) for task in pending]
        for fut in as_completed(futures):
            rec = fut.result()
            done += 1
            if rec.get("status") != "DONE":
                failed += 1
                _log(f"FAIL {rec.get('pair_id')} {rec.get('error')}")
            elif done % 10 == 0 or done == len(pending):
                _log(f"PROGRESS {done}/{len(pending)} failed {failed}")
    if failed:
        raise SystemExit(f"PAGE_FAILURES {failed}")


def _load_done(pair_ids: list[str], protocol_hash: str) -> list[dict]:
    rows = []
    for pid in pair_ids:
        path = P.ROOT / "results" / "pages" / f"{_safe(pid)}.json"
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") != "DONE" or rec.get("protocol_hash") != protocol_hash:
            raise SystemExit(f"INCOMPLETE_PAGE {pid}")
        rows.append(rec)
    return rows


def _verify_discovery(rows: list[dict], samples: list[dict]) -> None:
    by_id = {r["pair_id"]: r for r in rows}
    s05 = {}
    with (P.BENCH01 / "systems" / "S05_close_division_black_white" / "variants" / "V05" / "results.csv").open(encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            s05[rec["benchmark_id"]] = float(rec["psnr"])
    controls = {r["benchmark_id"]: r for r in json.loads((P.BENCH01 / "controls" / "CURRENT6" / "control_pages.json").read_text(encoding="utf-8"))["rows"]}
    worst = 0.0
    for sample in samples:
        page = by_id[sample["pair_id"]]
        got = page["variants"]["S05_V05"]["metrics"]["psnr"]
        ref = s05[sample["benchmark_id"]]
        worst = max(worst, abs(got - ref))
        hy = page["controls"]["g128_hybrid"]["psnr"]
        href = float(controls[sample["benchmark_id"]]["g128_hybrid_psnr"])
        worst = max(worst, abs(hy - href))
    if worst > 0.005:
        raise SystemExit(f"DISCOVERY_RECOMPUTE_MISMATCH {worst}")
    _log(f"DISCOVERY_RECOMPUTE_MATCH max_abs_db {worst:.6f}")


def main() -> int:
    cv2.setNumThreads(1)
    P.ROOT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    splits = P.load_splits()
    protocol_hash = _protocol_hash()
    samples = _phase0(splits, protocol_hash)
    roles = {}
    subsets = {}
    for pid in splits["calibration"]:
        subsets[pid] = "calibration"
        roles[pid] = "discovery10" if pid in set(splits["discovery10"]) else "holdout21"
    for pid in splits["development_eval"]:
        subsets[pid] = "development_eval"
        roles[pid] = "development123"
    val_ids = splits["calibration"] + splits["development_eval"]
    _run_ids(splits["calibration"], roles, subsets, protocol_hash, val_ids)
    cal_rows = _load_done(splits["calibration"], protocol_hash)
    _verify_discovery(cal_rows, samples)
    from report_bench import evaluate_gate, emit

    gate = evaluate_gate(cal_rows, splits["holdout21"])
    _write_json(P.ROOT / "results" / "holdout_gate.json", gate)
    _log(f"HOLDOUT_GATE {gate['decision']} passers {gate['passers']}")
    dev_rows: list[dict] = []
    if gate["decision"] == "GO":
        _log("PHASE4 development-eval start")
        _run_ids(splits["development_eval"], roles, subsets, protocol_hash, val_ids)
        dev_rows = _load_done(splits["development_eval"], protocol_hash)
        _log("DEV_DONE")
    else:
        _log("PHASE4 skipped: holdout gate did not pass")
    emit(cal_rows, dev_rows, splits, gate, protocol_hash)
    _log("BENCH_02 COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
