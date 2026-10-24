"""Frozen split, canonical pages, and the shared Z cache.

Test ids are rejected before any image byte is read. Zip members whose stem
is not in the allowed train/calibration/development set are not extracted.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .metrics import structure_masks
from .operator import normalize_s31

CANONICAL = 1024
MODEL_INPUT = 512
PREP_VERSION = "single_resize_dirty_geometry_v2_01"
OPERATOR_SPEC = {
    "s": 31,
    "sigma": "s/6",
    "floor": 0.1,
    "border": "REFLECT_101",
    "prep": PREP_VERSION,
    "canonical": CANONICAL,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def spec_hash() -> str:
    blob = json.dumps(OPERATOR_SPEC, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def page_cache_name(pid: str, dirty_sha: str, clean_sha: str, operator_hash: str) -> str:
    """Identity of one cached page: pair, dirty bytes, clean bytes, operator spec."""
    return hashlib.sha256(f"{pid}|{dirty_sha}|{clean_sha}|{operator_hash}".encode()).hexdigest()


def load_split(path: Path, expected_sha: str) -> dict:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha:
        raise SystemExit(f"SPLIT_HASH_MISMATCH {digest}")
    split = json.loads(raw)
    counts = {k: len(split[k]) for k in ("train", "val", "test")}
    if counts != {"train": 1239, "val": 154, "test": 154}:
        raise SystemExit(f"SPLIT_COUNT {counts}")
    return split


def calibration_development(val_ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Same derivation as TESTLY_SPATIAL_CONTROL_V2 reconstruct()."""
    derived = sorted(val_ids, key=lambda pid: hashlib.sha256(f"{seed}::{pid}".encode()).hexdigest())
    calibration = list(derived[:31])
    development = list(derived[31:])
    if len(calibration) != 31 or len(development) != 123:
        raise SystemExit("CAL_DEV_COUNT")
    return calibration, development


def assert_allowed(ids: list[str], test_ids: set[str]) -> None:
    leaked = [pid for pid in ids if pid in test_ids]
    if leaked:
        raise SystemExit(f"TEST_SPLIT_TOUCHED {leaked[:3]}")


def parse_pid(pid: str) -> tuple[str, str, str]:
    source, stem, sha10 = pid.split("::")
    return source, stem, sha10


def _stem_of(name: str) -> str:
    return Path(name).stem


def _extract_allowed(zip_path: Path, dest: Path, stems: set[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            lowered = member.filename.replace("\\", "/").lower()
            if "/dirty/" not in lowered and "/clean/" not in lowered:
                continue
            if _stem_of(member.filename) not in stems:
                continue
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise SystemExit("ZIP_PATH_ESCAPE")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open("wb") as out:
                    out.write(src.read())


def _find_image(root: Path, kind: str, stem: str) -> Path | None:
    matches = []
    for path in root.rglob(stem + ".*"):
        lowered = str(path).replace("\\", "/").lower()
        if f"/{kind}/" in lowered and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            matches.append(path)
    return matches[0] if matches else None


def index_allowed_pairs(drive_root: Path, work_root: Path, ids: list[str], test_ids: set[str]) -> dict:
    assert_allowed(ids, test_ids)
    by_source: dict[str, set[str]] = {}
    for pid in ids:
        source, stem, _ = parse_pid(pid)
        by_source.setdefault(source, set()).add(stem)
    extract_root = work_root / "_allowed_images"
    for source, stems in by_source.items():
        zip_path = _find_named(drive_root, source + ".zip")
        dest = extract_root / source
        if zip_path is not None:
            _extract_allowed(zip_path, dest, stems)
        elif not dest.exists():
            raise SystemExit(f"MISSING_ZIP {source}.zip under {drive_root}")
    pairs = {}
    for pid in ids:
        source, stem, sha10 = parse_pid(pid)
        dirty = _find_image(extract_root / source, "dirty", stem)
        clean = _find_image(extract_root / source, "clean", stem)
        if dirty is None or clean is None:
            raise SystemExit(f"MISSING_PAIR {pid}")
        dirty_sha = sha256_file(dirty)
        if not dirty_sha.startswith(sha10):
            raise SystemExit(f"DIRTY_SHA_MISMATCH {pid}")
        clean_sha = sha256_file(clean)
        pairs[pid] = {
            "id": pid,
            "dirty": str(dirty),
            "clean": str(clean),
            "dirty_sha": dirty_sha,
            "clean_sha": clean_sha,
        }
    return pairs


def _find_named(root: Path, name: str, depth: int = 5) -> Path | None:
    direct = root / name
    if direct.is_file():
        return direct
    if depth <= 0 or not root.exists():
        return None
    for child in root.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            found = _find_named(child, name, depth - 1)
            if found is not None:
                return found
    return None


def find_split(drive_root: Path) -> Path:
    for relative in (
        Path("Testly_Learned_Paper_Scale_v1") / "manifests" / "split.json",
        Path("manifests") / "split.json",
        Path("source") / "V1" / "manifests" / "split.json",
    ):
        candidate = drive_root / relative
        if candidate.is_file():
            return candidate
    found = _find_named(drive_root, "split.json")
    if found is None:
        raise SystemExit(f"SPLIT_NOT_FOUND under {drive_root}")
    return found


def read_gray(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    return image.astype(np.float32) / 255.0


def prepare_pair(pair: dict) -> tuple[np.ndarray, np.ndarray]:
    dirty = read_gray(pair["dirty"])
    target = read_gray(pair["clean"])
    scale = CANONICAL / max(dirty.shape)
    height = max(1, round(dirty.shape[0] * scale))
    width = max(1, round(dirty.shape[1] * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    dirty = cv2.resize(dirty, (width, height), interpolation=interp)
    target = cv2.resize(target, (width, height), interpolation=interp)
    return dirty.astype(np.float32), target.astype(np.float32)


def letterbox(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scale = MODEL_INPUT / max(height, width)
    nh = max(1, round(height * scale))
    nw = max(1, round(width * scale))
    top = (MODEL_INPUT - nh) // 2
    left = (MODEL_INPUT - nw) // 2
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    canvas = np.ones((MODEL_INPUT, MODEL_INPUT), np.float32)
    mask = np.zeros_like(canvas)
    resized = cv2.resize(image, (nw, nh), interpolation=interp)
    canvas[top:top + nh, left:left + nw] = resized
    mask[top:top + nh, left:left + nw] = 1
    return canvas, mask, np.array([top, left, nh, nw], np.float32)


def letterbox_stack(channels: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    planes = []
    mask = geom = None
    for channel in channels:
        canvas, mask, geom = letterbox(channel)
        planes.append(canvas)
    return np.stack(planes, 0).astype(np.float32), mask, geom


class PageStore:
    """One shared Z cache. Identity is pair id, dirty sha, clean sha, and the operator spec."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hash = spec_hash()

    def path(self, pid: str, dirty_sha: str, clean_sha: str) -> Path:
        return self.root / (page_cache_name(pid, dirty_sha, clean_sha, self.hash) + ".npz")

    def ensure(self, pid: str, pair: dict, test_ids: set[str]) -> dict:
        if pid in test_ids:
            raise SystemExit("TEST_SPLIT_TOUCHED")
        clean_sha = pair.get("clean_sha") or ""
        if not clean_sha:
            raise SystemExit(f"CLEAN_SHA_MISSING {pid}")
        path = self.path(pid, pair["dirty_sha"], clean_sha)
        if path.exists():
            return self._read(path, pid, pair["dirty_sha"], clean_sha)
        dirty, target = prepare_pair(pair)
        z = normalize_s31(dirty)
        masks = structure_masks(target)
        meta = {
            "id": pid,
            "dirty_sha": pair["dirty_sha"],
            "clean_sha": clean_sha,
            "operator_hash": self.hash,
            "prep_version": PREP_VERSION,
            "target_resize_count": 1,
        }
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp, dirty=dirty, Z=z, T=target,
            ink=masks["ink"], edge=masks["edge"], bg=masks["bg"],
            metadata=json.dumps(meta),
        )
        tmp.replace(path)
        return self._read(path, pid, pair["dirty_sha"], clean_sha)

    def _read(self, path: Path, pid: str, dirty_sha: str, clean_sha: str) -> dict:
        with np.load(path, allow_pickle=False) as blob:
            raw_meta = blob["metadata"]
            if isinstance(raw_meta, np.ndarray):
                raw_meta = raw_meta.item()
            meta = json.loads(raw_meta if isinstance(raw_meta, str) else str(raw_meta))
            if (
                meta.get("id") != pid
                or meta.get("dirty_sha") != dirty_sha
                or meta.get("clean_sha") != clean_sha
                or meta.get("operator_hash") != self.hash
            ):
                raise SystemExit(f"Z_CACHE_MISMATCH {pid}")
            return {
                "dirty": np.ascontiguousarray(blob["dirty"], dtype=np.float32),
                "Z": np.ascontiguousarray(blob["Z"], dtype=np.float32),
                "T": np.ascontiguousarray(blob["T"], dtype=np.float32),
                "ink": blob["ink"],
                "edge": blob["edge"],
                "bg": blob["bg"],
                "meta": meta,
            }
