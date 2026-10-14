"""Frozen-inference checks for learned paper normalization. No training."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learned_paper_normalization import (
    CANONICAL_LONG_SIDE,
    MODEL_INPUT,
    S_VALUES,
    EXPECTED_PARAMS,
    estimate_s,
    load_s_estimator,
    normalize_paper,
    prepare_canonical_image,
    prepare_estimator_input,
    read_gray01,
    regression_against_historical,
)

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "results" / "bettertest" / "prepared" / "0000015_dirty.png"


@pytest.fixture(scope="module")
def bundle():
    return load_s_estimator("s-estimator.pt")


@pytest.fixture(scope="module")
def canonical_page(bundle):
    return prepare_canonical_image(PREPARED)


def test_checkpoint_strict_load(bundle):
    assert bundle.parameter_count == EXPECTED_PARAMS
    assert bundle.s_values == S_VALUES
    assert bundle.model_input == MODEL_INPUT
    assert bundle.model.training is False
    assert bundle.epoch is not None
    assert bundle.checkpoint_path.is_file()


def test_probability_sum(bundle, canonical_page):
    estimate = estimate_s(bundle, canonical_page)
    assert np.isfinite(estimate.logits).all()
    assert np.isfinite(estimate.probabilities).all()
    assert abs(float(estimate.probabilities.sum()) - 1.0) < 1e-5


def test_selected_s_in_candidate_set(bundle, canonical_page):
    estimate = estimate_s(bundle, canonical_page)
    assert estimate.selected_s in S_VALUES
    assert estimate.index == int(np.argmax(estimate.probabilities))


def test_deterministic_inference(bundle, canonical_page):
    first = estimate_s(bundle, canonical_page)
    second = estimate_s(bundle, canonical_page)
    assert first.selected_s == second.selected_s
    assert np.array_equal(first.logits, second.logits)
    assert np.array_equal(first.probabilities, second.probabilities)


def test_normalize_paper_is_finite_and_unit_interval():
    image = np.linspace(0.05, 0.95, 128 * 96, dtype=np.float32).reshape(128, 96)
    for scale in S_VALUES:
        output = normalize_paper(image, scale)
        assert output.shape == image.shape
        assert np.isfinite(output).all()
        assert float(output.min()) >= 0.0
        assert float(output.max()) <= 1.0


def test_historical_local_31_regression():
    image = read_gray01(PREPARED)
    report = regression_against_historical(image)
    assert report["max_abs_difference"] <= 1e-5
    assert report["mean_abs_difference"] <= 1e-5


def test_canonical_long_side_is_1024():
    wide = np.zeros((1500, 2000), dtype=np.uint8)
    tall = np.zeros((250, 100), dtype=np.uint8)
    wide_out = prepare_canonical_image(wide)
    tall_out = prepare_canonical_image(tall)
    assert wide_out.shape == (768, 1024)
    assert tall_out.shape == (1024, 410)
    assert max(wide_out.shape) == CANONICAL_LONG_SIDE
    assert max(tall_out.shape) == CANONICAL_LONG_SIDE


def test_estimator_canvas_is_512_and_mask_is_valid():
    canonical = prepare_canonical_image(np.zeros((300, 100), dtype=np.uint8))
    canvas, mask = prepare_estimator_input(canonical, MODEL_INPUT)
    assert canvas.shape == (MODEL_INPUT, MODEL_INPUT)
    assert mask.shape == (MODEL_INPUT, MODEL_INPUT)
    assert canvas.dtype == np.float32
    assert set(np.unique(mask).tolist()) <= {0.0, 1.0}
    assert 0 < float(mask.sum()) < MODEL_INPUT * MODEL_INPUT
    assert np.all(canvas[mask == 0] == 1.0)
    rows, cols = np.where(mask == 1)
    top, left = int(rows.min()), int(cols.min())
    height = int(rows.max() - top + 1)
    width = int(cols.max() - left + 1)
    assert top == (MODEL_INPUT - height) // 2
    assert left == (MODEL_INPUT - width) // 2
    assert max(height, width) == MODEL_INPUT
