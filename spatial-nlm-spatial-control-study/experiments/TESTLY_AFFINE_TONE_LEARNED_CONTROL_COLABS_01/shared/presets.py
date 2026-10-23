"""One builder per architecture. Scripts call a single builder; they do not switch models."""
from __future__ import annotations

from pathlib import Path


def _base(root: Path, **extra) -> dict:
    cfg = {
        "package_root": str(root),
        "allow_test_split": False,
        "seed": 20261014,
        "epochs": 40,
        "patience": 8,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "batch_size": None,
        "accumulation": None,
        "resume": True,
        "warm_start": False,
        "encoder_checkpoint": None,
        "lightweight": False,
        "augment": False,
        "loss_preset": "mse",
        "clamp_backward": "exact",
        "lattice_step": 0.05,
        "compute_current6": True,
        "num_panels": 4,
        "workers": 0,
        "bench03_present": False,
        "tone_bank_file": "tone_bank_bench01_s05.json",
        "bounds_file": "bounds_fallback.json",
        "fixed_file": "fixed_references.json",
        "manifold_file": "manifold.json",
        "delay_patience": False,
        "out_ch": 2,
    }
    cfg.update(extra)
    return cfg


def _continuous(until: int = 10**9) -> list[dict]:
    return [{
        "until_epoch": until,
        "stage": "full",
        "image_weight": 1.0,
        "lambda_kl": 0.0,
        "lambda_regret": 0.0,
    }]


def _discrete_stages() -> list[dict]:
    return [
        {"until_epoch": 5, "stage": "head", "image_weight": 0.0, "lambda_kl": 1.0, "lambda_regret": 0.25},
        {"until_epoch": 15, "stage": "upper", "image_weight": 0.0, "lambda_kl": 1.0, "lambda_regret": 0.25},
        {"until_epoch": 10**9, "stage": "full", "image_weight": 1.0, "lambda_kl": 0.25, "lambda_regret": 0.25},
    ]


def _distill_stages() -> list[dict]:
    return [
        {"until_epoch": 8, "stage": "head", "image_weight": 0.0, "lambda_kl": 1.0, "lambda_regret": 0.25},
        {"until_epoch": 20, "stage": "upper", "image_weight": 0.0, "lambda_kl": 1.0, "lambda_regret": 0.25},
        {"until_epoch": 10**9, "stage": "full", "image_weight": 1.0, "lambda_kl": 0.1, "lambda_regret": 0.1},
    ]


def build_exp01(root) -> dict:
    return _base(
        Path(root),
        experiment_id="EXP01_DISCRETE_G128_TONE_CLASSIFIER",
        family="discrete",
        grid=128,
        input_mode="x",
        supervision="oracle_class",
        oracle_kind="bank_hybrid",
        schedule=_discrete_stages(),
        delay_patience=True,
        research_question="Can a G128 classifier over the frozen five S05 presets recover spatial tone headroom?",
        stop_go=(
            "If development recovery of the frozen-bank hybrid headroom over S05 V03 is under 20%, "
            "discrete control of this bank looks difficult. Over 70% makes it a strong candidate. "
            "The thresholds are exploratory labels, not a result."
        ),
    )


def build_exp02(root) -> dict:
    return _base(
        Path(root),
        experiment_id="EXP02_CONTINUOUS_BW_G128_REGRESSION",
        family="continuous_bw",
        grid=128,
        input_mode="z",
        supervision="image",
        oracle_kind="bank_hybrid",
        schedule=_continuous(),
        research_question="Can a network directly regress spatial black/white controls at G128?",
        stop_go=(
            "If development recovery of the frozen-bank hybrid headroom over S05 V03 is under 20%, "
            "continuous black/white regression looks difficult. Over 70% makes it a strong candidate. "
            "INPUT_MODE x, z, and xz are separate runs of this same architecture."
        ),
    )


def build_exp03(root) -> dict:
    return _base(
        Path(root),
        experiment_id="EXP03_CONTINUOUS_AFFINE_G128_GAIN_BIAS",
        family="gain_bias",
        grid=128,
        input_mode="z",
        supervision="image",
        oracle_kind="bank_hybrid",
        schedule=_continuous(),
        research_question="Is gain/bias a more stable neural parameterization than black/white for the same affine family?",
        stop_go=(
            "Compare this run with EXP02, not with a new operator. Under 20% recovery means this "
            "parameterization was the harder geometry on this split. Over 70% means it is a viable controller. "
            "The two parameterizations share an affine family and use different bounded boxes."
        ),
    )


def build_exp04(root) -> dict:
    return _base(
        Path(root),
        experiment_id="EXP04_ONE_DIMENSIONAL_TONE_FIELD",
        family="scalar",
        grid=128,
        input_mode="z",
        supervision="image",
        oracle_kind="bank_hybrid",
        out_ch=1,
        schedule=_continuous(),
        research_question="Do spatial black/white controls lie near a single measured tone manifold?",
        stop_go=(
            "Training does not start without configs/manifold.json status measured_from_bench03. "
            "BENCH_03 has not measured that curve. If a later measured manifold recovers under 20% "
            "of the frozen-bank hybrid headroom, a 1-D control is too thin. Over 70% means the manifold carries the tone."
        ),
    )


def build_exp05(root) -> dict:
    return _base(
        Path(root),
        experiment_id="EXP05_GLOBAL_PAGE_BW_REGRESSION",
        family="global",
        grid=1,
        input_mode="x",
        supervision="image",
        oracle_kind="bank_page",
        schedule=_continuous(),
        research_question="Are two page-level numbers enough to control the confirmed tone operator?",
        stop_go=(
            "BENCH_02 page-oracle headroom above the best frozen preset S05 V04 was 0.414 dB, under the "
            "0.50 dB material line. Recovery here uses the frozen five-preset page oracle against S05 V03. "
            "Under 20% means the page gap is not being learned. Over 70% means the page oracle is learnable "
            "even when the absolute gap stays small. A constant map is this architecture, not a failure."
        ),
    )


def build_exp06(root) -> dict:
    return _base(
        Path(root),
        experiment_id="EXP06_COARSE_G32_BW_CONTROLLER",
        family="continuous_bw",
        grid=32,
        input_mode="z",
        supervision="image",
        oracle_kind="bank_hybrid",
        schedule=_continuous(),
        research_question="Is a smooth G32 black/white field enough, or does tone need a finer grid?",
        stop_go=(
            "Compare with EXP07 and EXP02. Under 20% recovery means G32 is too coarse for this headroom. "
            "Over 70% means a coarse smooth field is already a strong controller."
        ),
    )


def build_exp07(root) -> dict:
    return _base(
        Path(root),
        experiment_id="EXP07_G64_BW_CONTROLLER",
        family="continuous_bw",
        grid=64,
        input_mode="z",
        supervision="image",
        oracle_kind="bank_hybrid",
        schedule=_continuous(),
        research_question="Does G64 capture the spatial tone field that G32 misses and G128 might not need?",
        stop_go=(
            "This is the middle grid. Under 20% recovery, with G32 also low, points away from resolution "
            "and toward the parameterization. Over 70%, while G32 stays low, means G64 is enough."
        ),
    )


def build_exp08(root) -> dict:
    return _base(
        Path(root),
        experiment_id="EXP08_DUAL_INPUT_X_Z_G128_BW",
        family="continuous_bw",
        grid=128,
        input_mode="xz",
        supervision="image",
        oracle_kind="bank_hybrid",
        schedule=_continuous(),
        research_question="Does the pair [dirty x, normalized Z] control tone better than either channel alone?",
        stop_go=(
            "Compare with EXP02 runs at INPUT_MODE x and INPUT_MODE z. Under 20% means the dual input "
            "does not unlock the frozen-bank headroom. Over 70% makes [x, Z] a strong input."
        ),
    )


def build_exp09(root) -> dict:
    return _base(
        Path(root),
        experiment_id="EXP09_DISCRETE_ORACLE_DISTILLATION_THEN_IMAGE_LOSS",
        family="discrete",
        grid=128,
        input_mode="x",
        supervision="oracle_class",
        oracle_kind="bank_hybrid",
        schedule=_distill_stages(),
        delay_patience=True,
        research_question="Does oracle distillation before image loss train the discrete bank more stably than EXP01?",
        stop_go=(
            "Compare with EXP01. Under 20% after both stages means the bank is hard even with distillation. "
            "Over 70%, and above EXP01, makes the two-stage schedule the discrete candidate."
        ),
    )


def build_exp10(root) -> dict:
    cfg = _base(
        Path(root),
        experiment_id="EXP10_ORACLE_PARAMETER_REGRESSION",
        family="continuous_bw",
        grid=128,
        input_mode="z",
        supervision="oracle_param",
        oracle_kind="bank_hybrid",
        schedule=[{
            "until_epoch": 10**9,
            "stage": "full",
            "image_weight": 0.0,
            "lambda_kl": 0.0,
            "lambda_regret": 0.0,
        }],
        research_question="Can the network regress offline lattice black/white maps without seeing the clean page at inference?",
        stop_go=(
            "The lattice is a search grid of the fallback box, not an optimum. Recovery against the frozen-bank "
            "hybrid answers whether those regressed controls beat S05 V03. Under 20% means parameter regression "
            "is not reaching the bank headroom. Over 70% means the offline maps are a usable teacher."
        ),
    )
    return cfg


BUILDERS = {
    1: build_exp01,
    2: build_exp02,
    3: build_exp03,
    4: build_exp04,
    5: build_exp05,
    6: build_exp06,
    7: build_exp07,
    8: build_exp08,
    9: build_exp09,
    10: build_exp10,
}
