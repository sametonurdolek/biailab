# Affine tone learned control

Bu paket, donmuş s=31 kâğıt normalizasyonundan sonra affine tonu kontrol eden on ayrı Colab deneyidir. Ağ piksel üretmez. Eğitim bu oturumda başlatılmadı. BENCH_03 yoktur; discrete bank, BENCH_01'in donmuş S05 V01–V05 noktalarıdır. İlk script: `colabs/05_GLOBAL_PAGE_BW_REGRESSION.py`.

The network emits control values. Pixels come from a fixed operator:

```text
close 31, reflect-101
Gaussian sigma 31/6, reflect-101
Z = clip(x / max(B, 0.1), 0, 1)
Y = clip((Z - black) / (white - black), 0, 1)
  = clip(gain * Z + bias, 0, 1)
```

`s` is not a learned output. Clean targets are offline teachers and metrics only. The test split is never opened. `ALLOW_TEST_SPLIT` stays `False`, and the loader raises `TEST_SPLIT_TOUCHED` if a test id is requested.

## What was not found

BENCH_03 is not in this repository. There is no measured affine basin and no measured 1-D manifold. Because of that:

- `configs/tone_bank_bench01_s05.json` is the frozen BENCH_01 S05 catalog, labeled `bench01_s05_frozen`.
- `configs/bounds_fallback.json` is the prompt fallback box, not a fitted optimum: black in `[0, 0.30]`, white in `[0.70, 1]`, minimum span `0.05`.
- `configs/manifold.example.json` has status `EXAMPLE_DO_NOT_TRAIN` and is not loaded.
- EXP04 exits `MANIFOLD_CONFIG_REQUIRED` until `configs/manifold.json` has status `measured_from_bench03`.

## Layout

```text
README.md
architecture_matrix.md
requirements.txt
configs/
shared/          dataset, operator, metrics, models, losses, oracle, utils, presets, engine
colabs/          01 through 10, each runnable with Run All
docs/
```

Copy the whole folder to Drive. A single `.py` file cannot find `shared/`.

## Selection

Checkpoints are chosen by calibration PSNR (31 pages) among epochs whose ink MAE and edge MAE stay within 1.25× fixed S05 V03 and whose calibration SSIM stays within 0.02 of that baseline. The 0.02 drop is a guard, not a tuned optimum. Development (123 pages) is reported for that checkpoint and is not the early-stopping objective. The summary PSNR and SSIM are the development scores. `CALIBRATION_PSNR` is printed beside them. `RESUME = True` refuses a checkpoint whose input mode, loss, grid, or supervision does not match the current run.

Fixed reference is S05 V03 (`black=0.10`, `white=0.92`), the predeclared BENCH_02 confirmation, not a new tune. The oracle used for headroom is the frozen five-preset page oracle for the global model and the frozen five-preset G128 hybrid for spatial models. Current6 hybrid is reported separately as `current6_g128_hybrid_oracle` when the six-scale cache can be filled. It is an oracle, not a deployed network score.

## First run

`colabs/05_GLOBAL_PAGE_BW_REGRESSION.py`

Then the discrete G128 bank, then continuous black/white at G64. The full order is in `docs/COLAB_USAGE.md`.
