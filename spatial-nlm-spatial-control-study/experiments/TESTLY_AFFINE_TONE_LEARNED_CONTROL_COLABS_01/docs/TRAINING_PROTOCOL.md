# Training protocol

This file describes what a Colab run does. It does not report a trained model.

## Splits

The frozen split file must hash to `ab03834bcf1cb56a752228287b830eb30a33c9ff4fb2cb0a6f7d3816c52561e1` and contain 1239 train, 154 val, and 154 test ids.

Calibration is the first 31 validation ids after sorting by `sha256("{20261014}::{pair_id}")`. Development is the other 123. That is the same derivation as the spatial-control V2 experiment.

Train pages update weights. Calibration PSNR selects the checkpoint. Development is scored once, at that checkpoint. Test ids are held only as a rejection set. Requesting one raises `TEST_SPLIT_TOUCHED` before an image byte is read. Zip extraction skips every member whose stem is not in train, calibration, or development.

## Operator cache

Each allowed page is stored once under:

```text
DRIVE_ROOT/Testly_Affine_Tone_DL/_shared_cache/z_s31/
```

The filename is the hash of pair id, dirty sha256, clean sha256, and the operator spec (`s=31`, sigma `s/6`, floor `0.1`, reflect-101, long side 1024, one resize). A file is reused only when those fields match. The dirty sha256 must start with the ten-character hash in the pair id. The clean sha256 is part of the identity because the stored target, ink mask, and edge mask come from the clean page.

Train and calibration pages are cached before the first optimizer step. Development pages, their discrete cell costs, and their lattice targets are written only after a checkpoint has been selected. If no epoch passes the guard, those development caches are not built.

Discrete costs, lattice targets, S05 V03 metrics, the frozen-bank oracle, and the Current6 hybrid oracle use the same shared root. Architectures do not keep private copies of those arrays. Cost and lattice filenames also include the clean sha256.

## Optimization

- Seed `20261014` for Python, NumPy, and PyTorch. cuDNN deterministic is on and benchmark is off. TF32 is off.
- AdamW, learning rate `3e-4`, weight decay `1e-4`, gradient clip `1.0`.
- Default 40 epochs, patience 8.
- Mixed precision wraps the network forward on CUDA only. The tone operator and the loss run in float32.
- `num_workers` is 0 so cache writes do not race.
- Augmentation is off. Rounded cell edges are not an exact mirror, so a flip would have to rebuild the oracle. The flip path does that rebuild if `augment` is turned on.

## What is allowed to improve the checkpoint

An epoch is eligible when calibration ink MAE and edge MAE are each at most 1.25× the calibration scores of fixed S05 V03, and calibration SSIM is at most `0.02` below that same fixed baseline. The `0.02` drop is a guard (`structural_ssim_drop_max`), not a tuned optimum. Among eligible epochs, the highest calibration PSNR is `checkpoints/best.pt`. An ineligible epoch is not promoted and does not reset patience.

For the staged discrete runs, the early-stop clock starts when image loss turns on. A classification-era checkpoint can still be the one that is kept if later image epochs never beat its calibration PSNR.

`latest.pt` is written every epoch and is the resume point. `best.pt` is written immediately on each promotion. If no epoch passes the guard, `best.pt` is not created.

Resume loads `latest.pt` only when its `run_signature` matches the current run. The signature covers experiment id, family, input mode, loss preset, grid, supervision, image-with-oracle, clamp, lightweight, output channels, oracle kind, augmentation, warm start, operator hash, bank hash, minimum span, loss weights, the endpoint box, and the stage schedule. A mismatch raises `CHECKPOINT_SIGNATURE_MISMATCH` before any cache write or config overwrite. Batch size, epoch count, learning rate, and the recorded library versions are stored and are not part of the signature.

Each checkpoint and `config.json` records the git commit and the Python, PyTorch, CUDA, NumPy, and OpenCV versions of the process that started the run.

## Discrete schedule

EXP01:

| Epochs | Frozen | Image loss |
|---|---|---|
| 1–5 | encoder | off |
| 6–15 | stem, stage1, down2, stage2, down3, down4 | off |
| 16–40 | nothing | on, with KL weight 0.25 |

The upper stage matches the V2 thaw: only `stage3` and `stage4` train. EXP09 keeps the teacher through epoch 20 and then drops the KL weight to 0.1.

Cell regret in the classification loss is divided by the frozen calibration tau, so `lambda_regret` is on a similar scale to KL. Tau is the median positive cell regret on the 31 calibration pages and is written into `config.json` before the first optimizer step. The V2 constant `2.98085e-5` is not used.

Validation of a discrete model is argmax labels plus hybrid compose. The training image term, when it is on, is a bilinear upsample of class probabilities. Those are different operators on purpose. The soft mixture is not the deployed page.

## Continuous schedule

One stage. The default loss preset is `mse`, with weights 1, 0, 0, 0. Validation always uses the exact clip, including when training used the straight-through clamp.

## Oracle definitions used in the summary

| Experiment | Denominator of recovery |
|---|---|
| EXP05 | Best of the five frozen S05 presets on the whole page |
| Spatial experiments | Target-chosen G128 hybrid of those five presets |

Current6 six-scale hybrid is an extra reference named `current6_g128_hybrid_oracle`. If it cannot be computed, the summary says `unavailable`. It is not substituted with a guessed number.

The lattice used by EXP05 `oracle_param` and by EXP10 is `arange` with step 0.05 inside the fallback box. It is a search grid. It is not the recovery denominator, and several frozen S05 presets do not lie on that grid.

## Outputs

```text
DRIVE_ROOT/Testly_Affine_Tone_DL/EXP0N_.../
    checkpoints/best.pt
    checkpoints/latest.pt
    metrics/curves.png
    metrics/fixed_s05_v03_calibration.json
    metrics/development.json
    predictions/*.png
    logs/
    config.json
    best_metrics.json
    training_history.csv
```

If resume is off and that folder already has files, the run uses `EXP0N_..._2` rather than overwriting another architecture or an older attempt.
