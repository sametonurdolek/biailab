# Colab usage

## Before Run All

1. Put the three dataset zips where the script can see them: `binarizeddataset.zip`, `acceptednewnewbinarize.zip`, `acceptednewnewbinarizelast.zip`.
2. Put `split.json` at one of these locations under `DRIVE_ROOT`, or anywhere the depth-3 search can reach:
   - `Testly_Learned_Paper_Scale_v1/manifests/split.json`
   - `manifests/split.json`
   - `source/V1/manifests/split.json`
3. Copy the whole `TESTLY_AFFINE_TONE_LEARNED_CONTROL_COLABS_01` folder to Drive. Do not upload only the `.py` file.
4. Edit `DRIVE_ROOT` if the Drive mount is not `/content/drive/MyDrive`.
5. Leave `ALLOW_TEST_SPLIT = False`.

Optional: `Testly_Learned_Paper_Scale_v21/checkpoints/best.pt` with sha256 `1c2a30b4beb308474c24ab7e3646430606fb3a7a5d2be495e17cd587fcbef0f9` and epoch 54. Set `WARM_START = True` only when that file should be required. A missing file then stops the run. The default is scratch, so Run All works without it. The stem is skipped when the input has two channels, because its shape no longer matches.

OpenCV is not pinned below 4.12. BENCH_02 was measured on 4.14.0. The script records whatever version the runtime has. It does not reinstall PyTorch when `torch` is already importable.

## Order

Cheapest informative run first:

1. `05_GLOBAL_PAGE_BW_REGRESSION.py`
2. `01_DISCRETE_G128_TONE_CLASSIFIER.py`
3. `07_G64_BW_CONTROLLER.py`
4. `02_CONTINUOUS_BW_G128_REGRESSION.py`
5. `08_DUAL_INPUT_X_Z_G128_BW.py`
6. `04_ONE_DIMENSIONAL_TONE_FIELD.py` (stops until a measured manifold exists)
7. `03_CONTINUOUS_AFFINE_G128_GAIN_BIAS.py`
8. `09_DISCRETE_ORACLE_DISTILLATION_THEN_IMAGE_LOSS.py` and `10_ORACLE_PARAMETER_REGRESSION.py`

EXP06 (G32) can be inserted between the global run and G64 when the grid question is the next one to answer.

## Knobs that do not create a new architecture

- `INPUT_MODE`: `x`, `z`, or `xz`
- `LOSS_PRESET`: `mse`, `mse_ssim`, `mse_ssim_edge`, `mse_structural_smooth`
- `CLAMP_BACKWARD`: `exact` or `straight_through`
- `SUPERVISION` on EXP05: `image` or `oracle_param`
- `BATCH_SIZE`, `ACCUMULATION`, `EPOCHS`, `LIGHTWEIGHT`, `WARM_START`, `RESUME`

Each of those still writes the experiment id of that script. Changing `INPUT_MODE` inside EXP02 and running again with `RESUME = False` creates `EXP02_..._2` so the first run is kept.

## Batch size

The script reads GPU memory and prints a batch and an accumulation count. The aim is an effective batch near 8.

| GPU memory | Global | G32 | G64 | G128 |
|---|---:|---:|---:|---:|
| none (CPU) | 1 | 1 | 1 | 1 |
| under 14 GB | 8 | 4 | 2 | 1 |
| 14 GB up to 16 GB | 8 | 4 | 2 | 2 |
| 16 GB up to 24 GB (T4) | 16 | 8 | 4 | 2 |
| 24 GB and up (L4, A100) | 16 | 8 | 4 | 4 |

Full-resolution tone loss dominates memory, not the 1.8M-parameter backbone. A CUDA out-of-memory error stops with `CUDA_OOM` and leaves `latest.pt` resumable. Set `BATCH_SIZE = 1` and run again.

## Disconnects

`RESUME = True` loads `latest.pt` from that experiment folder, including the optimizer and the frozen tau, only when the run signature matches. A different `INPUT_MODE`, loss preset, grid, or supervision aborts with `CHECKPOINT_SIGNATURE_MISMATCH` and does not reuse the weights. A tau that does not match the recomputed calibration median aborts with `TAU_MISMATCH`. Changing one of those knobs requires `RESUME = False`, which keeps the old folder and writes `EXP0N_..._2`.

## End of the cell

The cell prints `EXPERIMENT`, `BEST EPOCH`, `PSNR`, `SSIM`, `FIXED BASELINE`, `ORACLE`, `HEADROOM RECOVERED`, `STRUCTURAL GUARD`, `CHECKPOINT`, and `DRIVE OUTPUT`. PSNR and SSIM in that block are development scores of the calibration-selected checkpoint. `CALIBRATION_PSNR` is printed on the next line. `STOP_GO` is an exploratory label (`BELOW_20`, `INTERMEDIATE`, `ABOVE_70`, or `UNMEASURED`). It is not a claim about a run that has not finished.
