# Better-clean-target diagnostic with frozen Scalar/Spatial NLM

Completed diagnostic evaluation of three new pairs. **No retraining, calibration, checkpoint selection or processing changes were performed.** Old CamScanner results remain unchanged.

## Frozen system and pair preparation

* scalar: `checkpoints\scalar\best.pt`, selected step 0, SHA256 `6ab484abcf791d7cfa1104a9d2d1f5d1d461e4fe11b902048f4536d78c824ee2`.
* spatial: `checkpoints\spatial\best.pt`, selected step 400, SHA256 `50a3036259b30c93bc00496b117678ac546da31c8f396b4c1f78e5688c9d5ba0`.

Shared preprocessing `local_31_0.0`; 31x31 closing, Gaussian sigma31/6, grayscale /255. NLM patch7/search21, tile256 with 13px halo. Bounds [0.0005213606195075, 0.1], historical h*=0.03845335368888; postprocessing [0,1] clip. Imported frozen implementations are unchanged.

Exact filename identity defines pairing. All three dirty/clean dimensions match. Same SIFT gate and the original shared resize/warp/valid-rectangle preparation are used. Identity resize is a no-op for equal dimensions.

| Pair | Source WxH | Registration | Prepared WxH | Feature residual median/p95 (px) |
|---|---|---|---|---|
| 0000015 | 760x928 | False | 760x928 | 0.1633 / 0.5992 |
| 0000025 | 580x1044 | True | 575x1040 | 0.1329 / 0.4708 |
| 0000035 | 708x936 | False | 708x936 | 0.0229 / 0.1290 |

No exact dirty-source SHA256 matches to the prior 100 pairs. Detected prior-content matches: 0. See `results/bettertest/content_overlap.json`; absence of a detected match is not source-provenance proof.

## Metrics

| Pair | Dirty PSNR | Scalar PSNR | Spatial PSNR | ΔPSNR | Dirty SSIM | Scalar SSIM | Spatial SSIM | ΔSSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0000015 | 8.993591 | 24.518421 | 24.462774 | -0.055647 | 0.867155 | 0.969634 | 0.966396 | -0.003238 |
| 0000025 | 9.403995 | 23.989790 | 23.991615 | 0.001825 | 0.863586 | 0.969464 | 0.969031 | -0.000433 |
| 0000035 | 7.044511 | 28.576042 | 28.587238 | 0.011196 | 0.783433 | 0.989673 | 0.986575 | -0.003098 |
| Arithmetic mean | 8.480699 | 25.694751 | 25.680543 | -0.014208 | 0.838058 | 0.976257 | 0.974001 | -0.002256 |

Source: [metrics.json](results/bettertest/metrics.json) -> `means`, `per_image`; [CSV](results/bettertest/per_image.csv). Metrics are computed from float32 arrays, never display PNGs. Dirty metrics use the same prepared grayscale dirty input/valid rectangle as both methods, before document preprocessing.

| Cohort | Scalar PSNR | Spatial PSNR | ΔPSNR | Scalar SSIM | Spatial SSIM | ΔSSIM |
|---|---:|---:|---:|---:|---:|---:|
| Original CamScanner test (20) | 25.896911 | 26.324671 | 0.427759 | 0.960412 | 0.928183 | -0.032229 |
| Bettertest diagnostic (3) | 25.694751 | 25.680543 | -0.014208 | 0.976257 | 0.974001 | -0.002256 |

Spatial PSNR is higher on 2/3; spatial SSIM is lower on 3/3.

## Target-cleanliness descriptors

Descriptors were fixed before scoring and do not select any restoration parameters. Raw clean source grayscale is used to avoid a registration/resampling confound in the descriptor comparison.

| Equal-image mean descriptor | Previous test20 | Previous all100 | Bettertest3 |
|---|---:|---:|---:|
| bright_background_fraction | 0.68060839 | 0.67734208 | 0.77245182 |
| bright_background_variance | 0.00018714058 | 0.00024237076 | 2.9420125e-07 |
| background_laplacian_abs | 0.0053277497 | 0.0053098409 | 0.00021242336 |
| background_hf_energy | 0.00012575004 | 0.00011687174 | 4.1984394e-06 |
| estimated_background_noise | 0.00099618779 | 0.0010282448 | 6.204321e-05 |
| global_laplacian_abs | 0.039234229 | 0.03989821 | 0.055655985 |
| white_fraction | 0.79386306 | 0.7865372 | 0.9165995 |
| edge_density | 0.17890799 | 0.17649399 | 0.1081842 |

Source: [target_cleanliness.json](results/bettertest/target_cleanliness.json), containing formulas, raw source hashes and all per-image values. Background mask is gray>.8 and outside an 11x11 dilation of gray<.7, with the 5px border excluded. Population variance, reflect101 Laplacian and Immerkaer absolute response summarize this support. White fraction is gray>=.98; edge density uses Sobel magnitude>.2. These are content-dependent descriptive proxies, not objective perfection or calibrated camera noise.

New mean versus old test mean: bright_background_variance lower=True, background_laplacian_abs lower=True, estimated_background_noise lower=True; white fraction higher=True.

## Interpretation and limitations

The spatial SSIM deficit persists on all three, with mean -0.002256 versus the historical -0.032229. The deficit is smaller, not eliminated. PSNR improves on 2/3 and the three-image mean paired difference is -0.014208 dB, so the earlier uniform PSNR gain is not replicated. Cleaner backgrounds do not by themselves eliminate the SSIM loss here; whether old target imperfections made a partial contribution remains unresolved because content and framing change too.

Changing both dirty photographs and clean references also changes content, framing, margins, illumination and ink coverage. This is a frozen-system dataset diagnostic, **not a same-input intervention on target cleanliness alone**. New images have large white margins, which affect whole-image metrics and cleanliness descriptors. Three selected pairs, one original training seed, application-produced targets, grayscale scoring, sparse-feature alignment and a subpixel warp on one pair limit inference. No publication-level, production or medical claim follows.

## Artifacts and reproduction

* [0000015 eight-panel figure](results/bettertest/figures/0000015_comparison.png).
* [0000025 eight-panel figure](results/bettertest/figures/0000025_comparison.png).
* [0000035 eight-panel figure](results/bettertest/figures/0000035_comparison.png).

`results/bettertest/` also preserves explicit pairs, preparation matrices/ROIs, source/checkpoint/code SHA256s, environment, CSV/JSON scores, original-artifact preservation proof, float32 output/error/h-map NPYs and logs. The first exploratory audit was read-only and then incorporated into the executable audit. No pair was excluded.

One preliminary audit attempt failed on an empty SIFT inlier array during prior-content screening. A guard was added and the audit rerun before any scoring; the failed log remains. No frozen implementation was changed and no model evaluation failed. Later report wording/layout revisions read only saved results; evaluation_started.json records the exact helper hash used at scoring.

Environment: Python 3.11.9, Torch 2.10.0+cu128, CUDA 12.8, OpenCV 4.13.0, NVIDIA GeForce RTX 5060 Laptop GPU; strict deterministic algorithms, no AMP/TF32, original seed20261014.

From this experiment directory, with an EMPTY diagnostic result directory:

```powershell
$env:PYTHONHASHSEED = "20261014"
python bettertest_diagnostic.py audit
python bettertest_diagnostic.py evaluate
python bettertest_diagnostic.py render
python bettertest_diagnostic.py report
```

For an independent reproduction, use `--output results/bettertest_reproduction` with each command. Preserve this attempt. Existing evaluation-start/metrics markers refuse a second score; render/report read saved arrays and do not run the models. Do not run `train.py`, `calibrate.py` or the original `evaluate.py` for this diagnostic.

The exact helper used for scoring and the frozen source files are additionally preserved under `results/bettertest/source_snapshot/`; SHA256s are in `source_snapshot.json`. `evaluation_started.json` identifies the scoring version. Later helper changes affect report wording only.
