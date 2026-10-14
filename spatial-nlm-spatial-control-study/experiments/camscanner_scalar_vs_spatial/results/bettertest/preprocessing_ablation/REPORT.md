# Preprocessing-only ablation on Bettertest

Three prepared dirty/clean pairs; same frozen local_31_0.0 preprocessing and [0,1] clipping. A removes H-Net/NLM. B/C reuse the previously evaluated frozen scalar/spatial outputs and scores. No training, calibration, checkpoint loading, model inference or new geometry preparation occurred. Old results remain unchanged.

## Mean scores

| System | PSNR (dB) | SSIM | ΔPSNR versus A | ΔSSIM versus A |
|---|---:|---:|---:|---:|
| A Preprocess only | 25.660421371 | 0.972269654 | +0.000000000 | +0.000000000 |
| B Preprocess + scalar NLM | 25.694751104 | 0.976257126 | +0.034329732 | +0.003987471 |
| C Preprocess + spatial NLM | 25.680542628 | 0.974000831 | +0.020121257 | +0.001731177 |

## Per-image scores

| Pair | A PSNR | B PSNR | C PSNR | A SSIM | B SSIM | C SSIM | B−A PSNR | C−A PSNR | B−A SSIM | C−A SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0000015 | 24.517915726 | 24.518421173 | 24.462774277 | 0.965285718 | 0.969633996 | 0.966396034 | 0.000505447 | -0.055141449 | 0.004348278 | 0.001110315 |
| 0000025 | 23.989696503 | 23.989789963 | 23.991615295 | 0.967988551 | 0.969464064 | 0.969031155 | 0.000093460 | 0.001918793 | 0.001475513 | 0.001042604 |
| 0000035 | 28.473651886 | 28.576042175 | 28.587238312 | 0.983534694 | 0.989673316 | 0.986575305 | 0.102390289 | 0.113586426 | 0.006138623 | 0.003040612 |

Source: [metrics.json](metrics.json) -> `means`, `per_image`; [CSV](per_image.csv). Means give each image equal weight. Scores use the same peak1, float32 domain and zero-padded Wang SSIM as Bettertest. Existing B/C scores are imported rather than rescored or regenerated.

## What NLM adds on these three pairs

Scalar NLM improves PSNR on 3/3 and SSIM on 3/3 relative to preprocessing only. Mean differences: +0.034329732 dB and +0.003987471 SSIM.

Spatial NLM improves PSNR on 2/3 and SSIM on 3/3 relative to preprocessing only. Mean differences: +0.020121257 dB and +0.001731177 SSIM.

These paired differences directly measure adding the existing NLM controllers after the fixed preprocessing, on this three-pair diagnostic. They do not establish the usefulness or uselessness of NLM for all documents, other bandwidths/windows, or the original 20-image cohort. Three deliberately selected pairs, large white margins, application-produced targets, grayscale evaluation and one original training seed limit generalization. No significance claim is made. No region or parameter was selected after seeing the scores.

## Files and reproduction

* [0000015 comparison](figures/0000015_ablation.png); [preprocessing-only PNG](outputs/0000015_preprocess_only.png); float32 NPY and error map are also saved.
* [0000025 comparison](figures/0000025_ablation.png); [preprocessing-only PNG](outputs/0000025_preprocess_only.png); float32 NPY and error map are also saved.
* [0000035 comparison](figures/0000035_ablation.png); [preprocessing-only PNG](outputs/0000035_preprocess_only.png); float32 NPY and error map are also saved.

`provenance.json` records checkpoint/source/prepared-pair/cached-output hashes and the exact command. `preservation_verification.json` confirms original tracked experiment files are unchanged; `environment.json` records deterministic settings and versions. Logs retain failure tracebacks.

Run in the existing experiment directory with an empty output folder:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:PYTHONHASHSEED = "20261014"
python preprocessing_ablation.py
```

For an independent reproduction use `--output results/bettertest/preprocessing_ablation_reproduction`. Preserve the completed attempt. The original Bettertest pairs, prepared PNGs, cached B/C NPYs and scores are required. No training/evaluation driver is invoked.
