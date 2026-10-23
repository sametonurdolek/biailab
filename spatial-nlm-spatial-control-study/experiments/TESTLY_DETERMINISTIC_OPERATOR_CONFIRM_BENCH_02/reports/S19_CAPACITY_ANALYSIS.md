# S19

| Candidate | PSNR | ΔPSNR | CI lo | SSIM | ΔSSIM | PSNR wins | Severe frac | Structural |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S19_V01 | 19.513062 | -6.573446 | -6.989560 | 0.928975 | -0.018558 | 0/123 | 1.000 | FAIL |
| S19_V02 | 22.991295 | -3.095213 | -3.368368 | 0.952350 | 0.004817 | 2/123 | 0.959 | FAIL |
| S19_V03 | 25.723159 | -0.363350 | -0.675643 | 0.964377 | 0.016843 | 44/123 | 0.260 | PASS |
| S19_V04 | 26.461757 | 0.375249 | -0.122710 | 0.966453 | 0.018920 | 68/123 | 0.187 | PASS |
| S19_V05 | 24.568777 | -1.517731 | -2.156272 | 0.958810 | 0.011277 | 39/123 | 0.520 | PASS |

| Stage | PSNR | ΔPSNR vs previous | SSIM | ΔSSIM vs previous |
|---|---:|---:|---:|---:|
| Current6 hybrid | 26.086509 |  | 0.947533 |  |
| Fixed S19_V04 | 26.461757 | 0.375249 | 0.966453 | 0.018920 |
| Per-page global oracle | 27.040235 | 0.578478 | 0.969476 | 0.003023 |
| G128 hard oracle | 28.940933 | 1.900698 | 0.980698 | 0.011221 |
| G128 hybrid oracle | 28.911216 | -0.029717 | 0.980141 | -0.000557 |

Preset sigmas and gains are the frozen BENCH_01 values. Global-oracle page counts are in PARAMETER_SELECTION_DISTRIBUTIONS.md.
