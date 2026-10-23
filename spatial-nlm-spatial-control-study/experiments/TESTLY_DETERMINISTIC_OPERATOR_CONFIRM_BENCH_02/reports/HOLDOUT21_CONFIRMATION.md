# Holdout-21 confirmation

These 21 calibration pages were not in the BENCH_01 discovery ten.

Gate decision: GO. Passers: S05_V03, S05_V05.

A passer needs mean ΔPSNR > 0.50 dB, 95% CI lower bound > 0, mean ΔSSIM > 0, and severe-page fraction ≤ 0.25. Severe means ΔPSNR < -1 dB or ΔSSIM < -0.05.

| Candidate | PSNR | ΔPSNR | CI lo | SSIM | ΔSSIM | PSNR wins | Severe frac | Structural |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S05_V03 | 28.382719 | 1.404018 | 0.851549 | 0.977221 | 0.012382 | 20/21 | 0.000 | PASS |
| S05_V05 | 28.537279 | 1.558578 | 0.716251 | 0.977982 | 0.013143 | 15/21 | 0.095 | PASS |
| S05_V04 | 28.801908 | 1.823207 | 1.088265 | 0.978845 | 0.014006 | 18/21 | 0.048 | PASS |
| S03_V04 | 27.505364 | 0.526663 | -0.229662 | 0.975152 | 0.010314 | 12/21 | 0.190 | PASS |
| S03_V05 | 26.882614 | -0.096087 | -0.928900 | 0.972180 | 0.007341 | 9/21 | 0.333 | PASS |
| S04_V04 | 28.458997 | 1.480296 | 0.771568 | 0.964741 | -0.000098 | 19/21 | 0.048 | FAIL |
| S19_V04 | 27.952153 | 0.973452 | 0.100878 | 0.975894 | 0.011055 | 13/21 | 0.238 | PASS |
| S28_V04 | 28.482700 | 1.503999 | 0.174348 | 0.953591 | -0.011248 | 16/21 | 0.190 | FAIL |

Current6 hybrid mean PSNR 26.978701, SSIM 0.964839.

## S05 chains

| Stage | PSNR | ΔPSNR vs previous | SSIM | ΔSSIM vs previous |
|---|---:|---:|---:|---:|
| Current6 hybrid | 26.978701 |  | 0.964839 |  |
| Fixed S05_V05 | 28.537279 | 1.558578 | 0.977982 | 0.013143 |
| Per-page global oracle | 29.185074 | 0.647795 | 0.980314 | 0.002332 |
| G128 hard oracle | 29.762803 | 0.577729 | 0.983474 | 0.003160 |
| G128 hybrid oracle | 29.830059 | 0.067256 | 0.983227 | -0.000246 |

## S03 chain at V04

| Stage | PSNR | ΔPSNR vs previous | SSIM | ΔSSIM vs previous |
|---|---:|---:|---:|---:|
| Current6 hybrid | 26.978701 |  | 0.964839 |  |
| Fixed S03_V04 | 27.505364 | 0.526663 | 0.975152 | 0.010314 |
| Per-page global oracle | 28.253898 | 0.748534 | 0.977777 | 0.002624 |
| G128 hard oracle | 28.980910 | 0.727012 | 0.982373 | 0.004596 |
| G128 hybrid oracle | 29.050698 | 0.069788 | 0.982121 | -0.000252 |

## S19 chain at the frozen-five best is in the oracle table. Chain at V04:

| Stage | PSNR | ΔPSNR vs previous | SSIM | ΔSSIM vs previous |
|---|---:|---:|---:|---:|
| Current6 hybrid | 26.978701 |  | 0.964839 |  |
| Fixed S19_V04 | 27.952153 | 0.973452 | 0.975894 | 0.011055 |
| Per-page global oracle | 28.251067 | 0.298915 | 0.976449 | 0.000556 |
| G128 hard oracle | 30.346166 | 2.095099 | 0.984381 | 0.007931 |
| G128 hybrid oracle | 30.413202 | 0.067036 | 0.983924 | -0.000456 |
