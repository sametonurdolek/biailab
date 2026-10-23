# S05

| Candidate | PSNR | ΔPSNR | CI lo | SSIM | ΔSSIM | PSNR wins | Severe frac | Structural |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S05_V01 | 26.286963 | 0.200455 | 0.116696 | 0.954037 | 0.006504 | 96/123 | 0.008 | PASS |
| S05_V02 | 26.785254 | 0.698746 | 0.540627 | 0.965039 | 0.017506 | 109/123 | 0.008 | PASS |
| S05_V03 | 27.593155 | 1.506646 | 1.220047 | 0.971745 | 0.024212 | 116/123 | 0.008 | PASS |
| S05_V04 | 27.826189 | 1.739681 | 1.395118 | 0.974728 | 0.027195 | 107/123 | 0.024 | PASS |
| S05_V05 | 27.365478 | 1.278969 | 0.878069 | 0.972612 | 0.025079 | 86/123 | 0.081 | PASS |

| Stage | PSNR | ΔPSNR vs previous | SSIM | ΔSSIM vs previous |
|---|---:|---:|---:|---:|
| Current6 hybrid | 26.086509 |  | 0.947533 |  |
| Fixed S05_V05 | 27.365478 | 1.278969 | 0.972612 | 0.025079 |
| Per-page global oracle | 28.240600 | 0.875122 | 0.975417 | 0.002805 |
| G128 hard oracle | 28.697441 | 0.456840 | 0.978672 | 0.003254 |
| G128 hybrid oracle | 28.765459 | 0.068018 | 0.978669 | -0.000002 |

| Stage | PSNR | ΔPSNR vs previous | SSIM | ΔSSIM vs previous |
|---|---:|---:|---:|---:|
| Current6 hybrid | 26.086509 |  | 0.947533 |  |
| Fixed S05_V03 | 27.593155 | 1.506646 | 0.971745 | 0.024212 |
| Per-page global oracle | 28.240600 | 0.647446 | 0.975417 | 0.003673 |
| G128 hard oracle | 28.697441 | 0.456840 | 0.978672 | 0.003254 |
| G128 hybrid oracle | 28.765459 | 0.068018 | 0.978669 | -0.000002 |

Below-region MSE, eps=1/255, mean over pages. This is the squared error on pixels where the clean target is darker than the dirty page by more than 1/255.

- g128_hybrid: 0.068720
- fixed31: 0.069929
- S04_V04: 0.027134
- S05_V03: 0.056124
- S05_V05: 0.046282
