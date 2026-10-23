# S05 failure regimes

Descriptors are computed on the dirty page only. The threshold for a Q03-like page is the dirty mean of discovery page Q03.

Q03 dirty mean: 0.763117

## Correlation with S05 V05 ΔPSNR

| Descriptor | Pearson | Spearman |
|---|---:|---:|
| mean | -0.298257 | -0.243838 |
| std | 0.131236 | 0.186222 |
| p95 | -0.113637 | -0.106770 |
| p99 | -0.101687 | -0.096149 |
| frac_below_0_2 | 0.333839 | 0.354174 |
| frac_above_0_9 | 0.037884 | -0.055675 |
| low_frequency_variance | 0.103818 | 0.293863 |
| low_frequency_range | 0.183418 | 0.293966 |

Pages at least as bright as Q03: 0


## Severe pages (ΔPSNR < -1 dB or ΔSSIM < -0.05)

- binarizeddataset::0000040::5f4f72ac25: ΔPSNR -1.029, ΔSSIM +0.0076, dirty mean 0.521, p95 0.576, low-frequency range 0.108
- binarizeddataset::0000021::1b4334b134: ΔPSNR -2.716, ΔSSIM +0.0053, dirty mean 0.508, p95 0.557, low-frequency range 0.093
- binarizeddataset::0000395::d7cdace0f7: ΔPSNR -2.097, ΔSSIM -0.0139, dirty mean 0.557, p95 0.639, low-frequency range 0.118
- binarizeddataset::0000165::9fef5af8b8: ΔPSNR -3.040, ΔSSIM +0.0049, dirty mean 0.572, p95 0.678, low-frequency range 0.163
- acceptednewnewbinarizelast::0000687::05af53ab80: ΔPSNR -3.261, ΔSSIM -0.0031, dirty mean 0.611, p95 0.678, low-frequency range 0.146
- acceptednewnewbinarizelast::0000358::db57ad65d6: ΔPSNR -4.126, ΔSSIM -0.0353, dirty mean 0.554, p95 0.627, low-frequency range 0.101
- acceptednewnewbinarizelast::0000356::12c46adb8b: ΔPSNR -3.485, ΔSSIM -0.0229, dirty mean 0.530, p95 0.596, low-frequency range 0.127
- binarizeddataset::0000640::75ffc7a29f: ΔPSNR -1.550, ΔSSIM +0.0641, dirty mean 0.625, p95 0.904, low-frequency range 0.556
- acceptednewnewbinarize::0089::17ea5bd7c2: ΔPSNR -1.259, ΔSSIM +0.0388, dirty mean 0.623, p95 0.809, low-frequency range 0.311
- acceptednewnewbinarize::0091::3d3bf7d256: ΔPSNR -1.269, ΔSSIM -0.0182, dirty mean 0.737, p95 0.910, low-frequency range 0.317

This is a diagnosis of the frozen V05 preset. The preset was not edited.
