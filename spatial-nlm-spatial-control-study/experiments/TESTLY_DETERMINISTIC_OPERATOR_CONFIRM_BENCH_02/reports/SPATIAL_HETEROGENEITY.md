# Spatial heterogeneity

Area fractions are inside one family. Entropy is in bits. A high entropy says conditional capacity exists. It does not say a network can learn the map.

- holdout21 S03: mean dominant fraction 0.857, entropy 0.824 bits, presets over 5% area 1.71, spatial headroom 1.254 dB.
- holdout21 S05: mean dominant fraction 0.769, entropy 1.150 bits, presets over 5% area 2.57, spatial headroom 1.028 dB.
- holdout21 S19: mean dominant fraction 0.477, entropy 1.996 bits, presets over 5% area 5.00, spatial headroom 2.461 dB.
- development123 S03: mean dominant fraction 0.818, entropy 0.912 bits, presets over 5% area 1.87, spatial headroom 1.084 dB.
- development123 S05: mean dominant fraction 0.707, entropy 1.263 bits, presets over 5% area 2.86, spatial headroom 0.939 dB.
- development123 S19: mean dominant fraction 0.453, entropy 2.032 bits, presets over 5% area 4.96, spatial headroom 2.449 dB.

## Spatial s after a frozen tone

Each row applies one frozen tone to all six Current6 scales, then composes them. Fixed is s=31 plus that tone.

| Split | Tone | Fixed s=31 PSNR | Modal page-level s | Page-level s oracle PSNR | Spatial hybrid | Spatial minus fixed |
|---|---|---:|---:|---:|---:|---:|
| holdout21 | S05_V03 | 28.382719 | 91 | 28.572870 | 28.717230 | 0.334511 |
| holdout21 | S05_V05 | 28.537279 | 91 | 28.846844 | 28.896395 | 0.359116 |
| holdout21 | S03_V04 | 27.505364 | 91 | 27.851058 | 27.862393 | 0.357030 |
| holdout21 | S03_V05 | 26.882614 | 91 | 27.217454 | 27.217642 | 0.335028 |
| development123 | S05_V03 | 27.593155 | 91 | 27.810312 | 27.928982 | 0.335827 |
| development123 | S05_V05 | 27.365478 | 91 | 27.630045 | 27.672124 | 0.306646 |
| development123 | S03_V04 | 26.571660 | 91 | 26.860745 | 26.868906 | 0.297246 |
| development123 | S03_V05 | 25.943385 | 91 | 26.211703 | 26.218514 | 0.275129 |
