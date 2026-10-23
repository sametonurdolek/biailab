# S03 and S05 equivalence

Code-derived identity, from the imported operators:

```
S05: Y = clip((Z - black) / (white - black))
S03: Y = clip(gain * Z + bias)
S04: Y = clip((Z - black) / (1 - black))
Z = canonical closing-division at s=31
```

So `(Z-black)/(white-black) = gain * Z + bias` with

`gain = 1/(white-black)` and `bias = -black/(white-black)`.

Every frozen S05 preset is an affine map. S04 is the same affine with the white endpoint fixed at 1, which forces `gain + bias = 1`. S03 is the same output equation without that endpoint constraint. The five S03 points and the five S05 points are not the same numeric pairs.

| Preset | black | white | gain | bias |
|---|---:|---:|---:|---:|
| S05 V01 | 0.02 | 0.98 | 1.041667 | -0.020833 |
| S05 V02 | 0.05 | 0.95 | 1.111111 | -0.055556 |
| S05 V03 | 0.10 | 0.92 | 1.219512 | -0.121951 |
| S05 V04 | 0.15 | 0.88 | 1.369863 | -0.205479 |
| S05 V05 | 0.22 | 0.82 | 1.666667 | -0.366667 |
| S04 V01 | 0.02 | 1.00 | 1.020408 | -0.020408 |
| S04 V02 | 0.05 | 1.00 | 1.052632 | -0.052632 |
| S04 V03 | 0.10 | 1.00 | 1.111111 | -0.111111 |
| S04 V04 | 0.18 | 1.00 | 1.219512 | -0.219512 |
| S04 V05 | 0.28 | 1.00 | 1.388889 | -0.388889 |
| S03 V01 | 0.045455 | 0.954545 | 1.10 | -0.05 |
| S03 V02 | 0.096000 | 0.896000 | 1.25 | -0.12 |
| S03 V03 | 0.151724 | 0.841379 | 1.45 | -0.22 |
| S03 V04 | 0.205882 | 0.794118 | 1.70 | -0.35 |
| S03 V05 | 0.250000 | 0.750000 | 2.00 | -0.50 |

S03's implied white point is the Z value that the unclipped line sends to 1. Several S03 presets place that point above 1, so paper at Z=1 clips to white while the slope and intercept still differ from every S05 preset.

No new affine point was added. A denser gain/bias grid would be BENCH_03.
