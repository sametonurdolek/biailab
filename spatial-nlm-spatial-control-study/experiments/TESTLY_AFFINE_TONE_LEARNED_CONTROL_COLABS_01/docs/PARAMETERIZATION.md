# Parameterization

## Frozen photometry

For a canonical grayscale page `x` in `[0, 1]`:

```text
B0 = close(x, 31×31 ones, reflect-101)
B  = Gaussian(B0, ksize 0, sigma 31/6, reflect-101)
Z  = clip(x / max(B, 0.1), 0, 1)   float32
```

On valid inputs this `Z` does not go below `x` except for float error around `1e-7`. The tone stage is what can darken a pixel. The floor `0.1` is part of the frozen operator; it is not tuned here.

## Black / white

Raw channels `(u, v)` go through sigmoids. With the fallback box and `min_span = 0.05`:

```text
black = 0 + (0.30 - 0) * sigmoid(u)
low   = max(0.70 - black, 0.05)
high  = max(1 - black, low)
span  = low + (high - low) * sigmoid(v)
white = black + span
```

This keeps `0 ≤ black`, `black + 0.05 ≤ white`, and `white ≤ 1`. It also keeps `white ≥ 0.70` inside this box. `min_span` is a safety floor so the denominator cannot collapse. It is not an estimated optimum.

Deployment:

```text
Y = clip((Z - black) / (white - black), 0, 1)
```

Spatial maps are bilinearly resized to the canonical page before that formula. A page-level model broadcasts the two scalars.

## Gain / bias

```text
a = 1 / (white - black)
c = -black / (white - black)
Y = clip(aZ + c, 0, 1)
```

EXP03 does not go through the endpoint sigmoid. It sigmoids directly into `a ∈ [1, 2.5]` and `c ∈ [-0.75, 0]`. Those limits are the smallest axis-aligned rectangle that contains the corners of the fallback box. The most negative bias in the box is `-0.30 / 0.40 = -0.75`. The interiors are not identical: a gain/bias pair inside the rectangle can correspond to an endpoint pair outside the black/white box, and the reverse is also true. S05 V03 (`a ≈ 1.219512`, `c ≈ -0.121951`) and S05 V05 (`a ≈ 1.666667`, `c ≈ -0.366667`) sit in both.

## Frozen bank

| Preset | black | white | gain | bias |
|---|---:|---:|---:|---:|
| V01 | 0.02 | 0.98 | 1.041667 | -0.020833 |
| V02 | 0.05 | 0.95 | 1.111111 | -0.055556 |
| V03 | 0.10 | 0.92 | 1.219512 | -0.121951 |
| V04 | 0.15 | 0.88 | 1.369863 | -0.205479 |
| V05 | 0.22 | 0.82 | 1.666667 | -0.366667 |

V03 is the fixed baseline because it was a predeclared BENCH_02 gate candidate and it confirmed. V04 scored higher on the development pages and was not a gate candidate. These scripts do not promote it.

Discrete deployment expands the argmax labels with the rounded G128 edges. Interior pixels copy one preset. A 5×5 dilation of the label change is blended with a dirty-guided 9×9 box filter, radius 4, epsilon `0.001`, reflect-101. The clean page is not an input to that blend.

## Scalar manifold

`t` is a sigmoid in `[0, 1]`. `interp_manifold` linearly interpolates knots that must already satisfy `0 ≤ black < white ≤ 1`. The loader accepts a file only when `status` is `measured_from_bench03`. No such file is shipped.

## Illegal maps

The boxes above are the whole legal set for training and export. A target-fit gain near 1.46 with bias near -0.33, from an earlier diagnostic, is not a preset and is not a bound.
