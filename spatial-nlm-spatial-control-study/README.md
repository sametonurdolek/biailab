# Spatial control of classical Non-Local Means

Handoff for the next study. This folder does not contain a new experiment.
It records why the work exists, what the first experiment actually did, which
numbers are finished, and which numbers are not.

The preserved baseline lives at `C:\testly_external_workspace\spatial-nlm`.
That directory is evidence. Do not modify, delete, rename, overwrite, or
reorganize it. Copies used for reading sit in `baseline_snapshot/` and
`baseline_results/`. Future code goes under `experiments/`.

Status of the science, as of the artifacts copied on 2026-10-07:

- Stationary AWGN test: finished. One seed. Numbers below come from
  `baseline_results/metrics.json`.
- Spatially varying noise test: **incomplete. Not claimable.**
- No biomedical dataset has been run. This is not a medical method.

Source code is authoritative for constants. Where the old README, a log, and
the current source disagree, the discrepancy is listed in
[Inconsistencies](#inconsistencies-between-prose-logs-and-source) and the
code wins for anything that would be run next.

---

## 1. Project purpose

Classical algorithms often have a small number of meaningful control
parameters. The broader idea is to let a network learn only that control,
and to leave the classical mathematical operator unchanged.

The concrete test case is Non-Local Means (Buades, Coll, and Morel, CVPR
2005). Classical NLM uses a bandwidth `h`. This study asks whether a network
can learn a spatially varying bandwidth `h(x, y)` that improves NLM while the
estimator itself stays the Buades estimator.

The network does not output a restored image. It outputs `h`, or a map
`h(x, y)`. Every denoised image in the comparison is produced by the same NLM
operator. A constant `h` is that operator with a constant map.

The long-term reason to care is the paradigm: learned control plus a
classical solver. The present study is a controlled natural-image
proof of mechanism on Kodak photographs. It is the evidence one would bring
to a conversation about whether the paradigm is worth taking into biomedical
imaging research. It is not itself that biomedical study.

## 2. Personal motivation

I am doing this before approaching an AI / biomedical imaging research lab.
The aim is to arrive with a finished, rerunnable, honest experiment rather
than with an intention. Respect and a referral, if they come, should come
from the controls and from a result that was allowed to be null.

That imposes a tone on the work:

- The question is narrow enough to answer.
- The operator is shared by every method, so a win cannot be "a different
  denoiser."
- The comparison that matters is scalar `h` versus spatial `h(x, y)`, not
  merely spatial versus an untuned constant.
- An incomplete run stays incomplete in the write-up.
- Kodak gray images plus synthetic noise are not a clinical claim.

## 3. Scientific hypothesis

Classical NLM uses one fixed `h`. The hypothesis is that the useful bandwidth
varies across the image, and that a network can predict that field well
enough to improve the classical estimator without replacing its weights.

What would support the hypothesis, on a given noise protocol:

- the spatial network beats the scalar network;
- both used the same operator, the same allowed range of `h`, the same trunk,
  the same loss, and the same step budget;
- the scalar network does not itself explain the gain by finding a better
  single `h`;
- a hand-written spatial rule does not already explain the gain.

What would weaken it:

- spatial and scalar land on the same score;
- spatial loses to the validation-chosen constant `h`;
- the gain appears only because the map's histogram of values is useful, while
  the assignment of those values to locations is not (the shuffled-map
  control, not yet run);
- the result depends on one seed or on one noise family.

The stationary-noise case is the sharper version of the hypothesis. There the
true noise scale does not vary in space. If a spatial map still helps, the
network may be responding to image structure rather than to local noise
amplitude. That reading is a hypothesis with one supporting comparison and
one correlation. It is not yet a demonstrated mechanism.

## 4. Classical NLM formulation

Implemented in `baseline_snapshot/nlm.py`. Images are linear values in
`[0, 1]`. For grayscale Kodak, there is one channel.

```text
D(i, j) = mean over the patch and channels of (u(i+k) - u(j+k))^2
w(i, j) = exp( - D(i, j) / h(i)^2 )
restored(i) = sum_j w(i, j) u(j) / sum_j w(i, j)
```

`j` runs over an odd search window. `h` depends only on the reference pixel
`i`. `h^2` is clamped below by `1e-12` so the denominator of the exponential
cannot be zero. Weights are positive and sum to one, so each restored pixel
stays inside the convex hull of its search window.

Geometry used for every scored method:

| Piece | Value in code |
|---|---|
| Patch | 7 (odd) |
| Search | 21 (odd) |
| Distance | mean squared difference over the patch (`sum of squares / patch_elems`) |
| Padding | `reflect`, by `search_radius + patch_radius` on each side |
| Bandwidth | `h(i)` at the reference pixel only |

Fixed `h`, the hand-crafted map, the scalar network, and the spatial network
all call this function. There is no second NLM. OpenCV
`fastNlMeansDenoising` appears in the driver only as a side measurement on
one validation image, on the uint8 scale. Those OpenCV `h` values are not on
the same scale as `h` in `[0, 1]` and do not belong in the main table.

The patch distance does not depend on `h` or on the network. The
implementation computes it under `torch.no_grad()`. The gradient of the
restoration with respect to `h` is unchanged by that choice. The neighbor
sum is accumulated in groups so that one allocation stays near a byte budget.
The current source budget is 64 MiB. The completed AWGN run used a 256 MiB
budget. The formula is the same sum; the grouping of the sum changed after
that run. See [Inconsistencies](#inconsistencies-between-prose-logs-and-source).

A direct implementation check lives in `nlm.py` (`_self_test`). On 2026-10-07
it reported max absolute error `1.19e-07` against the direct formula, including
after the budget change. That check uses a tiny image. It does not re-score
Kodak.

## 5. What is learned, what is fixed, what is chosen on validation

### Fixed before any score

These are not fit to the test set. They are written as constants.

- The NLM formula, patch 7, search 21, reflect padding.
- Kodak split: `kodim01`–`kodim12` train, `kodim13`–`kodim16` validation,
  `kodim17`–`kodim24` test.
- Noise families and their sigma values.
- Crop 128, batch 4, Adam, learning rate `3e-3`, L1 loss, 400 steps,
  validation every 50 steps, gradient clip 1.0, one training seed index
  `SEEDS = (0,)`.
- H-Net width 32, depth 4, residual blocks, head initialization
  `Normal(0, 0.2)`.
- The `h` sweep grid and the `k` grid.
- Test realization count 2, and the NumPy seeds that draw validation and test
  noise.

### Learned

Only the H-Net weights. The scalar model and the spatial model are separate
fits of the same class. They do not share weights. The state dict does not
store the `spatial` flag; loading a checkpoint into the wrong flag is a silent
error because the parameter shapes match. `awgn_scalar.pt` must be loaded with
`HNet(spatial=False)`. `awgn_spatial.pt` must be loaded with
`HNet(spatial=True)`.

The network input is the noisy grayscale image, one channel. The output is
bandwidth, mapped by a sigmoid into `[h_lo, h_hi]`:

```text
h = h_lo + (h_hi - h_lo) * sigmoid(raw)
```

On both protocols that interval stayed `[0.02, 0.50]`, because the best grid
point was interior. The network cannot win by leaving the range the constant
baseline was allowed to use.

### Chosen on validation, then frozen

- `h*`: one constant, argmax of validation PSNR on the `h` grid.
- `k*` for the hand rule, and a separate `k*` for the true-sigma oracle.
- The training checkpoint: highest validation PSNR, not the last step and not
  anything on the test set.
- The decision not to extend the `h` grid. Extension happens only if the best
  grid point is an endpoint. It did not fire for either protocol.

The test split is not used to choose `h`, `k`, the checkpoint, the step
count, the architecture, or the learning rate. The kodim17 figure is drawn
after those choices. It is a picture, not a score, and it must not be used to
change them.

## 6. Dataset and noise protocols

Kodak lossless PNGs, loaded as grayscale (`PIL` mode `L`) and divided by 255.
They were not copied into this study folder. Read them from
`C:\testly_external_workspace\spatial-nlm\data\kodak\kodimNN.png` and do not
modify that directory. There are 24 files, `kodim01.png` through `kodim24.png`.

| Split | Files | Role |
|---|---|---|
| Train | kodim01–12 | random 128 crops only |
| Validation | kodim13–16 | sweeps, checkpoint selection |
| Test | kodim17–24 | scored once, after choices are frozen |

Noise is added in NumPy and clipped to `[0, 1]`. Clipping slightly violates
the stated sigma near saturated pixels. Keep the clip; report it. Do not drop
it in order to match a textbook sigma.

### Stationary AWGN

`sigma = 25/255` at every pixel (`SIGMA_AWGN` in `data.py`).

### Spatially varying noise

Sigma is a linear ramp from `15/255` to `45/255`. For every crop and every
full-image realization, the axis (horizontal or vertical) and the direction
are drawn at random. The network cannot solve the task by memorizing a fixed
coordinate such as "the right half is noisier."

The kodim17 picture, if the spatial protocol ever finishes, is specified
separately as a left-to-right ramp so the figure is readable. That draw is
not a scored test realization. It does not exist yet.
`figure_spatial_kodim17.png` is absent from the baseline, and it must not be
invented.

### Frozen noise

Validation noise is drawn once (`realizations=1`) and reused for every sweep
and for checkpoint selection.

| Protocol | Validation seed | Test seed | Test realizations |
|---|---:|---:|---:|
| `awgn` | 1000 | 3000 | 2 |
| `spatial` | 2000 | 4000 | 2 |

Training noise is new at every step. Crops are random, with independent
horizontal and vertical flips at probability 1/2.

Training seeds, with `SEEDS[0] = 0`:

| Run | Seed |
|---|---:|
| AWGN scalar | 10 |
| AWGN spatial | 11 |
| Spatial-noise scalar | 20 |
| Spatial-noise spatial | 21 |

Formula in `run_experiment.py`:
`(10 if awgn else 20) + (1 if spatial else 0) + 100 * SEEDS[0]`.

## 7. Controls, and the alternative each one removes

Deployable comparisons and oracles are different objects. Oracles see
information a method is not allowed to see. They are ceilings or diagnostics.

| Name | What is free | Sees the clean image at test time? | What a win or a loss means |
|---|---|---|---|
| Noisy | nothing | no | the unprocessed observation |
| Fixed `h*` | one scalar, chosen on validation | no | a tuned classical baseline, not the library default |
| Hand-crafted map | `h(i) = clip(k * local_sigma(noisy), h_lo, h_hi)`, `k` chosen on validation | no | "any spatial map would have done this" is false if this loses while the learned map wins |
| Scalar H-Net | one `h` per image, from the pooled trunk | no | the central neural control. Same trunk, same loss, same range, one number instead of a map |
| Spatial H-Net | `h(x, y)` from the same trunk without pooling | no | the hypothesis |
| Oracle constant `h` | best constant on the sweep grid for that observation | yes, it uses the clean target to pick `h` | upper bound on constant-`h` NLM. Not a method |
| Oracle true sigma | `clip(k * the sigma field that generated the noise)`, `k` from validation | it sees the noise field, not by denoising | ceiling of a linear rule in the true sigma. Not a method |

### Fixed `h`

`h` grid is `np.geomspace(0.02, 0.50, 12)`. If the best point is an endpoint
the grid is extended once, by a half-decade of five geometric steps, and the
sweep is repeated. Both protocols landed inside `(0.02, 0.50)`, so the logged
bounds stayed `[0.0200, 0.5000]`.

AWGN validation chose `h* = 0.08638876637025905`. The log prints `0.0864`.
Validation PSNR at that point is 27.205. Neighbors on the same grid: 0.0645
gives 25.238, 0.1158 gives 26.060. The optimum is interior.

Spatial-noise validation chose `h* = 0.1158` (the grid value
`0.115789...` is printed as `0.1158`). Validation PSNR 25.667. Neighbors:
0.0864 gives 23.566, 0.1551 gives 24.666. Also interior. This sweep was
reproduced by the later interrupted run. It is a validation fact. It is not a
test result.

### Hand-crafted map

Immerkaer local sigma. The image Laplacian uses the kernel

```text
[[ 1, -2,  1],
 [-2,  4, -2],
 [ 1, -2,  1]]
```

with reflect padding. Sigma is the 21×21 average of the absolute Laplacian,
times `sqrt(pi/2) / 6`. Texture leaks into this estimate: edges and fine
detail raise the local "noise" even when the noise is stationary. That is a
property of the control, and it is why the control can lose to a constant.

`k` grid: `0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8`.

| Protocol | Chosen `k` | Validation PSNR | Fixed-`h` validation PSNR |
|---|---:|---:|---:|
| AWGN | 0.75 | 26.742 | 27.205 |
| Spatial noise | 0.75 | 25.902 | 25.667 |

On stationary noise the hand rule is worse than the tuned constant. On
spatially varying noise it is better than the tuned constant, on validation
only. The spatial-noise number is not a test score.

### Scalar and spatial H-Net

`baseline_snapshot/models.py`.

- Stem: `Conv2d(1, 32, kernel_size=3, padding=1)`.
- Four residual blocks. Each block is two 3×3 convolutions:
  `x + conv2(ReLU(conv1(ReLU(x))))`.
- Head: `Conv2d(32, 1, kernel_size=1)`.
- Scalar path: global average pool over height and width, then the head, then
  expand to the image size. The map is constant by construction.
- Spatial path: the head runs at every pixel.
- Parameter count of the spatial module, logged by the driver and confirmed by
  loading a checkpoint: **74337**. The scalar module has the same count. The
  difference is the forward pooling, not the number of weights.
- Theoretical receptive field from the kernel sizes: stem 3, then four blocks
  of two 3×3 convolutions, so 19 pixels. This was not measured with an
  occlusion test.
- Head weight initialized `Normal(0, 0.2)`, head bias then set so that a flat
  map sits at `h*`. Four extra bias updates on `torch.rand(4, 1, 128, 128)`
  pull the mean predicted `h` back to `h*` without touching the head weight.
  Exact-zero head weights are rejected by the code comments: the bandwidth
  gradient passes through that layer, and a dead trunk cannot test the
  hypothesis.
- At step 0 the validation PSNR must lie within 0.3 dB of fixed-`h*`
  validation PSNR. The completed AWGN inits were inside that band
  (scalar 27.143 vs 27.205, spatial 27.045 vs 27.205).

A discarded earlier trunk, a plain stack of ReLU convolutions with a zeroed
head, is described in the `models.py` comment. Its activations shrank and the
bandwidth did not move. That run's log is **not** among the preserved
artifacts. It is not a scientific result, and it must not be cited as
evidence against spatial `h`.

### Oracles

True-sigma `k` on AWGN validation: `k = 1.0`, validation PSNR 26.839, which
is worse than fixed `h*` at 27.205. Sigma is constant, so this oracle is
another constant, and `k * sigma = 25/255 ≈ 0.0980`, which is not the grid
point `h*`. The gap is expected from the grid, not a bug in NLM.

True-sigma `k` on spatial-noise validation: `k = 0.75`, validation PSNR
26.035, which is 0.368 dB above fixed `h*` on that same validation draw.
Still not a test score. Still not a method.

Oracle constant-`h` is computed on the **test** observations after training,
by sweeping the same grid with the clean image visible. On the AWGN test set
its mean PSNR equals the fixed-`h` mean exactly
(`28.03282642364502`). The validation-chosen `h*` was already the per-image
optimum of that grid under stationary noise. The driver records PSNR for this
oracle and does not record SSIM.

### OpenCV side note

One validation image only. Not the four-image validation mean, and not the
eight-image test mean.

| Protocol | OpenCV uint8 `h` | OpenCV PSNR | Our fixed `h*` on that same image |
|---|---:|---:|---:|
| AWGN, first validation image | 20 | 23.834 | 24.046 |
| Spatial noise, first validation image | 30 | 22.340 | 22.393 |

Do not mix these with the main table. A single image at ~24 dB is not in
conflict with a four-image validation mean of 27.205 or an eight-image test
mean of 28.033.

## 8. Why scalar versus spatial is the central comparison

"Spatial beats fixed `h`" has a boring explanation: the network found a
better single number, and the spatial freedom was unused. The scalar network
is there to remove that explanation. It has the same stem, the same residual
blocks, the same head class, the same initialization around `h*`, the same
sigmoid bounds, the same L1 loss, the same optimizer, the same step budget,
and the same NLM. It is trained separately. The designed difference is
pooling.

Read a result in this order:

1. If scalar ties fixed `h`, the network did not help by picking a better
   global bandwidth.
2. If spatial then beats scalar, the extra score tracks the right to vary `h`
   inside the image.
3. If the hand-crafted spatial rule does not capture that extra score, a
   generic "make `h` spatial" story is not enough either.
4. If oracle constant-`h` ties fixed `h`, there was no better constant on the
   grid for the test observations, so the spatial gain is not an oracle
   constant in disguise.

That chain is what the AWGN test currently supports. It does not, by itself,
say that spatial NLM is a better general denoiser, that the same gap will
appear under spatially varying noise, or that the map has been shown to mean
"edges" or "texture."

## 9. Confirmed evidence: stationary AWGN test

Artifact: `baseline_results/metrics.json`, key `protocols.awgn`.
The same bytes are in `metrics_awgn.json` (SHA256
`B9C5665AE3E75DD3BB8985DFBDEDB68DCB8B107DB711D85A535669B3EA550728`).

Eight test images, two frozen realizations each. The table is the mean across
images. Each image's score is already the mean of its two realizations. The
standard deviation is the population standard deviation across the eight
images (`ddof=0`), not a standard error and not a confidence interval. PSNR
uses peak 1. SSIM is the Wang form: 11×11 Gaussian with sigma 1.5,
`C1 = 0.01^2`, `C2 = 0.03^2`, then the mean over the image.

| Method | PSNR mean | PSNR std | SSIM mean | SSIM std |
|---|---:|---:|---:|---:|
| Noisy | 20.491 | 0.409 | 0.323 | 0.058 |
| Fixed `h*` | 28.033 | 1.557 | 0.766 | 0.059 |
| Hand-crafted | 27.442 | 1.316 | 0.731 | 0.038 |
| Oracle constant `h` | 28.033 | 1.557 | not recorded |  |
| Oracle true sigma | 27.725 | 1.671 | 0.753 | 0.070 |
| Scalar H-Net | 28.036 | 1.547 | 0.767 | 0.057 |
| Spatial H-Net | 28.314 | 1.554 | 0.774 | 0.055 |

Full precision is in the JSON. Rounded gaps on the means:

- spatial minus fixed = **+0.281 dB**
- spatial minus scalar = **+0.279 dB**
- scalar minus fixed = **+0.003 dB**
- hand minus fixed = **−0.591 dB**
- oracle true sigma minus fixed = **−0.308 dB**

SSIM moves in the same direction by a small amount (spatial 0.774 vs scalar
0.767 vs fixed 0.766 vs hand 0.731).

### How to read this, and how not to

This is evidence, on this protocol and this seed, that spatial control did
something a single learned `h` did not. Scalar tying fixed `h` is the piece
that makes the +0.28 dB about the map rather than about "a neural net found
`h*`." The hand rule moving the wrong way says the gain is not "spatial
variation of any kind." Oracle constant-`h` tying fixed `h` says the test
images did not want a different constant on the swept grid.

It is not a proof of a universally better NLM. The spatial validation curve
was still rising at the precommitted cutoff (step 200 → 27.259, 250 → 27.265,
300 → 27.359, 350 → 27.424, 400 → 27.496). The tested checkpoint is step 400,
which is the best validation point inside the budget and also the last point.
A longer budget might score higher. Raising the budget after seeing the test
number would be tuning on the test set. Do not do it unless every learned row
is rerun under the new budget and the test set is scored again from scratch.

The scalar checkpoint is not step 400. Its best validation PSNR is step 300
(27.217). Step 400 was 27.202. `awgn_scalar.pt` matches the step-300 log:
mean absolute head weight 0.18987. `awgn_spatial.pt` matches step 400:
mean absolute head weight 0.16578.

Step 1 of training drops validation PSNR hard (AWGN spatial step 1 was
20.857, with `h_mean` jumping to about 0.043). By step 50 it has returned to
the fixed-`h` neighborhood. That dip is the `3e-3` learning rate overshooting.
The saved model is the best later checkpoint, not step 1.

### Mechanism numbers recorded with the AWGN test

These are means of per-observation statistics on the test realizations, from
`protocols.awgn.mechanism`.

| | Scalar | Spatial |
|---|---:|---:|
| Mean `h` | 0.0858 | 0.0846 |
| Mean within-image std of `h` | ~0 (1.1e-8) | 0.00685 |
| Min `h` across observations | 0.0849 | 0.0517 |
| Max `h` across observations | 0.0881 | 0.169 |
| Pearson with true sigma | null | ~0 (stored −9.7e-8) |
| Pearson with noisy gradient magnitude | ~0 (stored −3.0e-9) | **0.385** |

Gradient magnitude is the forward difference used in `grad_mag`: vertical and
horizontal first differences, padded with zeros on the last row and column.
The 0.385 is a real association between the learned map and the gradient of
the noisy image. The correlations against true sigma are not interpretable
here, because the true sigma field is constant; a constant field has no
Pearson correlation. The tiny stored values are numerical residue or null.
They are not evidence that the map tracks noise amplitude.

The within-image standard deviation 0.00685 is small next to the gap between
grid points (0.086 versus 0.116). The map is a mild modulation around `h*`,
not a swing across the whole legal range. The legal range is still `[0.02,
0.50]`, and individual pixels did reach about 0.052 and 0.169.

### The picture

`baseline_results/figure_awgn_kodim17.png` (SHA256
`61F04AB2EF2160EDDEC359008A6639F842E5CB9F4001C2A3701AC8E2804CC2B9`).

kodim17 is a test image. The noise in the figure is a separate draw
(`numpy` seed 18 through `realize`), not one of the two scored realizations
(seed 3000). Top row: clean, noisy, fixed `h`, hand, scalar, spatial. Bottom
row, one shared color scale: blank, `k * true sigma`, the constant `h*`,
`k * local sigma`, the scalar `h`, the learned `h(x, y)`.

Qualitatively, the learned map rises on the statue's face and on the textured
ball and stays low on the flat stone. The hand-crafted map is speckled. The
scalar map and the constant are flat. True sigma is flat at a different level
(`k * 25/255` versus `h*`). This is one image. It is consistent with the
gradient correlation. It is not a second test score, and it is not a reason
to change the model.

## 10. Spatially varying noise: incomplete, not claimable

**Do not report a spatial-noise test PSNR, SSIM, or mechanism correlation.
None was written.**

`metrics.json` has no `protocols.spatial` block. There is no
`figure_spatial_kodim17.png`. There is no `spatial_spatial.pt`.

What does exist, and what it is allowed to mean:

### Validation sweeps, reproduced

Both `run.log` and the later `run_spatial.log` record the same choices:
`h* = 0.1158`, hand `k* = 0.75` at validation PSNR 25.902, oracle-sigma
`k* = 0.75` at validation PSNR 26.035, fixed-`h` validation PSNR 25.667.
OpenCV on the first validation image is the side note in section 7. These
are frozen-validation facts. They can be reused as the definition of the
protocol. They cannot be promoted to a test result.

### Scalar training finished once, then the weights were overwritten

`run.log` (SHA256
`7365969112F57CCD33430F23B51C243AD43F356BBB01378A3960AF6D5D255780`, identical
to `run_awgn_preserved.log`) records a full 400-step scalar run, seed 20.

- Init validation 25.670 versus fixed 25.667.
- Step 1 validation 18.961, the usual overshoot.
- From step 50 through step 400, validation PSNR stays between 25.566 and
  25.668. Within-image `h` std stays 0.
- Best logged validation point is step 100 at 25.668, which is the constant
  `h*` for practical purposes. Step 400 is 25.566 with `h_mean` 0.1207.

That curve says the scalar model, given a spatial ramp, did not move off a
constant near `h*`. It is a validation curve. The test set was never scored
with this model.

The file `spatial_scalar.pt` is **not** that 400-step model. Its timestamp
matches the end of `run_spatial.log` (2026-10-07 13:32:38). Its mean absolute
head weight is 0.148868665, which matches the logged `head_w=0.14887` at
**step 1** of the restarted scalar run, not the step-400 value 0.12476.
The restart saves a checkpoint whenever validation PSNR beats the running
best, and the running best starts at `-1e9`, so step 1 overwrote the finished
file. Do not load `spatial_scalar.pt` as a trained scalar baseline. The
finished scalar weights are gone. The curve in `run.log` is the record.

### Spatial training died during the first attempt

Same log, seed 21, after the scalar run:

| Step | Validation PSNR | `h` mean | `h` std |
|---:|---:|---:|---:|
| init | 25.490 |  |  |
| 1 | 18.894 | 0.0396 | 0.0097 |
| 50 | 25.604 | 0.1180 | 0.0035 |
| 100 | 25.745 | 0.1134 | 0.0062 |
| 150 | 25.729 | 0.0884 | 0.0125 |
| 200 | 26.195 | 0.0973 | 0.0225 |
| 250 | 26.299 | 0.0975 | 0.0234 |
| 300 | 26.341 | 0.0943 | 0.0214 |

At step 300 the validation mean was about +0.674 dB over fixed `h*` (25.667)
and about +0.306 dB over the true-sigma oracle's validation score (26.035),
with within-image std near 0.02. The curve was still moving. Then the process
died:

```text
FAILED OutOfMemoryError: CUDA out of memory. Tried to allocate 246.00 MiB.
GPU 0 has a total capacity of 7.96 GiB of which 6.00 GiB is free.
Of the allocated memory 610.09 MiB is allocated by PyTorch, and 249.91 MiB
is reserved by PyTorch but unallocated.
```

The allocation was inside the training forward, in the patch-distance tensor,
under `torch.no_grad()`. The 246 MiB figure is the neighbor stack produced by
the 256 MiB group budget on a batch of four 128×128 crops. The driver at that
time retried out-of-memory only around full-image evaluation, not around the
training step, and it saved the spatial checkpoint only after the training
loop returned. The step-300 weights were not written. There is no
`spatial_spatial.pt`.

Treat 26.341 as a log line from an unfinished validation curve. It is a
reason to finish the protocol. It is not a result.

### The restart did not finish either

`run_spatial.log` records a new process that skipped the completed AWGN block,
reproduced the spatial-noise sweeps, passed the scalar init check
(25.670 vs 25.667), logged scalar step 1 (validation 18.961, `head_w`
0.14887), and stopped. There is no Python traceback in that file. The session
was interrupted. Step 1 is the same overshoot already seen in the finished
scalar log. It is not a new finding.

After that failure the source was changed so a later run can survive the
allocator: 64 MiB groups, a one-time retry around the training step, cache
release every 10 steps, and a checkpoint write whenever validation improves.
Those changes are in the copied `nlm.py` and `run_experiment.py`. They do not
repair the missing test score. They are the starting implementation for
Priority 1, not a completed Priority 1.

## 11. Parameter inventory

Values below are from the copied source on 2026-10-07. If a future edit
changes a constant, update this table in the same change and say so.

### NLM

| Parameter | Value |
|---|---|
| Patch | 7 |
| Search | 21 |
| Distance | mean squared patch difference |
| Weight | `exp(-D / h(i)^2)` |
| Padding | reflect |
| `h` lower clamp inside the exp | `1e-12` on `h^2` |
| Group budget in current source | 64 × 1024 × 1024 bytes |
| Group budget during the AWGN result | 256 MiB, inferred from the 246 MiB OOM and from the comment now in `nlm.py` |

### H-Net

| Parameter | Value |
|---|---|
| Input | noisy image, 1 channel, values in `[0, 1]` |
| Width | 32 |
| Depth | 4 residual blocks |
| Block | two 3×3 convs, ReLU, residual add |
| Head | 1×1 conv, 32 → 1 |
| Scalar behavior | global average pool, then head, then expand |
| Spatial behavior | head at full resolution |
| Output | `h_lo + (h_hi - h_lo) * sigmoid` |
| `h_lo`, `h_hi` on both finished sweeps | 0.02, 0.50 |
| Head init | weight `Normal(0, 0.2)`, bias set to `h*` then recentered for 4 steps on random crops |
| Init acceptance | validation PSNR within 0.3 dB of fixed `h*` |
| Parameters | 74337 |

### Training

| Parameter | Value |
|---|---|
| Crop | 128 |
| Batch | 4 |
| Optimizer | Adam |
| Learning rate | `3e-3`, halved at step `STEPS // 2` = 200, after that update |
| Loss | mean absolute error against the clean crop |
| Steps | 400 |
| Validation interval | 50, plus step 1 and step 400 |
| Checkpoint | highest validation PSNR |
| Gradient clip | global norm 1.0 |
| Seed index | `(0,)` only |
| Augmentation | random crop, flip up-down, flip left-right, new noise each step |

### Sweeps

| Parameter | Value |
|---|---|
| `h` grid | `geomspace(0.02, 0.50, 12)` |
| Endpoint rule | extend once if the best point is an endpoint; did not fire |
| `k` grid | 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8 |
| Hand sigma | Immerkaer, 21×21 pool |
| Oracle constant | per test observation, best grid `h` using the clean image; PSNR only |

### Evaluation

| Parameter | Value |
|---|---|
| PSNR | `10 * log10(1 / MSE)` per image, MSE clamped at `1e-12`, peak 1 |
| SSIM | 11×11 Gaussian sigma 1.5, C1=`0.01^2`, C2=`0.03^2` |
| Test realizations | 2 |
| Aggregation | mean over realizations inside an image, then mean and population std across images |
| Mechanism | per-observation mean, std, min, max of `h`; Pearson with true sigma; Pearson with noisy gradient magnitude |

### Hardware observed on the machine that produced the logs

| Item | Observed |
|---|---|
| OS | Windows |
| Python | 3.11.9 |
| PyTorch | 2.10.0+cu128 |
| GPU name in the log | NVIDIA GeForce RTX 5060 Laptop GPU |
| Capacity in the OOM message | 7.96 GiB |
| Other libraries at handoff time | NumPy 2.4.4, Pillow 12.1.0, OpenCV 4.13.0, Matplotlib 3.10.8 |
| `expandable_segments` | not supported on this build; setting it does not fix the allocator |

A full Kodak frame forward was about 0.80 s with either group budget. A 128
crop forward-backward was a few hundredths of a second. Validation sweeps are
on the order of 10 s per grid point for four images.

## 12. Next experiments

Do not run these as part of reading this file. When they are run, each one
gets its own directory under `experiments/`, its own config, its own log, and
its own metrics file. Results that are allowed to be quoted are copied to
`results/` only after the run finishes. A failed run is copied too, with the
failure in the name, and is not replaced quietly by a later success.

The scientific target of this study is to make the scalar-versus-spatial
comparison harder to explain away. The five priorities below are ordered.
Do not start a later priority by quietly changing the protocol of an earlier
one.

### Priority 1. Finish spatially varying noise without changing the definition

Question: under a ramp from 15/255 to 45/255 with random axis and direction,
does spatial `h(x, y)` beat scalar `h` on the frozen test set?

Keep patch, search, split, seeds, steps, learning rate, loss, architecture,
grids, and the NLM formula aligned with section 11. The 64 MiB grouping is an
acceptable implementation detail only if it is disclosed as a change in
summation order relative to the AWGN run. Do not also change the step count
or the learning rate in the same run. If a memory fix requires a smaller
batch, that is a second factor: record it, and do not pretend the training
distribution was identical.

Score the test set once, after validation has chosen `h*`, `k*`, and the
checkpoints. Write PSNR and SSIM for noisy, fixed, hand, both oracles, scalar,
and spatial. Write the mechanism block. Write the kodim17 figure with the
precommitted left-to-right ramp, and label it as a picture.

The scalar model has to be trained again. The file on disk is step 1.

Interpretation, decided before looking at the test number:

- Spatial above scalar, with scalar near fixed `h*`: the AWGN reading
  survives a noise field that really does vary. Still one seed.
- Spatial near scalar: the network's freedom was not spatial under this
  protocol, whatever the validation curve looked like at step 300.
- Spatial below fixed `h*`: the hypothesis fails on this protocol. Report
  that.
- A validation number, including 26.341, is never the substitute for the
  paragraph above.

### Priority 2. Resolution ladder

Question: does test performance change systematically as the learned
bandwidth is forced to coarser spatial resolution?

Use the same trunk and the same NLM. Constrain the control to a grid of
`1×1`, `2×2`, `4×4`, `8×8`, `16×16`, and full resolution. Upsample each
coarse map to the image size before NLM. Bilinear or nearest must be chosen
once and used for every rung. `1×1` is the scalar control. Full resolution is
the spatial model. Intermediate rungs are the experiment.

Do not assume the curve is monotone. A jump at `1×1` to `2×2` and a flat line
after that is a different claim from a steady climb. Report the shape.

Run this on the protocol that Priority 1 has actually finished, and on
stationary AWGN if the comparison is meant to be about structure rather than
about the ramp. Say which protocol a given ladder used. Do not pool them.

### Priority 3. Map-location controls

Question: does the gain require the right value in the right place?

For a trained spatial model, evaluate three maps through the same NLM:

- A. the learned map, unmoved;
- B. the mean map: replace every pixel by the spatial mean of A;
- C. the shuffled map: keep A's multiset of values, destroy the assignment
  to locations.

Choose the shuffle unit before running (pixel, or a tile) and keep it. A
pixel shuffle and a tile shuffle answer different questions.

Interpretation, if A beats C: location matters, not only the histogram of
`h`. If A ties C: the network helped by producing a useful bag of bandwidths.
If A ties B: the map's variation was unused at test time. These are tests,
not findings.

### Priority 4. Content versus noise

Question: is the map tracking local noise strength, image structure, or both?

Stationary AWGN cannot be explained by a sigma ramp. For that protocol,
compare the learned map with the clean image, the noisy image, gradient
magnitude, and local variance. The existing 0.385 correlation with the noisy
gradient is a starting measurement, not the analysis. Do not assume the sign.
Flat regions, edges, and texture can want different bandwidths, and a
positive correlation with gradient magnitude does not say which of those won.

For spatially varying noise, after Priority 1, compare the learned map with
the true sigma field and with image structure on the same observations.
Report both correlations. A map can follow sigma and still be modulated by
texture.

The kodim17 figure is the qualitative version of this section. Add quantitative
summaries per image. Do not pick a different Kodak image because kodim17
looks messy.

### Priority 5. More seeds

The finished AWGN result is one training seed. The gap is about 0.28 dB on
the test mean, with an across-image standard deviation near 1.55 dB. That
standard deviation is the spread of image difficulty, not the uncertainty of
the spatial-minus-scalar difference.

Before calling the gap stable, repeat the learned rows with 3 to 5 training
seeds. Keep the data split and the frozen test noise fixed so the seeds differ
in initialization and training order, not in the test draw. Report the paired
difference (spatial minus scalar) per seed, and a confidence interval or a
simple seed-wise range. Do not average in a failed seed. Do not add seeds
until the protocol being repeated is the finished one.

A second seed is not required merely because a gap is small, and a large gap
on one seed is not a confidence interval. Five seeds is the intended
confirmation, not a search for a seed that widens the gap.

## 13. Threats to validity

- One training seed. Initialization of the head is random (`std 0.2`) even
  though the bias is centered on `h*`.
- Twelve training images, four validation images, eight test images. A split
  fixed by filename is honest and also small.
- Grayscale only. The operator's channel average has not been tested in color.
- Synthetic Gaussian noise, clipped to `[0, 1]`. Not camera noise, not
  speckle, not a scanner.
- Patch and search are frozen at 7 and 21. The study does not say those
  windows are optimal. It says the bandwidth was the isolated factor.
- The spatial AWGN model was still improving at step 400. The reported gain
  is the gain inside a precommitted budget.
- Scalar and spatial are the same architecture trained separately, not two
  heads on one shared trunk. A shared-trunk ablation has not been run.
- The hand-crafted rule uses a 21×21 noise estimate. A different classical
  rule might do better. The claim is about this rule, which was specified
  before the test score.
- Oracle constant-`h` searches only the validation grid, not a continuous `h`.
- The figure uses a test image. Looking at it and then changing the model
  would leak the test image even though the noise draw is not the scored one.
- Summation grouping changed from 256 MiB to 64 MiB after the AWGN result.
  That is not a new operator, and it is also not bit-identical.
- Validation and test each use one frozen noise generator. Two test
  realizations reduce draw noise; they do not explore the noise process.
- Pearson correlation is linear and global. A map that raises `h` on edges
  and lowers it on texture can show a weak correlation and still be
  structured. Priority 4 exists because of that.

## 14. Claims

### Defensible from the finished artifacts

- The estimator used for every main-table row is the Buades form in section 4,
  with patch 7 and search 21, and the network does not emit pixels.
- On Kodak `kodim17`–`kodim24`, stationary Gaussian noise at `25/255`, two
  frozen realizations, one training seed, the spatial model reached 28.314 dB
  mean PSNR and the scalar model 28.036 dB. Fixed `h*` was 28.033 dB.
  Hand-crafted local sigma was 27.442 dB. Oracle constant-`h` matched fixed
  `h*`. These are means across eight images.
- The scalar map on that test set is constant to numerical noise. The spatial
  map has mean within-image std 0.00685 and a 0.385 correlation with the
  gradient magnitude of the noisy image.
- The validation procedure that chose `h*`, `k*`, and the checkpoints is the
  one in section 5, and the test split was not used for those choices.
- A sigma-proportional map is expressible inside this operator. On the
  spatial-noise validation set, `k = 0.75` times the true sigma field scored
  26.035 against 25.667 for the best constant. That is a validation
  diagnostic, and it is labeled as an oracle.

### Not supported, and not to be said

- A test score, a test gap, or a mechanism correlation for spatially varying
  noise.
- That step-300 validation PSNR 26.341 is a result.
- That `spatial_scalar.pt` is the finished scalar model for the ramp.
- That the method is state of the art, or that it replaces NLM, or that the
  Buades weights were learned.
- That modern nonlocal networks (learned affinities, unrolled blocks) were
  compared. They were deliberately not trained. An undertrained denoiser must
  not be added later as a weak baseline.
- That the 0.385 correlation identifies edges, texture, or flat regions.
- That performance will rise monotonically with control resolution.
- That the gain is stable across seeds.
- Any statement about biomedical, clinical, microscopy, MRI, CT, or ultrasound
  performance. No such image has been in the study.
- That OpenCV `h = 20` or `h = 30` is the classical baseline of the main table.

## 15. Rules that future Codex work must preserve

- Never tune using the test split. That includes the step count, the learning
  rate, the architecture, the upsample kernel, the shuffle tile, and which
  image is shown in the figure.
- Never modify `C:\testly_external_workspace\spatial-nlm`. Read it. Copy from
  it. Do not write to it.
- Never report an incomplete run as a final result. A validation curve from a
  crashed job stays in the failure log.
- Never compare methods that used different NLM operators unless the
  comparison is explicitly a separate experiment about the operator.
- Keep the scalar and spatial backbones, the training recipe, the `h` bounds,
  and the operator aligned. If one of those must change, change it for every
  learned row that will be compared, or label the mismatch.
- Always label oracle constant-`h` and oracle true-sigma as oracles.
- Do not change two experimental factors in one run and then attribute the
  difference to one of them. Memory fixes that alter batch size or step count
  are factors.
- Preserve the config, the seed, the log, and the metrics file for every
  attempt, including failed attempts.
- Every reported number must point at a file in `baseline_results/` or in
  `results/`, and at the key or the log line.
- Keep three layers separate in prose: confirmed evidence, hypothesis,
  planned work.
- If a run stops, record the stop. Do not delete the failed log because a
  later run succeeded.
- Do not write medical claims from Kodak experiments.
- Do not wire this project into the Testly application, its training jobs, or
  its production pipeline.
- Do not load `spatial_scalar.pt` as a finished model.
- When loading `awgn_scalar.pt` or `awgn_spatial.pt`, pass the matching
  `spatial` flag. The checkpoint will not correct a wrong flag.

## 16. Inconsistencies between prose, logs, and source

1. **Group budget.** Current `nlm.py` uses 64 MiB and says a 256 MiB stack
   failed. The AWGN metrics and the OOM at spatial step 300 were produced
   before that edit (`nlm.py` timestamp 13:24, metrics timestamp 13:06, OOM
   log 13:19). The formula did not change. The grouping of the sum did.
   Quote AWGN numbers from `metrics.json`, not from a rerun, unless the rerun
   is a declared new experiment.

2. **`spatial_scalar.pt` is the wrong generation.** The 400-step curve is in
   `run.log`. The file is the step-1 overwrite from `run_spatial.log`, matched
   by head-weight 0.14887. The AWGN checkpoints do match their selected steps
   (scalar 0.18987 at step 300, spatial 0.16578 at step 400).

3. **Original README versus artifacts.** `baseline_snapshot/README.md` says
   both figure files are produced. Only `figure_awgn_kodim17.png` exists.
   The original README also does not mention the residual trunk, the 0.2 head
   initialization, the 400-step budget, or the learning rate. Those are in
   the source. The source wins.

4. **`run.log` and `run_awgn_preserved.log` are the same file** (same SHA256).
   The second name was a defensive copy made before the restart. Neither file
   contains a spatial-noise test score.

5. **`metrics.json` and `metrics_awgn.json` are the same file** (same SHA256).
   Both are the AWGN protocol only. A reader who expects `metrics_awgn.json`
   to be a partial extract and `metrics.json` to be the full study will not
   find a spatial block in either.

6. **Oracle constant-`h` has no SSIM** in the JSON. The driver never computed
   it. Do not fill it in by memory.

7. **Logged `h* = 0.0864`** is the print of
   `0.08638876637025905`. Use the JSON value if a later run must hit the same
   grid point. The grid itself is `geomspace`, so regenerating it is exact
   given NumPy's implementation. Pin the NumPy version (2.4.4) if the grid
   must match bit for bit.

8. **`run_spatial.log` has no failure line.** It simply stops after scalar
   step 1. The absence of a traceback is the signature of an interrupted
   process, not of a successful short run.

9. **Correlation with true sigma on AWGN** is null for the scalar model and
   about `1e-7` for the spatial model. Neither number is a measured
   relationship. Sigma does not vary.

## 17. Directory map

```text
spatial-nlm-spatial-control-study/
  README.md                  this handoff
  baseline_snapshot/         copy of the source at handoff time; do not edit
    data.py
    metrics.py
    models.py
    nlm.py
    run_experiment.py
    requirements.txt
    README.md                the short README from the first experiment
  baseline_results/          copy of artifacts; do not edit
    metrics.json
    metrics_awgn.json
    run.log
    run_awgn_preserved.log
    run_spatial.log
    figure_awgn_kodim17.png
    awgn_scalar.pt           finished AWGN scalar model, best val = step 300
    awgn_spatial.pt          finished AWGN spatial model, best val = step 400
    spatial_scalar.pt        NOT a finished model; step-1 overwrite
  experiments/
    spatial_noise/           Priority 1, empty
    resolution_ladder/       Priority 2, empty
    map_location_controls/   Priority 3, empty
    content_analysis/        Priority 4, empty
    multiseed/               Priority 5, empty
  results/                   only finished, or explicitly failed, new runs
  notes/                     working notes; this README stays the authority
```

Not copied, and not to be edited in place:

- `C:\testly_external_workspace\spatial-nlm\data\kodak\` (the 24 PNGs)
- anything under the Testly repository

Absent on purpose, because the first experiment never produced them:

- `figure_spatial_kodim17.png`
- `spatial_spatial.pt`
- a `protocols.spatial` block in `metrics.json`
- a preserved log of the discarded plain ReLU trunk

## 18. Codex start here

Read this file once, then follow the order. Do not start at Priority 2
because the ladder is more interesting than the unfinished protocol.

1. Verify the copies against the original workspace. Hashes for the result
   files that define the science:

   | File | SHA256 |
   |---|---|
   | `metrics.json` | `B9C5665AE3E75DD3BB8985DFBDEDB68DCB8B107DB711D85A535669B3EA550728` |
   | `run.log` | `7365969112F57CCD33430F23B51C243AD43F356BBB01378A3960AF6D5D255780` |
   | `figure_awgn_kodim17.png` | `61F04AB2EF2160EDDEC359008A6639F842E5CB9F4001C2A3701AC8E2804CC2B9` |
   | `awgn_scalar.pt` | `DC9DD4A21798595FAA810BB56C70DFDCFC8DD1855B05CD09DE3AF46DEAEC3232` |
   | `awgn_spatial.pt` | `6A36E5FB7EFBC301FFC567194075B5C379508C0E38384EA407E67A229C367F12` |
   | `spatial_scalar.pt` | `44E96447F5F108141C90CF7AA8865215C5E5C263E784100C82327340D393FD64` |

   `spatial_scalar.pt` matching that hash means the overwrite is still the
   overwrite. It does not become a valid checkpoint by being verified.
   Confirm `baseline_snapshot/nlm.py` still says `budget = 64 * 1024 * 1024`
   and that the original workspace was not written by the verification.

2. Finish the spatially varying noise experiment under the definition in
   Priority 1. Train both models again. Do not resume from
   `spatial_scalar.pt`. Score the test set once. Leave the AWGN table as it
   is unless a disclosed rerun replaces the whole table.

3. Implement the resolution ladder. Choose the upsampler once. Do not assume
   a monotone curve.

4. Implement the mean-map and shuffled-map controls on a finished spatial
   checkpoint. Decide the shuffle unit first.

5. Analyze `h(x, y)` against image structure and, where the protocol has a
   varying sigma, against the true sigma field. Keep the AWGN gradient
   correlation as a recorded number, and extend it rather than replacing the
   artifact.

6. Repeat the finished learned comparison on 3 to 5 training seeds. Report
   paired differences.

7. Consolidate. A number enters the final summary only with a file path.
   Incomplete logs stay in a failure section. Biomedical language stays out
   until a biomedical dataset is actually in the protocol, which is outside
   the five priorities above.
