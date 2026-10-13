# CamScanner document cleaning: scalar versus spatial NLM bandwidth

Status: calibration, both 400-step training budgets, frozen test evaluation and
visualization completed. Verified results and artifact paths are below.

This is the first real paired-document experiment in the Spatial-NLM study.
The question is whether a spatial bandwidth controller reconstructs CamScanner
clean targets better than an otherwise matched controller restricted to one
bandwidth per image. The prior Kodak AWGN table stays in the preserved baseline;
it is not a document result. This experiment does not add synthetic noise.

## Testly source evidence

The located CUDA pipeline is
`C:\testly\testly_backend\backend\services\pipeline_cuda.py`.
The technical trace and inspected modules are in [testly_cleaning_note.md](testly_cleaning_note.md).
The camera path performs geometry preparation, full-page restoration and
figure-specific restoration/composition. Successful hybrid output goes directly
to final; mask-based seamless whitening is a fallback after hybrid failure.
The source seeks white paper, readable text and preserved diagrams, beyond
camera-noise suppression alone. These are local source findings, not claims
about the currently running production artifact.

The production binarizer, NAFNet, detectors, rotation classifier and learned
dewarper are intentionally omitted. Their learned pixel outputs would add a
different restoration model to the causal comparison. The experimental network
only emits bandwidth, and every restored pixel comes from the same classical NLM.

## Dataset audit and geometry

`manifests/pairs.json` records every source path, SHA256, dimensions, decoded
channels/bit depth, luminance distribution, background/ink proxies, local
contrast, edge strength, Laplacian noise proxy and histogram. Source images
are RGB, 8-bit JPEGs, and the targets are continuous rather than binary masks.
Targets visibly retain diagrams, highlights and handwritten annotations.
The experiment uses grayscale only; it does not assess preservation of color.
`manifests/pair_analysis.json` records TRAIN/VAL summaries separately. On train,
equal-image mean intensity changes from .64866 dirty to .92401 clean; the top
quartile background proxy from .74713 to .99959, and the >=.98 white fraction
from .00113 to .77650. These descriptive values support a background/contrast
cleaning task, beyond additive-noise removal. The dark-pixel statistic is a
quantile proxy, not a ground-truth semantic text segmentation.

There are 50 dirty and 50 clean files immediately under `camscanner`, plus a
second explicitly included collection of 50+50 under
`camscanner/pair_100405_pair_01076`. The main collection pairs numeric dirty
identity with the terminal clean filename suffix. The second collection uses
the sorted numeric source-pair identifiers' rank with the terminal clean suffix.
Those name proposals are verified with actual SIFT correspondences; arbitrary
directory order is never a pairing rule. All 100 proposals passed the feature
geometry gate; no source file was unmatched.

Geometry gate: SIFT, 6000 features, ratio .7; homography RANSAC 2px, 5000
iterations, confidence .999. Require >=30 inliers, >=.5 inlier fraction, page
coverage >=.2, median residual <=.6px and p95 <=1.5px. Overall maximum median
residual was .2404px; maximum p95 was 1.2162px. Feature-based checks do not
guarantee perfect correspondence at every thin stroke.

Dirty and target dimensions often differ by small rounding amounts. Dirty is
resized once to the target grid with linear interpolation. If median identity
feature displacement exceeds .5px or p95 exceeds 1.25px after that resize,
apply the fitted shared homography. Nineteen pairs require this extra registration.
Then crop BOTH images to a rectangle with complete valid source support,
with a 2px interior margin after registration. No per-method alignment occurs.
No PSNR/SSIM was computed on unprepared dirty/clean pairs.

The target is used to establish offline pixel correspondence. Thus this measures
restoration **after paired preparation**, not the ability to clean an arbitrary
unregistered camera capture end to end. Original JPEGs remain unmodified.
Prepared arrays are saved as grayscale PNGs, with hashes and ROI in split.json.

## Split and training data

`prepare_pairs.py` groups detected repeated question content before splitting.
Perceptual-hash candidates (distance <=32) are verified by SIFT homography:
>=35 inliers, >=.65 inlier fraction and >=.12 of the smaller descriptor set.
Near-duplicate screening is imperfect; the manifest records every detected link.
Sort groups by SHA256(`20261014:group_identity`), allocate the first rounded
20% to validation, the next rounded 20% to test, and the remainder to training.
Exact identities, group membership and all prepared paths are in split.json.
The concrete final counts are reported in the confirmed-results section.

`manifests/crops.json` contains all 400 batches, four crops per batch, image IDs,
128x128 coordinates, horizontal and vertical flip decisions. It is produced
once using NumPy seed 20261014. Both models read precisely the same manifest.
Preprocessing is applied to the full prepared page before cropping, so crop
statistics do not produce branch-specific normalization. Validation/test use
full prepared pages, not a sampled test crop.

## Shared deterministic document pipeline

Convert RGB to OpenCV grayscale and divide decoded intensities by 255. These
are gamma-encoded image intensities, not calibrated radiometric luminance.
Identical grayscale conventions apply to dirty and clean.

Validation compares a fixed candidate set **without learned models**:

* Identity.
* Global stretch: q_low=.02, q_high=.90 or .98;
  `clip((x-q_low)/max(q_high-q_low,10/255),0,1)` using each page's quantiles.
* Local paper normalization: grayscale morphological closing with 31 or 63
  square kernel, reflect101 boundaries, Gaussian sigma=kernel/6;
  `r=clip(x/max(background,.1),0,1)`, then
  `clip((r-b)/(1-b),0,1)` for fixed b=0 or .1.

Pick the highest equal-image-weight validation PSNR, then freeze the choice
before bandwidth calibration. Purpose: remove observed paper brightness and
slow illumination variation while retaining continuous strokes/diagram tones.
The candidate set is informed by Testly's contrast/background semantics;
the formulas are deliberately minimal and are not copied production restoration.
Per-image statistics follow one fixed rule; there is no manual per-image tuning.
Postprocessing is just `clip(output,0,1)` for both systems. There is no threshold,
sharpening, pixel-generating network, figure detector or branch-specific finish.

## NLM and document bandwidth calibration

`nlm.py`, `models.py`, `metrics.py` are byte-for-byte copies of the preserved
snapshot. NLM SHA256 is
`4D69F971F9D026DFC5ED305FD53E4C621EC4B80CC459C1339D82CD35924844CB`.

`D(i,j)=mean_patch,channels[(u(i+k)-u(j+k))²]`,
`w(i,j)=exp(-D(i,j)/max(h(i)²,1e-12))`,
`output(i)=sum_j w(i,j)u(j)/sum_j w(i,j)`.
Patch=7, search=21, reflect padding, reference-pixel h only, 64MiB neighbor groups.
No weights/affinities are learned. A constant h is a broadcast map through this
same function. Full-page inference calls the copied operator on 256px tiles
with a 13px halo, preserving all patch/search neighbors. H-Net runs on the full
page; scalar pooling is not replaced by independent per-tile pooling.
Sum grouping can differ from a monolithic call. In the CUDA tile check, max
absolute difference was 5.36e-7. The inherited direct-formula check was 1.19e-7;
gradients from restored pixels to h were also verified nonzero.

After preprocessing is frozen, estimate the median TRAIN mean-absolute
Immerkaer Laplacian scale. This is a scale proxy that includes document edges,
not an assertion of Gaussian noise. Sweep 12 geometric h values from
max(1e-4,proxy/8) to min(1,max(.1,proxy*16)), evaluating full validation pages.
If the optimum is an endpoint, extend that endpoint once by a decade (five
new values; high bound capped at 2). The grid minimum/maximum become the
identical neural bounds. h* is the highest validation mean PSNR; if it is
still a boundary optimum, initialization is nudged .1% inside the interval
because the sigmoid cannot represent a bound at finite logits. Report this
condition; do not keep extending after seeing a test score.
All candidate and grid scores are saved in calibration.json.

## Matched neural controllers and training

H-Net: 1->32 3x3 stem, four residual blocks of two 3x3 convolutions, 32->1
1x1 head. Each has 74337 parameters. The scalar branch globally pools the
trunk then broadcasts one sigmoid-bounded h. The spatial branch applies the
same head at every pixel. The theoretical spatial receptive field is 19px.

Both start from exactly identical state-dict tensors. Nonzero head weights
are Normal(0,.2). A shared head bias starts at h* and is centered with four
updates on the same first training batch; that single centered state is
copied to both branches. Both use seed 20261014, Adam, lr=.003, 400 updates,
batch4, crop128, L1 loss on the postprocessed output, gradient clip1.0,
validation at initialization, step1 and every50 steps. Best validation PSNR
selects a checkpoint; step0 is eligible to preserve a null/no-improvement result.
The gradient path is target -> L1 -> shared finish -> NLM -> h -> H-Net.
The test target is never loaded in calibration or training.

Python random, NumPy, Torch and CUDA seeds are recorded. Strict deterministic
Torch algorithms are enabled, cuDNN benchmark disabled, TF32 disabled,
float32/no AMP, CUBLAS_WORKSPACE_CONFIG=:4096:8, OpenCV threads1 and Torch
CPU threads4. run.py sets PYTHONHASHSEED. environment.json records actual
OS, interpreter, library versions, CUDA, GPU and source hashes. Determinism
within this setup does not imply bit equality across other hardware/software.

## Why this experiment isolates scalar vs spatial control

Both systems consume the same saved aligned dirty/target pair, same full-page
preprocessor, same postprocessor, copied NLM equation/windows, common h interval,
same feature architecture and initially identical tensors. The only forward-path
branch is whether trunk features are globally pooled before the bandwidth head.
Optimizer, loss, every crop/flip/order, budget, seed, validation observations,
checkpoint selection criterion and metrics match. Neither system emits pixels.
Checkpoint metadata explicitly records spatial mode, because state-dict shapes
alone would silently accept the wrong flag. Both training budgets must complete
and both checkpoint hashes must be frozen before any test score is computed.
This controls the prescribed contrast within one small dataset/one seed; it
does not establish general superiority, production performance, or seed stability.

## Metrics, visual artifacts and failure handling

PSNR uses peak1 and per-image MSE; SSIM is the copied Wang implementation,
11x11 Gaussian sigma1.5, C1=.01², C2=.03², including its zero-padding convention.
Same prepared rectangle and continuous intensity domain for both systems.
Means weight test images equally. per_image.csv reports scalar/spatial scores
and paired differences; metrics.json records those rows, means and frozen hash.
This is one real observation per test pair, not two synthetic noise draws.

Test starts only after frozen.json. Completed metrics cannot be overwritten.
An image journal supports continuing unscored images after an evaluation
failure while retaining completed observations. Checkpoints and manifests
must still match the original frozen hashes. Per-image float32 outputs, errors
and h maps are saved as NPY; PNGs are display-only and are not used for metrics.
Representative rule: first test identity and best/upper-median/worst paired
PSNR difference; every test pair also receives an eight-panel comparison.
An additional diagram crop from test identity `camscanner_004` was precommitted
from the source audit before test scoring (x [.38,.98], y [.05,.95] of the page).
Closeups use target-only edge, thin-stroke, bright-low-edge background and fixed
center rules, which do not select pixels according to model gains. Comparison
panels share intensity [0,1], error [0,.5] and h [h_min,h_max] display ranges.

Every run.py invocation creates a timestamped phase log and exit-status JSON.
Training logs every step and records tracebacks on failure. The code refuses
to overwrite a previous training attempt; a failed comparison is not scored.
No OOM or other failure is concealed by averaging or replacing a log.
Missing/ambiguous pairs stop preparation. Failed geometry is explicitly excluded.
Too few independent groups stops a held-out experiment rather than making a claim.

## Reproduction commands

Use a **fresh copy of this experiment directory with output directories removed**
to reproduce; preserve this completed attempt and the original input sources.
Required versions are the environment.json versions, not an unpinned install.
From Windows PowerShell in this directory:

```powershell
python run.py audit_dataset
python run.py prepare_pairs
python run.py verify
python run.py calibrate
python run.py train
python run.py evaluate
python run.py visualize
python run.py report
```

The input root is derived from the study directory; source JPEG paths are recorded
explicitly. No command downloads data or touches Testly/original spatial-nlm.
The first audit/preparation and verification during development were invoked
directly; their manifests/artifacts and verification values are preserved.

## Limitations

Small selected question-image sample; one training seed and fixed budget.
CamScanner clean targets are application-produced desired outputs, not measured
noise-free radiometric truth. Luminance scoring cannot assess color. Offline
target-guided alignment does not measure deployable geometry correction. Sparse
feature gates can miss local warp or small stroke mismatch. Detected repeated
content groups are not proof that all source-page dependence was recovered.
Training crop boundaries differ from full-page context, identically in both arms.
The small H-Net and classical convex-combination NLM cannot invent lost strokes
or necessarily reproduce a sophisticated document cleaner. Preprocessing itself
can account for most cleaning quality; the causal number is the paired difference.
No medical, production, or state-of-the-art claim is supported by this task.

## Confirmed results

Completed paired-document experiment, one seed. These numbers come from `results/metrics.json` -> `means`; per-image rows are in `results/per_image.csv`.

Valid pairs: 100; unmatched: 0; content groups: 92; geometric registrations: 19.

Split: train=59, val=21, test=20. Exact identities: `manifests/split.json` -> `partitions`.

Shared pipeline: `local_31_0.0`. Bandwidth interval [0.0005213606195, 0.1], fixed-h calibration optimum h*=0.03845335369; endpoint optimum=False. NLM patch7/search21. See `manifests/calibration.json`.

| System | Mean PSNR (dB) | Mean SSIM |
|---|---:|---:|
| Scalar | 25.896911 | 0.960412 |
| Spatial | 26.324671 | 0.928183 |
| Paired spatial − scalar | +0.427759 | -0.032229 |

Spatial has higher mean PSNR and lower mean SSIM in this completed run. The measurements do not show dominance across both metrics. No parameters were changed after this test observation.

Scalar checkpoint: `checkpoints/scalar/best.pt`, best validation step 0, validation PSNR 24.645559. Both completed 400 updates. SHA256: `6ab484abcf791d7cfa1104a9d2d1f5d1d461e4fe11b902048f4536d78c824ee2`.

Spatial checkpoint: `checkpoints/spatial/best.pt`, best validation step 400, validation PSNR 25.050228. Both completed 400 updates. SHA256: `50a3036259b30c93bc00496b117678ac546da31c8f396b4c1f78e5688c9d5ba0`.

| Image | Scalar PSNR | Spatial PSNR | ΔPSNR | Scalar SSIM | Spatial SSIM | ΔSSIM |
|---|---:|---:|---:|---:|---:|---:|
| camscanner_001 | 20.961714 | 21.367624 | +0.405910 | 0.931719 | 0.888812 | -0.042908 |
| camscanner_004 | 25.791977 | 26.301247 | +0.509270 | 0.970727 | 0.935925 | -0.034802 |
| camscanner_005 | 23.218803 | 23.362610 | +0.143806 | 0.969493 | 0.937554 | -0.031939 |
| camscanner_012 | 19.538500 | 20.165415 | +0.626915 | 0.907180 | 0.785757 | -0.121423 |
| camscanner_023 | 20.323921 | 21.041924 | +0.718002 | 0.902058 | 0.868403 | -0.033655 |
| camscanner_043 | 20.625334 | 21.120855 | +0.495522 | 0.927773 | 0.864145 | -0.063628 |
| camscanner_045 | 25.235569 | 26.062450 | +0.826881 | 0.978323 | 0.905880 | -0.072443 |
| camscanner_048 | 22.788639 | 23.450634 | +0.661995 | 0.931543 | 0.908125 | -0.023418 |
| pair_100405_pair_01076_003 | 28.923611 | 29.407080 | +0.483469 | 0.973593 | 0.955578 | -0.018015 |
| pair_100405_pair_01076_015 | 27.323185 | 27.400867 | +0.077682 | 0.978197 | 0.963321 | -0.014876 |
| pair_100405_pair_01076_023 | 26.726648 | 27.137814 | +0.411165 | 0.958334 | 0.936676 | -0.021658 |
| pair_100405_pair_01076_024 | 27.361736 | 27.674557 | +0.312820 | 0.973659 | 0.960488 | -0.013172 |
| pair_100405_pair_01076_026 | 25.895617 | 26.051365 | +0.155748 | 0.971312 | 0.948224 | -0.023089 |
| pair_100405_pair_01076_027 | 29.156643 | 29.322201 | +0.165558 | 0.987437 | 0.979010 | -0.008427 |
| pair_100405_pair_01076_028 | 27.856884 | 28.162369 | +0.305485 | 0.979809 | 0.963962 | -0.015848 |
| pair_100405_pair_01076_029 | 30.980736 | 31.179775 | +0.199039 | 0.987126 | 0.973497 | -0.013630 |
| pair_100405_pair_01076_030 | 27.103619 | 27.206804 | +0.103186 | 0.977837 | 0.965478 | -0.012359 |
| pair_100405_pair_01076_042 | 30.556490 | 31.307224 | +0.750734 | 0.974771 | 0.952080 | -0.022692 |
| pair_100405_pair_01076_047 | 27.559036 | 28.152235 | +0.593199 | 0.956743 | 0.921540 | -0.035203 |
| pair_100405_pair_01076_049 | 30.009565 | 30.618362 | +0.608797 | 0.970610 | 0.949216 | -0.021394 |

Representative figures (first / best / upper-median / worst rule, deduplicated):

* [camscanner_001 comparison](results/figures/camscanner_001_comparison.png) and [closeups](results/figures/camscanner_001_closeups.png).
* [camscanner_045 comparison](results/figures/camscanner_045_comparison.png) and [closeups](results/figures/camscanner_045_closeups.png).
* [pair_100405_pair_01076_003 comparison](results/figures/pair_100405_pair_01076_003_comparison.png) and [closeups](results/figures/pair_100405_pair_01076_003_closeups.png).
* [pair_100405_pair_01076_015 comparison](results/figures/pair_100405_pair_01076_015_comparison.png) and [closeups](results/figures/pair_100405_pair_01076_015_closeups.png).
* [Precommitted diagram crop](results/figures/camscanner_004_diagrams.png).

Every test image also has a comparison figure, float outputs, h maps and absolute errors.

Recorded phase failures: 0. Dataset exclusions: 0.

A preliminary console filename listing hit a Turkish-character encoding error; it was rerun with UTF-8 before the dataset audit. No training/evaluation failure or OOM was recorded.

Environment: Python 3.11.9, Torch 2.10.0+cu128, CUDA 12.8, OpenCV 4.13.0, NVIDIA GeForce RTX 5060 Laptop GPU. See `manifests/environment.json`.

The mean paired difference is the measured contrast on this prepared dataset and seed. It does not establish general document superiority, seed stability, color fidelity or production behavior. Target-guided geometry, CamScanner-generated targets and repeated/source-dependent question content remain limitations.
