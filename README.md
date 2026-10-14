# biailab

## Spatial-NLM research archive

This repository preserves experiments asking whether a neural controller that
predicts a spatial bandwidth `h(x,y)` can improve classical Non-Local Means
relative to the same controller family restricted to one scalar `h` per image.
H-Net predicts bandwidth, never restored pixels. Both systems use the same NLM.

### Contents

| Directory | Contents |
|---|---|
| [spatial-nlm](spatial-nlm/) | Original Kodak implementation, 24 input PNGs, preserved logs, checkpoints and results |
| [spatial-nlm-spatial-control-study](spatial-nlm-spatial-control-study/) | Scientific handoff, baseline copies, CamScanner source pairs, experiment implementation, prepared pairs and artifacts |
| [CamScanner experiment](spatial-nlm-spatial-control-study/experiments/camscanner_scalar_vs_spatial/) | Completed matched scalar/spatial document experiment |
| [Consolidated results](spatial-nlm-spatial-control-study/results/camscanner_scalar_vs_spatial/) | Verified document metrics, per-image CSV and provenance |

Scientific source files and artifacts were copied byte for byte. The
[publication manifest](PUBLICATION_MANIFEST.json) records SHA256 and size for
every copied file. Runtime Python bytecode/cache directories are excluded.
The original workspaces and Testly production source were not modified.
Testly production models and source are not bundled into the experimental path.

### Confirmed document result

100 real dirty/clean pairs, 92 detected content groups; train59 / validation21 /
test20. Detected repeated questions stay in one partition. Both neural arms
completed 400 updates using identical initial tensors, seed, crop/augmentation
schedule, optimizer, loss, preprocessing, NLM and postprocessing.

| System | Test mean PSNR (dB) | Test mean SSIM |
|---|---:|---:|
| Scalar | 25.896911 | 0.960412 |
| Spatial | 26.324671 | 0.928183 |
| Paired spatial minus scalar | +0.427759 | -0.032229 |

Source: [metrics.json](spatial-nlm-spatial-control-study/results/camscanner_scalar_vs_spatial/metrics.json),
`means`. [Per-image table](spatial-nlm-spatial-control-study/results/camscanner_scalar_vs_spatial/per_image.csv).
Spatial PSNR is higher and SSIM is lower on all 20 test images. These
measurements do not establish superiority across both metrics.

Validation selected the scalar initialization checkpoint (step0) and spatial
step400. Both full training budgets completed; the scalar's later updates did
not beat its initialization on validation. Checkpoints record the scalar/spatial
flag explicitly because parameter shapes alone cannot distinguish the modes.

Common document processing is grayscale /255, local paper-brightness
normalization using 31x31 closing and Gaussian sigma31/6, unchanged NLM with
patch7/search21, and [0,1] clipping. The shared calibrated bandwidth interval is
[0.0005213606195, 0.1], with validation h*=0.03845335369. Ninety-five pairs had
dimension differences; nineteen required shared homography registration.
No pairs were excluded and no training/evaluation failure or OOM occurred.

Read the [experiment README](spatial-nlm-spatial-control-study/experiments/camscanner_scalar_vs_spatial/README.md)
for exact formulas, split, determinism, loss, checkpoint rules, limitations and
reproduction commands. The [Testly source inspection note](spatial-nlm-spatial-control-study/experiments/camscanner_scalar_vs_spatial/testly_cleaning_note.md)
explains the cleaning goals that informed the deterministic pipeline.

### Figures

* [First test image: dirty, clean, scalar/spatial outputs, errors and h maps](spatial-nlm-spatial-control-study/experiments/camscanner_scalar_vs_spatial/results/figures/camscanner_001_comparison.png)
* [Largest PSNR gain](spatial-nlm-spatial-control-study/experiments/camscanner_scalar_vs_spatial/results/figures/camscanner_045_comparison.png)
* [Median PSNR gain](spatial-nlm-spatial-control-study/experiments/camscanner_scalar_vs_spatial/results/figures/pair_100405_pair_01076_003_comparison.png)
* [Smallest PSNR gain and closeups](spatial-nlm-spatial-control-study/experiments/camscanner_scalar_vs_spatial/results/figures/pair_100405_pair_01076_015_closeups.png)
* [Diagram crop selected before test scoring](spatial-nlm-spatial-control-study/experiments/camscanner_scalar_vs_spatial/results/figures/camscanner_004_diagrams.png)

Every test image has a comparison figure; float outputs, h maps and errors are
preserved alongside the visualizations. Test scoring was performed once after
the two checkpoints were frozen. Figure layout revisions reused saved outputs.

### Preserved Kodak evidence and limitations

The [scientific handoff](spatial-nlm-spatial-control-study/README.md) distinguishes
the completed one-seed stationary AWGN result from the incomplete spatial-noise
attempt. The preserved `spatial_scalar.pt` is a step1 overwrite, not a finished
model; no finished spatial-noise test result or spatial checkpoint exists.
These incomplete artifacts are retained as evidence, not promoted to results.

The document experiment uses one training seed, grayscale scoring,
CamScanner-produced targets and offline target-guided geometry preparation.
It is not a production Testly benchmark or an end-to-end camera geometry system.
No biomedical images were run and no medical performance claim is supported.

### Working with this archive

Keep the archived metrics, logs, baseline snapshots and checkpoints intact.
Use a fresh experiment copy/run directory when reproducing training; the
completed training scripts deliberately refuse to overwrite an existing attempt.
The original Windows absolute paths in provenance manifests are historical
evidence, not required installation paths. Repository layout places Kodak data
under `spatial-nlm/data/kodak/` and both CamScanner collections under
`spatial-nlm-spatial-control-study/camscanner/`.

The CamScanner audit derives the input root from its experiment location;
prepared pair paths are relative to that experiment. Recreate manifests in a
fresh run copy as described in its README rather than rewriting frozen evidence.
Exact library versions are preserved in the experiment's `requirements.txt`
and environment manifests. No new training was performed for this publication.

### Bettertest: frozen-system diagnostic on three cleaner references

The [bettertest report](spatial-nlm-spatial-control-study/experiments/camscanner_scalar_vs_spatial/BETTERTEST_REPORT.md)
evaluates pairs `0000015`, `0000025` and `0000035` with the existing scalar
step0 and spatial step400 checkpoints. There was no retraining, recalibration
or change to the preprocessing, NLM, bandwidth bounds or postprocessing.
Only `0000025` required shared homography registration. The original 20-image
CamScanner test and its artifacts remain intact.

| System | Bettertest mean PSNR (dB) | Bettertest mean SSIM |
|---|---:|---:|
| Scalar | 25.694751 | 0.976257 |
| Spatial | 25.680543 | 0.974001 |
| Paired spatial minus scalar | -0.014208 | -0.002256 |

Source: [diagnostic metrics](spatial-nlm-spatial-control-study/experiments/camscanner_scalar_vs_spatial/results/bettertest/metrics.json),
`means`; [per-image CSV](spatial-nlm-spatial-control-study/experiments/camscanner_scalar_vs_spatial/results/bettertest/per_image.csv).
Spatial SSIM is lower on all three pairs, with a smaller deficit than the
original test. PSNR improves slightly on two pairs and declines on one.
Background variance/noise proxies are lower in the new clean targets, but
content, framing and margins change too. Three selected pairs do not isolate
a causal effect of target cleanliness or establish general performance.

The [diagnostic result directory](spatial-nlm-spatial-control-study/experiments/camscanner_scalar_vs_spatial/results/bettertest/)
contains the preparation manifest, environment, checkpoint/source/image hashes,
float32 arrays, PNG outputs, figures, preserved logs and exact scoring source
snapshot. The [six raw images](spatial-nlm-spatial-control-study/bettertest/)
are included. A failed preliminary content-overlap audit is retained and
documented; scoring succeeded once after its guard was fixed.

### See how the cleaning pipeline works

Read the [six-stage implementation explanation](spatial-nlm-spatial-control-study/experiments/camscanner_scalar_vs_spatial/PIPELINE_EXPLANATION.md)
and inspect [0000015 through the actual pipeline](spatial-nlm-spatial-control-study/experiments/camscanner_scalar_vs_spatial/results/bettertest/pipeline_explanation_0000015.png).
The educational figure reuses saved outputs and regenerates only deterministic
preprocessing/background intermediates; no model inference or scoring was repeated.

Standalone cleaned PNGs: [scalar](spatial-nlm-spatial-control-study/experiments/camscanner_scalar_vs_spatial/results/bettertest/outputs/0000015_scalar.png)
and [spatial](spatial-nlm-spatial-control-study/experiments/camscanner_scalar_vs_spatial/results/bettertest/outputs/0000015_spatial.png).
All three pairs' scalar/spatial PNGs and authoritative float32 NPYs are included.
New-file integrity is recorded in [BETTERTEST_PUBLICATION_MANIFEST.json](BETTERTEST_PUBLICATION_MANIFEST.json).
