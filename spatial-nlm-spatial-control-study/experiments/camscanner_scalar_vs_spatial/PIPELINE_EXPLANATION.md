# How this implementation cleans a document

The image restoration is deterministic paper normalization followed by classical
Non-Local Means. H-Net produces bandwidth values, never restored pixels.
This explanation follows `document_pipeline.py`, `models.py`, `nlm.py`,
`evaluate.py`, `config.py`, and the frozen manifests/checkpoint metadata.

[Real 0000015 pipeline figure](results/bettertest/pipeline_explanation_0000015.png)
shows prepared dirty, preprocessing, both saved bandwidth maps, both saved final
outputs, clean target, and the estimated paper field. It is educational only.
No training, recalibration, model inference or metric evaluation was repeated
to make that figure. Existing source/checkpoint/metric files were not changed.

## 1. Input and geometry

* **Input:** RGB 8-bit JPEG, values0..255.
* **Operation:** Apply EXIF orientation; OpenCV RGB-to-grayscale conversion.
  Paired preparation may resize dirty to the clean grid, apply a shared
  homography, and crop both to the same valid rectangle.
* **Output:** Prepared grayscale uint8 PNG.
* **Visual effect:** Geometry/pixel correspondence is corrected. Interpolation
  can change pixel values; cropping discards border content. This is not a
  noise-removal stage.
* **Scalar/spatial:** Receive exactly the same prepared dirty image.

0000015 is 760x928, with no registration or crop. The identity resize is a
no-op. Clean is an offline alignment/evaluation reference, not a network input.

## 2. Deterministic paper normalization: local_31_0.0

* **Input:** Prepared grayscale, converted to float32 and divided by255 by
  the input loader, before `preprocess()`.
* **Operation:** Estimate paper brightness using a 31x31 closing and Gaussian
  smoothing with sigma31/6 (about5.17px), with reflect101 boundaries.
* **Output:** The input divided by that estimated brightness and clipped.
* **Visual effect:** Paper moves toward white and slow illumination/shadows
  are reduced. Text is not forced to black. Gray diagram tones may change.
* **Scalar/spatial:** Still identical pixels.

```text
x = grayscale / 255
B = Gaussian(closing_31x31(x), sigma=31/6)
u = clip(x / max(B, 0.1), 0, 1)
```

Closing is dilation followed by erosion. It suppresses small dark foreground
features in the **background estimate**; its result does not replace the
document image. Gaussian smoothing makes the estimate a smooth paper field.
`max` is per pixel. The local pipeline's black-level b=0 makes the subsequent
`(r-b)/(1-b)` stretch an identity; there is no added black-level stretch.

Thin dark strokes can remain dark relative to the whitened paper. Broad dark
diagram regions can contaminate the paper estimate and get excessively
brightened. This stage is not guaranteed to preserve every foreground tone.

In 0000015, mean intensity changes from0.627734 prepared dirty to0.970068
after preprocessing. The large visible whitening happens before NLM. This
does not quantify a fraction of metric improvement; no ablation was performed.

## 3. H-Net and its scalar/spatial branch

* **Input:** The same normalized full page u.
* **Operation:** A 3x3 1-to-32 stem and four residual blocks with two 3x3
  convolutions each produce features. A 1x1 head predicts raw bandwidth logits.
* **Output:** Bandwidth h, not image pixels.
* **Visual effect:** None on the document yet; this prepares the NLM control.
* **Scalar/spatial:** Same architecture, separate checkpoint parameters.

```text
Scalar:  features -> mean over height,width -> head -> broadcast raw to HxW
Spatial: features -> head at each location         -> raw(x,y)

h = h_min + (h_max-h_min) * sigmoid(raw)
h_min = 0.0005213606195074932
h_max = 0.1
```

The chosen scalar checkpoint is validation step0; spatial is step400. Their
weight tensors differ. "Same trunk" refers to architecture, not identical
feature tensors. Both have74337 parameters. In the actual residual block,
ReLU is in-place: `return x + conv2(ReLU(conv1(ReLU(x))))`; the first ReLU also
clamps the tensor used by the skip addition. This behavior is shared by both.

h is patch-difference tolerance, not a measured sensor-noise sigma. Smaller h
rejects unlike patches more strongly; larger h gives them more weight.
Scalar cannot assign a different tolerance to paper, text edges and diagrams
within one page. Spatial can, although its choices are not guaranteed optimal.

For 0000015, scalar h=0.03794755 everywhere. Spatial h ranges from about
0.00052136 to0.09999976, with mean0.00323254.

## 4. Classical NLM makes every output pixel

* **Input:** Normalized u and that branch's h map.
* **Operation:** Compare patches, calculate exponential weights, average pixels.
* **Output:** Classical NLM restored image.
* **Visual effect:** Similar-region fluctuations may be averaged; strokes and
  diagram edges can be preserved or softened depending on the distances and h.
* **Scalar/spatial:** Exactly the same NLM function; different h maps.

```text
D(i,j) = mean_7x7[(u(i+k)-u(j+k))^2]
w(i,j) = exp(-D(i,j)/max(h(i)^2, 1e-12))
output(i) = sum_j[w(i,j)*u(j)] / sum_j[w(i,j)]
```

Each reference pixel uses its7x7 patch and all441 patch centers in the21x21
search window, including itself. The weighted values are the neighbor center
pixels. h is taken only at the reference pixel i. Distances use the one-channel
preprocessed grayscale image, with no learned affinities or noise subtraction.
Reflect padding handles borders.

For a fixed distance D=.001, h=.02 gives weight about.082, whereas h=.1 gives
about.905. Larger h mixes a wider range of patch appearances. Very small h
makes the self patch dominate unless other patches match equally well.

Full-page NLM uses256px tiles with13px halo to preserve the full patch/search
neighborhood. Neighbor sums use the copied64MiB grouping. These are memory
arrangements, not separate scalar/spatial operators. H-Net sees the full page;
scalar pooling is global, not per tile. No neural output is a restored pixel.

## 5. Postprocessing and display

* **Input:** NLM output.
* **Operation:** Only `clip(output,0,1)`.
* **Output:** Float32 cleaned image, authoritative for metrics.
* **Visual effect:** Bounds numerical values; adds no sharpening/thresholding.
* **Scalar/spatial:** Same function.

No hard binarization, sharpening, learned pixel restoration, figure detector
or figure-specific restoration is present. A display PNG is generated from
the saved NPY by `round(clip(output*255,0,255))` and uint8 conversion. That
conversion is not used for PSNR/SSIM. The educational figure labels saved
outputs as "NLM + clip" because separate pre-clip arrays were not stored.

## 6. Where the final outputs differ

```text
dirty -> common geometry -> common grayscale/255 and paper normalization
                                |
                 same H-Net architecture, separate weights
                     /                             \
             global pool                         no pool
             head + broadcast                    spatial head
                  one h                            h(x,y)
                     \                             /
                         same classical NLM
                                |
                            same clip
                                |
                     scalar / spatial output
```

The forward **structural** difference is global pooling before the head and
broadcasting the scalar logit. The deployed branches also use their separately
selected checkpoint weights. Their restored-pixel difference is mediated
through the resulting bandwidth maps, not a different restoration operator.

| Effect | Responsible stage |
|---|---|
| Illumination/background variation | Deterministic paper normalization |
| Averaging local noise/fluctuations | Classical NLM |
| Preserving or damaging text edges | NLM patch weights controlled by h; preprocessing may also affect tones |
| Scalar/spatial output difference | Different bandwidth predictions from the two frozen controllers |

The current cleaning algorithm is fundamentally deterministic paper
normalization followed by classical NLM, while the learned network only
controls the NLM bandwidth h.
