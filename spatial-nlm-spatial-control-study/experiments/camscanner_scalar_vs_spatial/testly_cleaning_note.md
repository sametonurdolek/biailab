# Testly camera cleaning semantics — local source evidence, 2026-10-14

Inspected file: `C:\testly\testly_backend\backend\services\pipeline_cuda.py`.
This describes the local source checkout, not a verified running deployment.
No Testly modules were imported or executed, and no Testly file was modified.

## Reachable camera path

`process_question_image()` starts at line 611. Input is BGR uint8. The longest
edge is capped at 1600 with area interpolation (helper at 264, call at 662).
Origin overrides / rotation classification choose camera versus digital.
Camera dewarping starts at 866, calling `LinearSmartDewarper.dewarp()` or
`dewarp_with_orientation()` depending on the orientation mode. A successful
dewarped page is the common `stage2_input`; unavailable/failed dewarping keeps
the original. This is a geometry operation before restoration, not denoising.
The experiment substitutes its own deterministic offline pair preparation.

At lines 935–999, the hybrid restoration path:

1. detects figure boxes on the page;
2. restores the full page with a binarizer;
3. restores each figure crop separately;
4. pastes the figure crops onto the restored full page.

At lines 1003–1014, a successful composed output is assigned directly to final.
The seamless blend is a **fallback** after hybrid failure, not an extra stage
applied to every successful binarizer output. `skip_enhancement` returns the
dewarped stage2 input directly.

## Modules read and observed behavior

* `enhancement/binarizer_pth_cuda_infer.py`: UnetPlusPlus, EfficientNet-B5,
  3 input and 3 output channels, sigmoid. RGB conversion and /255 normalization,
  aspect preserving resize + white pad to 1024 square, float inference,
  [0,1] clip, uint8 conversion, unpad, resize back. Despite its name, this
  returns continuous RGB restoration; there is no final hard binary threshold.
  Its stated output goal is clean text on a white background.
* `enhancement/figure_detector_cuda.py`: YOLO figure-class boxes, confidence
  .60, IoU .45, inference size 640, 4px box expansion. Boxes route figure
  content to its dedicated restoration rather than treating every pixel as text.
* `enhancement/figure_nafnet_pth_cuda_infer.py`: NAFNet width32, figure-only
  crops resized to 320 square, RGB /255, residual image output and clip,
  resized back. The module records a training contract with Charbonnier,
  Sobel edge and white-background terms; those are source comments about
  production model training, not experiments performed here.
* `mask_rescue.py`: reached only through the hybrid fallback. Includes
  LAB-L masks, background normalization, blur-dependent mask parameters,
  connected-component/coverage quality checks and rescue choice. At 1033,
  background normalization stretches p2/p98 and applies gamma (default .7),
  with a bright-background skip. At 1607–1648 the selected soft text mask
  blends LAB foreground onto white (L=255, neutral chroma), then enforces
  white again. At 1658 onward, optional ink-black boost darkens text; an
  optional min-merge attempts to preserve original dark detail.
* `seamless_blend.py`: inspected related legacy/delegating blend entry point;
  the CUDA pipeline directly calls `mask_rescue.make_seamless_with_rescue`.
* `config_cv.py`: inspected the seamless and pipeline configuration classes.
  The configured background brightness is 260, but the current rescue blend
  explicitly forces 255 in its reached blend code; source behavior controls
  the interpretation.
* `dewarp/linear_smart_dewarper.py`: inspected the reached perspective-warp
  sections of `dewarp` and `dewarp_with_orientation`. Segmentation-derived
  corners are scaled back to the original/upright resolution, expanded around
  their centroid, and passed through `getPerspectiveTransform` / linear
  `warpPerspective` with white borders. This is why geometry preparation is
  treated as a common stage ahead of document restoration.

## Transformation being sought

The source seeks a geometrically usable question page with clean white paper,
strong text/background separation, retained text detail and restored figures.
Noise suppression alone cannot explain this target: illumination, paper texture,
and contrast change too. Figure routing is evidence that diagrams need different
treatment from a page-wide text cleaner. The CamScanner pairs visibly retain
colored drawings, highlights and handwritten marks; this experiment scores
grayscale luminance only and does not claim color restoration.

## How this informs the experiment

The deterministic candidate set compares identity processing, global percentile
contrast stretching and local paper-brightness division with a fixed black-level
stretch. Validation chooses one shared candidate. All parameters and formulas
are in `document_pipeline.py`. Each image's statistics are computed by the same
fixed rule; no per-image hand tuning occurs. Outputs remain continuous, and
postprocessing is only [0,1] clipping, which keeps gradients through NLM.

Production binarizer, NAFNet, YOLO figures, rotation classifier, learned dewarper,
mask rescue and compositing are not part of either experimental branch. The only
learned restored-image control is `h`; the sole restoration estimator is the
copied classical NLM. This is a controlled document experiment informed by Testly
cleaning goals, not a reproduction or evaluation of Testly's production cleaner.
