# Loss design

Weights are stored in `config.json` as `lambda_mse`, `lambda_ssim`, `lambda_edge`, and `lambda_smooth`. Discrete runs also store `lambda_kl` and `lambda_regret` on the active schedule row.

## Presets

| Name | mse | ssim | edge | smooth |
|---|---:|---:|---:|---:|
| `mse` | 1 | 0 | 0 | 0 |
| `mse_ssim` | 1 | 0.2 | 0 | 0 |
| `mse_ssim_edge` | 1 | 0.2 | 0.1 | 0 |
| `mse_structural_smooth` | 1 | 0.2 | 0.1 | 0.01 |

The first run of every continuous script uses `mse`. PSNR, which selects checkpoints, is a monotone function of MSE. The other presets are there for a later run, not as a hidden default.

`mse` is the mean square error of the toned page. `ssim` is one minus a Gaussian SSIM with an 11×11 window, sigma 1.5, and the usual constants `0.01²` and `0.03²`, cropped by 5 pixels when the page is large enough. `edge` is the L1 gap of Sobel gradients scaled by 1/8. `smooth` is total variation of the control maps at grid resolution: black and white, or gain and bias, or the scalar `t`. It is not total variation of the page. The default smooth weight is small because a strong penalty collapses the map toward EXP05.

## Clamp

`CLAMP_BACKWARD = "exact"` is the default. Values outside `[0, 1]` receive no gradient.

`straight_through` uses the same clip in the forward value and passes the incoming gradient through unchanged. Validation and any exported page ignore this switch and use the exact clip.

## Discrete losses

KL is the area-weighted divergence from `q = softmax(-regret / tau)`. Expected regret is area-weighted too, and the regret tensor is divided by tau before the weight is applied. Both use the cell-area weights from the rounded grid, so a short edge cell does not count as a full cell.

The image term is off during the teacher stages. When it turns on, class probabilities are bilinearly resized to the page, renormalized, and mixed across the five tone images. Hybrid compose is not inside the backward graph.

## Oracle-parameter loss

EXP10, and EXP05 when `SUPERVISION` is `oracle_param`, use the mean square error of the predicted controls against the cached lattice targets. Image loss stays off unless `IMAGE_WITH_ORACLE` is set. The checkpoint is still chosen by page PSNR, not by parameter MSE.

## What the loss is not

The clean page never enters the network. Masks used by the structural guard are computed from the clean page for evaluation only. There is no adversarial term and no pixel generator.
