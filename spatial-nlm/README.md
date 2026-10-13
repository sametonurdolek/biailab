# Spatial bandwidth for classical Non-Local Means

Question: can a learned map h(x, y) improve Non-Local Means without replacing the algorithm?

The estimator is unchanged from Buades, Coll, and Morel (CVPR 2005):

```text
D(i, j) = mean over the patch of (u(i+k) - u(j+k))^2
w(i, j) = exp( - D(i, j) / h(i)^2 )
restored(i) = weighted average of u(j) inside the search window
```

Patch window 7, search window 21. A fixed bandwidth is the same function with a constant map. The network never emits pixels. Its output is clipped to the validation sweep range, so it cannot win by leaving the range the constant baseline was allowed to use. Because the weights are positive and sum to one, every restored pixel stays inside the convex hull of its search window.

## Controls

Chosen on Kodak kodim13–16, then frozen. kodim17–24 is scored once.

| Method | What is free | Sees the clean image at test time? |
|---|---|---|
| fixed | one h for every image | no |
| hand | h(i) = clip(k · local noise estimate) | no |
| scalar | a CNN that emits one h per image | no |
| spatial | the same CNN, emitting h(i) | no |
| oracle constant h | best constant h per observation | yes |
| oracle true sigma | clip(k · the sigma field that generated the noise) | yes, and it sees the noise field |

The scalar network is the control that matters. If it matches the spatial network, the gain is "a better single h," not spatial control.

## Model

Both networks share one trunk: a 3×3 stem to 32 channels and four residual blocks of two 3×3 convolutions. The spatial head is 1×1 and emits h(i). The scalar head global-pools the same features and emits one h per image. The output is mapped into the validation sweep range with a sigmoid, so the network cannot leave the interval the constant baseline was allowed to use. Training is Adam on L1 to the clean crop, 400 steps, checkpoint chosen by validation PSNR. One seed. The test split is not used to pick h, k, the checkpoint, or the step count.

## Two noise families

- Stationary: Gaussian sigma = 25/255 everywhere.
- Spatial: a linear ramp from 15/255 to 45/255. Axis and direction are random on every crop, so the network cannot memorize "the right half is noisier."

## Run

```text
python run_experiment.py
```

`nlm.py` checks the vectorized operator against a direct implementation before anything else. Scores land in `results/metrics.json`. The pictures are `results/figure_awgn_kodim17.png` and `results/figure_spatial_kodim17.png`.

This folder is not part of the Testly pipeline and is not imported by it.
