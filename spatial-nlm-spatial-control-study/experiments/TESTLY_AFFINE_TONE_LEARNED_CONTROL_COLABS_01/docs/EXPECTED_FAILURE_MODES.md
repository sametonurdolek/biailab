# Expected failure modes

These are failure modes to look for after a run. They are not measurements.

## MANIFOLD_CONFIG_REQUIRED

EXP04 prints this and writes `STOP.json` when `configs/manifold.json` is missing or does not have status `measured_from_bench03`. The example file in `configs/` is supposed to fail this gate. No PSNR is invented for that stop.

## No promotable checkpoint

If every calibration epoch fails the ink, edge, or SSIM guard, `best.pt` is absent and the summary says `NO_PROMOTABLE_CHECKPOINT`. `latest.pt` is still there for inspection. That is a failed controller, not a successful constant model. Development target caches are not built in that case.

## Global collapse

A page whose black and white maps both have standard deviation below `1e-3` is flagged. EXP05 is constant by construction, so its flag is `expected_for_global_architecture`. On a spatial model the flag is reported and does not by itself reject the checkpoint. The page-level oracle gap is small enough that a nearly constant map can be the right answer on some pages.

## Saturated tone

The summary tracks the fraction of output pixels within `1e-6` of 0 or 1. A large fraction with a poor ink MAE means the clip is eating the gradient. The next run of that same script can set `CLAMP_BACKWARD = "straight_through"`. Validation remains the exact clip, so a surrogate that only looks good in the backward pass will not be selected on its training loss.

## Denominator collapse

The endpoint map refuses spans below `min_span`. If training loss goes to NaN anyway, the usual cause is an AMP overflow in the backbone rather than `white - black`. The operator path is float32. A NaN in `train_loss` with a finite calibration PSNR means the logged training term diverged while the eval clip stayed finite.

## Tau

If every calibration cell has the same best and second-best cost, tau calibration raises. That would mean the frozen bank has no local decision on the calibration pages. It is a data/bank failure, not a reason to type in a constant.

## Cache identity

`Z_CACHE_MISMATCH`, `COST_CACHE_MISMATCH`, `ORACLE_CACHE_MISMATCH`, `SPLIT_HASH_MISMATCH`, `DIRTY_SHA_MISMATCH`, and `CLEAN_SHA_MISSING` stop the run. Deleting the shared cache and letting the script rebuild it is the recovery. Do not point a new operator spec at an old `z_s31` directory by renaming files.

## Resume signature

`CHECKPOINT_SIGNATURE_MISMATCH` means `RESUME = True` found a `latest.pt` from a different input mode, loss, grid, supervision, or other signed field. The run stops and does not load those weights. Set `RESUME = False` to keep that folder and start a suffixed directory, or point `DRIVE_ROOT` at a fresh experiment folder.

## Over-reading the recovery ratio

A ratio above 1 means the learned page beat the frozen-bank reference. For a continuous model that can be legitimate, because the bank has only five tones. A ratio below 0 means the learned page is worse than fixed S05 V03. A missing ratio means the oracle did not clear the fixed score by more than `1e-6` dB, so there was no headroom to recover. None of these ratios are a percentage of visual quality.

## Test split

Any attempt to include a test id raises `TEST_SPLIT_TOUCHED`. There is no flag that turns the test split on. `ALLOW_TEST_SPLIT = True` aborts before discovery.

## Warm start

`WARM_START = True` with a missing file, the wrong sha256, an epoch other than 54, or Current6 `s_values` other than `(15, 21, 31, 45, 63, 91)` stops the run. A two-channel input loads every encoder tensor whose shape matches and says which names were skipped. A one-channel full backbone requires every encoder tensor.

## Structural regression that still raises PSNR

S04 in BENCH_02 gained PSNR and lost background SSIM behavior. The guard exists so a checkpoint with that pattern is not promoted. The development report still says whether the selected checkpoint also passes the guard on the 123 pages. Selection itself does not look at those 123 pages.
