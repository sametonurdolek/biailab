# Architecture comparison

These comparisons are the reason the ten scripts are separate. They are questions, not outcomes.

## A vs B — discrete vs continuous

EXP01 picks one of five frozen tones per cell and blends only the decision boundary. EXP02 emits a black map and a white map, so a pixel can sit between the frozen presets. If EXP02 cannot beat EXP01, the extra coordinates are not helping optimization. If EXP01 cannot beat fixed S05 V03, the bank itself is the limit.

## B vs C — black/white vs gain/bias

EXP02 and EXP03 implement the same family, `clip((Z-b)/(w-b)) = clip(aZ+c)`. EXP02 constrains the endpoints. EXP03 constrains gain and bias with the axis-aligned rectangle implied by that box (`gain` in `[1, 2.5]`, `bias` in `[-0.75, 0]`). A point inside one set is not always inside the other. The comparison asks which coordinates AdamW handles, not which formula is more expressive in the abstract.

## B vs D — two maps vs one scalar

EXP04 forces `(black, white)` through a measured curve `b(t), w(t)`. Until BENCH_03 writes that curve, the script stops. A hand-drawn curve would answer a different question.

## E vs B — global vs spatial

EXP05 has two numbers per page. EXP02 has two maps. BENCH_02 put the page-oracle gap above the best frozen preset near 0.414 dB and the spatial gap near 0.939 dB on the development pages. EXP05 tests whether the smaller gap is learnable. It should be read against the page oracle, not against the spatial hybrid.

## F / G / B — G32 vs G64 vs G128

EXP06, EXP07, and EXP02 share the black/white parameterization, the loss, and the default input `z`. Grid size is the intended difference. Bilinear expansion means a coarse grid is a smooth field, not a blocky mosaic.

## B vs H — one channel vs `[x, Z]`

EXP02's `INPUT_MODE` runs the same head on `x`, on `Z`, or on both. EXP08 is the dual-input run with that choice fixed at `xz`, so its Drive folder cannot be confused with an EXP02 folder. `x` is the degradation the normalization has not removed. `Z` is the photometric state after the frozen operator.

## A vs I — direct classification vs distillation

EXP01 and EXP09 use the same bank, the same hybrid deployment, and the same backbone. EXP09 stays on the oracle teacher longer before image loss. A gain that appears only in the soft training mixture and disappears under hybrid validation is not a gain.

## Also in the set

EXP10 asks whether cell-wise lattice targets are a usable teacher for the continuous head. The clean page is used while those targets are built and is not used at inference.
