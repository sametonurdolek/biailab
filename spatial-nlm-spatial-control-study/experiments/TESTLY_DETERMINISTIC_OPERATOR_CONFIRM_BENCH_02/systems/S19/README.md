# S19

Imported from BENCH_01. Parameters are in `frozen_candidates.json`. This folder does not contain a second implementation.

```
R = log(x+eps) - log(G+eps). Y = clip(1 + gain * R). eps=1/255. This is not exp(R), which would collapse to x/G (S06).
```
