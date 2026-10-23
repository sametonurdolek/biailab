# Architecture matrix

Nothing in this table has been trained. BENCH_03 was absent, so the discrete bank is the frozen BENCH_01 S05 catalog and the continuous box is the documented fallback.

| ID | Architecture | Output | Grid | Input | Supervision | Main question |
|---|---|---|---:|---|---|---|
| EXP01 | Discrete tone classifier, hybrid deployment | K class logits | 128 | x | Oracle class, then ramped image loss | Can the frozen five-preset bank be spatially classified? |
| EXP02 | Continuous black/white | black, white | 128 | z (switch x / z / xz) | Direct image loss | Can black and white be regressed as maps? |
| EXP03 | Continuous gain/bias | gain, bias | 128 | z | Direct image loss | Is gain/bias an easier geometry than black/white? |
| EXP04 | Scalar field on a measured manifold | t | 128 | z | Direct image loss | Does one measured curve carry the tone? Blocked until BENCH_03. |
| EXP05 | Page-level black/white | one black, one white | 1 | x | Image, or page-lattice regression | Are two page numbers enough? |
| EXP06 | Continuous black/white | black, white | 32 | z | Direct image loss | Is a coarse smooth field enough? |
| EXP07 | Continuous black/white | black, white | 64 | z | Direct image loss | Where does grid resolution stop helping? |
| EXP08 | Continuous black/white, dual input | black, white | 128 | xz | Direct image loss | Does [x, Z] beat either channel alone? |
| EXP09 | Discrete classifier, longer distillation | K class logits | 128 | x | Oracle distillation, then image loss | Does a staged teacher beat EXP01? |
| EXP10 | Continuous black/white | black, white | 128 | z | Cell-lattice parameter regression | Can offline (b*, w*) maps be regressed? |

K is 5 while the frozen S05 file is unchanged. Grid 1 means one pair of numbers for the page, not a 1×1 feature map that is later upsampled.

Recovery for EXP05 uses the best frozen S05 preset on the whole page. Recovery for the spatial experiments uses the target-chosen G128 hybrid of those same five presets. A continuous model is allowed to exceed that reference. The ratio is a ratio of PSNR gaps. PSNR is logarithmic in MSE, so the ratio is not a linear fraction of visual quality.
