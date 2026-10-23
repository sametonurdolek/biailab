# Final decision

Classification: CONFIRMED_FIXED_OPERATOR_GAIN and CONFIRMED_CONTROL_HEADROOM

```text
BENCH_02 STATUS:
CONFIRMED_FIXED_OPERATOR_GAIN and CONFIRMED_CONTROL_HEADROOM

HOLDOUT-21 RESULT:
GO; passers: S05_V03, S05_V05

DEVELOPMENT-123 RESULT:
RUN on all 123 pages. Frozen presets. No retune.

CURRENT6 HYBRID:
PSNR: 26.086509
SSIM: 0.947533

BEST CONFIRMED FIXED TONE OPERATOR:
System: S05
Variant: S05_V03
PSNR: 27.593155
SSIM: 0.971745
ΔPSNR: 1.506646
ΔSSIM: 0.024212
Page win rate: 116/123

S03 / S05 MATHEMATICAL RELATIONSHIP:
Same post-normalization affine family. S05 is the two-endpoint parameterization. S03 is an unconstrained gain/bias grid. No frozen preset pair is numerically identical. See S03_VS_S05_EQUIVALENCE.md.

WHAT DEGREE OF FREEDOM WAS ACTUALLY DISCOVERED?
A bidirectional affine tone map after the frozen s=31 closing-division. Black and white endpoints are one parameterization of that affine.

S05 FAMILY GLOBAL ORACLE:
PSNR: 28.240600
SSIM: 0.975417

S05 FAMILY G128 HYBRID ORACLE:
PSNR: 28.765459
SSIM: 0.978669

S05 SPATIAL HEADROOM ABOVE BEST FIXED:
0.939269 dB above S05_V04

S19 BEST FIXED:
S19_V04 PSNR 26.461757 SSIM 0.966453

S19 G128 HYBRID ORACLE:
PSNR 28.911216 SSIM 0.980141

S19 SPATIAL HEADROOM:
2.449459 dB

STRUCTURAL GUARDS:
PASS
Reason: ink and edge MAE ratio versus Current6 hybrid, limit 1.25, plus mean SSIM delta > 0. Severe-page fraction for S05_V03 is 0.008.

MINIMAL CONFIRMED OPERATOR:
S05_V03: canonical s=31 closing-division, then the frozen affine/endpoint preset.

HIGH-CAPACITY CONFIRMED OPERATOR:
S19 S19_V04 fixed, spatial oracle 28.911216 dB, headroom 2.449459 dB.

DOES BACKGROUND ESTIMATION STILL LOOK LIKE THE MAIN BOTTLENECK?
NO
Reason: The confirming gain, where present, is a tone stage on the unchanged s=31 background. S19 is a separate operator family, reported as capacity, not as a replacement background search.

DOES POST-NORMALIZATION TONE CAPACITY LOOK LIKE THE MAIN BOTTLENECK?
YES
Reason: S03 and S05 share the affine form Y = clip(a Z + b) after Z = current_s31(x). Confirmation is judged on the 123, not on the discovery 10.

IS A FIXED IMPROVED OPERATOR ALREADY ENOUGH TO JUSTIFY IMPLEMENTATION STUDY?
YES

IS A NEW LEARNED CONTROLLER JUSTIFIED YET?
YES

IF YES, WHAT SHOULD IT PREDICT?
The affine controls (gain and bias, or equivalently black and white) at page level if one preset dominates area, and at G128 cells if several presets each hold more than 5% of area. S19 sigma and gain are a separate high-capacity target only if that family's spatial headroom stays material. This bench does not train it.

IF NO, WHAT MUST BE TESTED NEXT?
An implementation study of the confirmed fixed tone stage, or BENCH_03 if a denser affine grid is wanted. Do not train on the 10-page slice.

TEST SPLIT USED:
NO
```

## Predeclared primary candidates on the 123

S05 V03 ΔPSNR 1.506646 (CI 1.220047, 1.814695), ΔSSIM 0.024212, wins 116/123, severe fraction 0.008.
S05 V05 ΔPSNR 1.278969 (CI 0.878069, 1.705975), ΔSSIM 0.025079, wins 86/123, severe fraction 0.081.

## Reading notes

These notes do not change the predeclared gate or the frozen presets.

S05 V04 was scored with the other frozen presets. On the 123 it has the highest S05 PSNR, 27.826 dB, Δ +1.740, SSIM 0.9747, 107/123 wins, severe fraction 0.024, structural PASS. It was not one of the four gate candidates, so it is not promoted. Promoting it on the confirmation split would be a new selection.

Among the four predeclared candidates, S05 V03 leads PSNR and has one severe page in 123. S05 V05 has a slightly higher SSIM and ten severe pages. S03 V04 missed the holdout gate because its 95% CI included zero. On the 123 its mean gain is +0.485 dB.

S04 black-only reaches a PSNR close to S05 V05 and loses SSIM. Background MAE on the 123 is 0.0153 for S04 V04 and 0.0020 for S05 V03, against 0.0086 for Current6 hybrid. The white endpoint is what keeps the paper from being pulled down.

S19 V04 fixed gain is +0.375 dB and the 95% CI includes zero. The S19 result that survives is the spatial oracle, 28.911 dB, 2.449 dB above its best frozen preset. That is conditional capacity. It is not a confirmed fixed operator.

S05 page-level oracle headroom above the best frozen preset is 0.414 dB, below the predeclared 0.50 dB line. The spatial gap, 0.939 dB, is above that line. A learned controller is a spatial question only. Nothing was trained.

# Research questions

1. S05 V05 on the 21 unseen calibration pages: mean ΔPSNR +1.559 dB (95% CI +0.716 to +2.444), wins 15/21, worst page -1.754 dB. The discovery-10 recomputed ΔPSNR was +2.543 dB. The +2.54 dB exploratory figure does not survive at that size. A smaller positive gain does: the holdout gate passed for S05 V03 and S05 V05.

2. On all 123 development-eval pages, S05 V05 mean ΔPSNR is +1.279 dB (95% CI +0.878 to +1.706), median +0.649 dB, page wins 86/123.

3. S05 V05 mean ΔSSIM on the 123 is +0.0251 (95% CI +0.0174 to +0.0340), SSIM wins 111/123. S05 V03 ΔSSIM is +0.0242.

4. Structural guard for S05 V05 is PASS. For S05 V03 it is PASS. The guard requires mean ink MAE and edge MAE at or below 1.25 times Current6 hybrid, and mean ΔSSIM > 0. Severe-page fraction (ΔPSNR < -1 dB or ΔSSIM < -0.05) is 0.081 for V05 and 0.008 for V03. Ink MAE: hybrid 0.1786, V03 0.1581, V05 0.1493. Edge MAE: hybrid 0.1409, V03 0.1254, V05 0.1203.

5. On the 123, V05 PSNR 27.365 versus V03 PSNR 27.593. V05 SSIM 0.9726 versus V03 SSIM 0.9717. The higher mean PSNR is V03. V05 severe fraction 0.081 versus V03 0.008. V05 is the more aggressive endpoint pair. It is not uniformly safer.

6. V05 ΔPSNR correlates with dirty mean at Pearson -0.298, with dirty p95 at -0.114, and with low-frequency range at 0.183. No development page is as bright as discovery Q03. Page-level detail is in S05_FAILURE_REGIMES.md. The preset was not changed.

7. Yes. S05 is Y = clip((Z - black) / (white - black)). That is the affine Y = clip(gain * Z + bias) with gain = 1/(white-black) and bias = -black/(white-black). S03 is the same equation on an unconstrained (gain, bias) grid. No frozen S03 point equals a frozen S05 point. See S03_VS_S05_EQUIVALENCE.md.

8. One degree of freedom family: a bidirectional affine tone map after the frozen s=31 closing-division. BENCH_01 did not find two independent winners. It sampled that affine family twice, once as endpoints and once as gain and bias.

9. Black-only S04 V04 (black 0.18, white fixed at 1) mean PSNR on the 123 is 27.365 (Δ +1.278 dB versus Current6 hybrid 26.087). S04 V03 (black 0.10) is 27.270 (Δ +1.184 dB). A black endpoint alone moves the score. It is not the whole S05 gain.

10. At the same black point 0.10, S05 V03 (white 0.92) PSNR 27.593 versus S04 V03 (white 1) PSNR 27.270. The white endpoint is the extra degree of freedom. S05 V05 (black 0.22, white 0.82) reaches 27.365.

11. On pixels with T < x - 1/255, mean per-page MSE is 0.06872 for Current6 hybrid, 0.06993 for fixed s=31, 0.05612 for S05 V03, and 0.04628 for S05 V05. Mean SSE share of that region is 0.656 for hybrid and 0.433 for S05 V05. On T > x + 1/255 the MSE is 0.00096 for hybrid and 0.00121 for S05 V05. The tone stage is able to write values below Z, which the plain division cannot.

12. The measured gain uses the unchanged s=31 closing and the unchanged division. Only the map after Z changes. Background estimation was not replaced. The gain is a photometric output-space effect on top of the current background.

13. S05 per-page oracle choices on the 123: V01 6, V02 1, V03 36, V04 39, V05 41. Mean dominant-cell fraction 0.707, mean entropy 1.263 bits, mean number of presets above 5% area 2.86. Global-oracle headroom above the best frozen preset is 0.414 dB. The predeclared material line is 0.50 dB.

14. S05 per-page five-preset oracle headroom above best fixed is 0.414 dB (best fixed S05_V04, oracle PSNR 28.241).

15. S05 G128 hybrid oracle PSNR 28.765, spatial headroom above best fixed 0.939 dB. Hard oracle PSNR 28.697.

16. After one frozen tone is applied to every Current6 scale: S05_V03: fixed 27.593, page-s oracle 27.810 (Δ +0.217), spatial hybrid 27.929 (Δ +0.336); S05_V05: fixed 27.365, page-s oracle 27.630 (Δ +0.265), spatial hybrid 27.672 (Δ +0.307); S03_V04: fixed 26.572, page-s oracle 26.861 (Δ +0.289), spatial hybrid 26.869 (Δ +0.297); S03_V05: fixed 25.943, page-s oracle 26.212 (Δ +0.268), spatial hybrid 26.219 (Δ +0.275). This is the existing six-scale bank, not a new joint grid.

17. S19 G128 hybrid oracle is 30.100 dB on the recomputed discovery 10, 30.413 dB on holdout-21, and 28.911 dB on the 123. Best frozen S19 preset on the 123 is S19_V04 at 26.462 dB.

18. S19 best fixed Δ versus Current6 hybrid on the 123 is +0.375 dB. Its spatial headroom above that fixed preset is 2.449 dB. The 30 dB discovery oracle does not carry over as a deployment score. What carries is whether the spatial gap above the best frozen preset stays large. See the decision block.

19. The minimal confirmed operator, if the decision block names one, is canonical s=31 closing-division plus one frozen affine/endpoint preset. It is not a new background estimator and not S19.

20. A learned controller is justified only when a confirmed family's oracle sits at least 0.50 dB above its best frozen preset. That comparison is in the decision block. Justification here means a later estimator study. This bench does not train one. Oracle label entropy is conditional capacity, not evidence that a network can learn the map.
