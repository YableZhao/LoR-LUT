# PPR10K-a Experiments Log

Tracking val PSNR / SSIM / DE2000 across different configurations on PPR10K expert-a.
Group-based train/val split (cutoff group_id=1345 → 8792/2369 in non-aug, 52752/2369 with aug).

## Comparison targets

| Method | Params | PSNR | DE_ab |
|---|---|---|---|
| 3D-LUT (PPR10K paper baseline) | 590K | 25.64 | 6.97 |
| AdaInt | 620K | 26.33 | 6.56 |
| SepLUT-S (lightweight) | 47K | 26.19 | 6.43 |

## Our runs

### Run 1: K=0 R=4, no aug, patch=0, batch=1 (initial baseline)
- **Best val PSNR: 22.64 @ 200k** (also: SSIM 0.9254, DE2000 7.418)
- 200k iters in 39 min on Colab L4
- Plateau immediately, R=4 too low capacity

### Run 2: K=0 R=8, no aug, batch=8, patch=256, 1M iters
- **Best val PSNR: 23.46 @ 900k** (peaks around 600-900k)
- Final 1M val: 23.42 / SSIM 0.9296 / DE2000 6.813
- ~9h on T4
- Plateau at 23.4 from 600k onwards
- **Diagnosis**: training data 1/6 of paper baseline (no augmentation)

### Run 3: K=0 R=8, **WITH aug** (52K samples), batch=8, patch=256, 1M iters
- **Best val PSNR: 23.51 @ 600k** (SSIM 0.9306, DE2000 6.717)
- ~15h on T4
- Plateau at 23.5 from 300k+
- **Diagnosis**: 6x more training data didn't significantly help. Bottleneck is elsewhere.

### Run 4: K=0 R=8, aug + **16-bit TIFF fix** (imageio replaces PIL.convert)
- **Best val PSNR: 23.58 @ 600k** (SSIM 0.9322, DE2000 6.681)
- ~5.5h on T4 (faster, imageio better)
- Plateau at 23.5-23.6 from 300k+
- **Diagnosis**: 16-bit precision was a real bug (verified ~118x precision loss in audit) but only +0.07 dB
  → 16-bit not the dominant bottleneck

## Observations

1. **All K=0 R=8 configs plateau at 23.5 PSNR** regardless of augmentation, batch size, patch size, bit precision
2. **Mild overfitting after 600k iters** (val PSNR slightly decreases)
3. **Gap to 3D-LUT baseline: ~2 dB** stable across configurations
4. **Likely remaining bottlenecks** (in order of suspected impact):
   - Predictor encoder too small (16→32 CNN vs ResNet18 in baselines)
   - K=0 capacity insufficient for portrait diversity (try K=8 R=8 or K=0 R=32)
   - Missing ΔE_ab loss (PPR10K-specific)
   - Predictor outputs lack softmax/tanh activations (paper docstring vs code mismatch)

## Next: Run 5 — K=0 R=32

Same setup as Run 4 but R=8 → 32 (capacity 32K → 113K params, matches LoR-LUT paper's "best" FiveK config).

Hypothesis: portrait complexity needs more rank-1 components than landscape (FiveK).
Expected: 24-25 PSNR if bottleneck is capacity. If still ~23.5, capacity isn't the issue.
