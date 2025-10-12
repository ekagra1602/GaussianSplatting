# Spectral-GS Implementation

Implementation of **Spectral-GS: Taming 3D Gaussian Splatting with Spectral Entropy** (SIGGRAPH Asia 2025).

Paper: [arXiv:2409.12771](https://arxiv.org/abs/2409.12771)

## Overview

This implementation extends standard 3D Gaussian Splatting with spectral entropy-based densification to reduce needle-like artifacts and improve reconstruction quality.

### Key Features

1. **Spectral Entropy Computation**: Analyzes the covariance matrix of each Gaussian to detect needle-like shapes
2. **Shape-Aware Splitting**: Splits Gaussians with low spectral entropy (< τ) even if gradients are low
3. **Anisotropic Covariance Reduction**: Reduces scales intelligently during splitting to maintain valid shapes
4. **2D View-Consistent Filtering**: Optional post-processing to reduce aliasing artifacts

## Architecture

```
src/spectral_gs/
├── __init__.py              # Package exports
├── spectral_entropy.py      # Core entropy computation & splitting logic
├── spectral_strategy.py     # Custom densification strategy
└── filtering.py             # 2D view-consistent filtering
```

## Usage

### Basic Training

```bash
bash scripts/run_spectral_gs.sh data/campus_scene results/spectral_gs
```

### Advanced Options

```bash
python scripts/train_spectral_gs.py \
  --data_dir data/campus_scene \
  --result_dir results/spectral_gs \
  --max_steps 10000 \
  --spectral_threshold 0.5 \
  --enable_spectral_splitting \
  --enable_filtering \
  --filter_type gaussian \
  --verbose
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--spectral_threshold` | 0.5 | Entropy threshold τ for needle detection |
| `--enable_spectral_splitting` | True | Enable spectral-aware splitting |
| `--spectral_split_factor` | 1.6 | Scale reduction factor during splitting |
| `--enable_filtering` | False | Enable 2D view-consistent filtering |
| `--filter_type` | gaussian | Filter type: gaussian, box, or combined |

## Algorithm Details

### Spectral Entropy

The spectral entropy of a Gaussian's covariance matrix Σ is:

```
H(Σ) = -∑(sᵢ²/tr(Σ) * ln(sᵢ²/tr(Σ)))
```

where sᵢ² are the eigenvalues of Σ.

**Key insight**: For Σ = RSS^TR^T, the eigenvalues are simply the squared scales, making computation efficient.

**Interpretation**:
- High entropy (→ log(3) ≈ 1.1): Isotropic Gaussian (sphere-like)
- Low entropy (< 0.5): Anisotropic Gaussian (needle-like)

### Splitting Criteria

A Gaussian is split if:
1. **Gradient-based** (from original 3D-GS): High 2D positional gradient + large 3D scale, OR
2. **Spectral-based** (new): Low spectral entropy (< τ) + visible opacity

This catches needle-like Gaussians that don't accumulate high gradients.

### Anisotropic Splitting

When splitting:
1. Reduce all scales by factor (1.6 by default)
2. Sample new positions along principal axis (largest scale direction)
3. Distribute opacity among children

This prevents creating even more extreme needle shapes.

## Implementation Notes

### Integration with gsplat

The `SpectralStrategy` extends gsplat's `DefaultStrategy`:
- Inherits standard gradient-based densification
- Adds spectral entropy computation in `step_post_backward`
- Uses gsplat's `split()` operation with custom parameters

### Efficiency

- **Spectral entropy**: O(N) computation (no matrix decomposition needed due to RSS^TR^T form)
- **Minimal overhead**: Spectral splitting runs every `refine_every` steps
- **Memory efficient**: Uses gsplat's in-place parameter updates

## Comparison with Standard 3D-GS

| Feature | Standard 3D-GS | Spectral-GS |
|---------|----------------|-------------|
| Splitting criterion | 2D gradient only | 2D gradient + spectral entropy |
| Needle artifacts | Common in sparse views | Significantly reduced |
| Densification | Gradient-based | Shape-aware |
| Training time | Baseline | +5-10% overhead |

## Troubleshooting

### Too many splits occurring
- Increase `spectral_threshold` (try 0.6 or 0.7)
- Increase `spectral_split_factor` (try 2.0)

### Not enough needle reduction
- Decrease `spectral_threshold` (try 0.3 or 0.4)
- Increase `refine_every` frequency

### Memory issues
- Increase `prune_opa` to prune more aggressively
- Decrease `max_steps` or `refine_stop_iter`

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{huang2025spectralgs,
  title={Spectral-GS: Taming 3D Gaussian Splatting with Spectral Entropy},
  author={Huang, Letian and Guo, Jie and Dan, Jialin and Fu, Ruoyu and Wang, Shujie and Li, Yuanqi and Guo, Yanwen},
  booktitle={SIGGRAPH Asia},
  year={2025}
}
```

## License

This implementation is provided for research purposes. Please refer to the original paper's license and gsplat's Apache 2.0 license.
