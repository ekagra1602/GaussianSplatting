# Running Spectral-GS in Google Colab

This guide shows how to use Spectral-GS in your existing Colab notebook.

## Quick Start

The Spectral-GS implementation is fully integrated and ready to use. All compatibility fixes for Colab (gsplat 1.5.3+, pycolmap-free dataset loading, gradient flow handling) are built into `scripts/train_spectral_gs.py`.

### 1. Update Your Repository

First, commit and push the Spectral-GS code to your GitHub repo:

```bash
# On your local machine
cd /Users/ekagragupta/Desktop/Projects/GaussianSplatting
git add src/spectral_gs/ scripts/train_spectral_gs.py scripts/run_spectral_gs.sh
git commit -m "Add Spectral-GS implementation with Colab compatibility"
git push origin main
```

### 2. Add These Cells to Your Colab Notebook

#### Cell: Install Spectral-GS Dependencies

Add this cell after your existing pip install cell:

```python
# Install Spectral-GS dependencies
!pip install scipy

print("✅ Spectral-GS dependencies installed!")
```

#### Cell: Verify Spectral-GS Installation

Add this verification cell to confirm everything is working:

```python
# Verify Spectral-GS modules are available
import sys
sys.path.insert(0, '/content/GaussianSplatting/src')

try:
    from spectral_gs import compute_spectral_entropy, SpectralStrategy, apply_view_consistent_filter
    print("✅ Spectral-GS successfully imported!")

    # Quick test
    import torch
    test_scales = torch.randn(10, 3).abs() + 0.1
    test_quats = torch.randn(10, 4)
    test_quats = test_quats / torch.norm(test_quats, dim=-1, keepdim=True)

    entropy = compute_spectral_entropy(test_scales, test_quats)
    print(f"   Test entropy computation: mean = {entropy.mean():.3f}")
    print("   ✓ All components working!")

except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\n📝 Troubleshooting:")
    print("1. Make sure you pushed spectral_gs code to your GitHub repo")
    print("2. Run 'git pull' to update your Colab repository")
    print("3. Check that /content/GaussianSplatting/src/spectral_gs/ exists")
```

#### Cell: Run Spectral-GS Training

Replace your current training cell with this:

```python
%cd /content/GaussianSplatting

print("🚀 Starting Spectral-GS Training...")
print("=" * 60)

# Run Spectral-GS training (all Colab compatibility built-in)
!python scripts/train_spectral_gs.py \
  --data_dir /content/GaussianSplatting/lantern_ds \
  --result_dir /content/GaussianSplatting/results/lantern_spectral_30k \
  --max_steps 30000 \
  --data_factor 1 \
  --spectral_threshold 0.5 \
  --enable_spectral_splitting \
  --save_ply \
  --log_every 100 \
  --save_every 5000 \
  --verbose

print("\n" + "=" * 60)
print("🎉 Spectral-GS training complete!")
print("=" * 60)
```

#### Cell: Compare Results (NEW)

Add this cell to compare Spectral-GS vs baseline:

```python
import os

print("📊 Training Results Comparison\n")
print("=" * 60)

baseline_dir = "/content/GaussianSplatting/results/lantern_baseline"
spectral_dir = "/content/GaussianSplatting/results/lantern_spectral"

for name, result_dir in [("Baseline", baseline_dir), ("Spectral-GS", spectral_dir)]:
    print(f"\n{name}:")
    if os.path.exists(result_dir):
        # Count checkpoints
        if os.path.exists(f"{result_dir}"):
            files = os.listdir(result_dir)
            ckpts = [f for f in files if f.endswith('.pt')]
            print(f"  ✅ {len(ckpts)} checkpoints saved")

            # Show final model
            if os.path.exists(f"{result_dir}/final.pt"):
                size = os.path.getsize(f"{result_dir}/final.pt") / (1024*1024)
                print(f"  ✅ Final model: {size:.1f} MB")
    else:
        print(f"  ❌ Not trained yet")

print("\n" + "=" * 60)
```

### 3. Complete Colab Workflow

Here's the recommended cell order for training with Spectral-GS:

1. **Session Restore** (your existing cell-16)
2. **Install gsplat** (your existing cell-19)
3. **Install Spectral-GS deps** (NEW - see above)
4. **Verify Installation** (NEW - see above)
5. **Check Dataset Structure** (your existing cell-22)
6. **Train Spectral-GS** (NEW - replaces cell-25)
7. **Compare Results** (NEW - see above)
8. **Download PLY** (your existing cell-27)
9. **Backup to Drive** (your existing cell-17)

## Alternative: Side-by-Side Comparison

If you want to train both baseline and Spectral-GS for comparison:

```python
# Train baseline first
%cd /content/gsplat/examples
!python simple_trainer.py default \
  --data_dir /content/GaussianSplatting/lantern_ds \
  --result_dir /content/GaussianSplatting/results/lantern_baseline \
  --max_steps 30000

print("\n✅ Baseline complete!\n")

# Then train Spectral-GS
%cd /content/GaussianSplatting
!python scripts/train_spectral_gs.py \
  --data_dir /content/GaussianSplatting/lantern_ds \
  --result_dir /content/GaussianSplatting/results/lantern_spectral \
  --max_steps 30000 \
  --spectral_threshold 0.5 \
  --enable_spectral_splitting \
  --verbose

print("\n✅ Spectral-GS complete!")
```

## Key Parameters for Colab

### Memory Management

If you run out of memory, reduce these:
```python
--data_factor 2            # Downsample images by 2x
--max_steps 15000          # Fewer training steps
--refine_stop_iter 10000   # Stop densification earlier
```

### Needle Artifact Control

Adjust spectral threshold based on your scene:
```python
--spectral_threshold 0.3   # More aggressive (splits more)
--spectral_threshold 0.5   # Balanced (default)
--spectral_threshold 0.7   # Conservative (splits less)
```

### Speed vs Quality

For faster iterations:
```python
--max_steps 5000           # Quick test
--refine_every 200         # Less frequent refinement
--log_every 500            # Less frequent logging
```

For best quality:
```python
--max_steps 30000          # Full training
--refine_every 100         # Default refinement frequency
--enable_filtering         # Enable post-processing
```

## Troubleshooting

### Import Error: "No module named 'spectral_gs'"

**Solution:**
```python
# Make sure repository is up to date
%cd /content/GaussianSplatting
!git pull origin main

# Add to Python path
import sys
sys.path.insert(0, '/content/GaussianSplatting/src')
```

### CUDA Out of Memory

**Solution:** Reduce memory usage
```python
--data_factor 2            # Downsample images
--max_steps 15000          # Fewer steps
```

## Compatibility Notes

**The training script (`scripts/train_spectral_gs.py`) includes built-in compatibility for:**

- ✅ gsplat 1.5.3+ (custom knn implementation)
- ✅ No pycolmap dependency (simplified dataset loader)
- ✅ Proper gradient flow for densification
- ✅ Shape compatibility with latest gsplat API
- ✅ Warmup period to avoid early training instabilities

No additional patches or fixes are needed in your Colab notebook!

## Expected Output

During training, you should see:
```
Step 00500 | Loss: 0.0234 | PSNR: 28.45 | Gaussians: 125430 | Entropy: 0.723 | Needles: 12543
[SpectralStrategy] Step 500: Splitting 234 needle-like Gaussians (entropy < 0.5)
```

**Key metrics:**
- **Entropy**: Average spectral entropy (higher = more isotropic)
- **Needles**: Count of needle-like Gaussians (should decrease over training)
- **PSNR**: Image quality (higher is better)

## Downloading Results

```python
from google.colab import files

# Download final PLY
files.download("/content/GaussianSplatting/results/lantern_spectral/final.ply")

# Download checkpoint
files.download("/content/GaussianSplatting/results/lantern_spectral/final.pt")
```

## Next Steps

After training completes:
1. Download the PLY file
2. View it in a 3D Gaussian Splatting viewer (e.g., https://antimatter15.com/splat/)
3. Compare baseline vs Spectral-GS results
4. Look for reduced needle artifacts in sparse view regions
