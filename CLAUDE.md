# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project reconstructs ASU campus scenes using 3D Gaussian Splatting, combining:
- **VGGT** (Visual Geometry Grounded Transformer, CVPR 2025) for camera pose estimation
- **gsplat** (Nerfstudio's CUDA Gaussian Splatting library, JMLR 2025) for training and rendering

The pipeline processes video footage to create high-quality 3D reconstructions with real-time rendering capabilities.

## Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
# Note: VGGT may need manual installation from GitHub
git clone https://github.com/facebookresearch/vggt.git
```

### Full Pipeline Workflow

1. **Extract and prepare frames from video:**
```bash
python scripts/dataset_prep.py --video input.mp4 --out data/campus_scene --target_frames 150
```
- Performs blur detection (sharpness threshold: 80.0)
- Downscales to 1600px width
- Creates train/test split (every 10th frame is test)
- Outputs to `data/campus_scene/images/` and `splits/`

2. **Run VGGT pose estimation (TODO - stub):**
```bash
python scripts/run_vggt.py --frames_dir data/campus_scene/images --out_dir data/campus_scene/sparse/0
```
- Should output COLMAP-format: `cameras.bin`, `images.bin`, `points3D.bin`
- Currently a stub - needs VGGT integration

3. **Train Gaussian Splatting:**

**Option A: Standard 3D-GS (baseline)**
```bash
bash scripts/run_gsplat.sh data/campus_scene results/baseline 1 10000
# Arguments: DATA_DIR RESULT_DIR DATA_FACTOR MAX_STEPS
```
Or using Python wrapper:
```bash
python src/baseline/train_gsplat.py --data_dir data/campus_scene --result_dir results/baseline --max_steps 10000
```

**Option B: Spectral-GS (reduces needle artifacts)**
```bash
bash scripts/run_spectral_gs.sh data/campus_scene results/spectral_gs 1 10000 0.5
# Arguments: DATA_DIR RESULT_DIR DATA_FACTOR MAX_STEPS SPECTRAL_THRESHOLD
```
Or with custom parameters:
```bash
python scripts/train_spectral_gs.py \
  --data_dir data/campus_scene \
  --result_dir results/spectral_gs \
  --max_steps 10000 \
  --spectral_threshold 0.5 \
  --enable_spectral_splitting \
  --enable_filtering \
  --verbose
```

### Quick Commands
```bash
# Extract frames from video manually
ffmpeg -i input.mp4 -vf fps=3 data/campus_scene/frames/frame_%04d.png

# Train with custom parameters
python examples/simple_trainer.py default \
  --data_dir data/campus_scene \
  --result_dir results/baseline \
  --max_steps 10000 \
  --sh_degree 3 \
  --ssim_lambda 0.2 \
  --save_ply
```

## Architecture

### Pipeline Flow
1. **Data Capture** → video/images of campus scene
2. **Dataset Prep** → blur filtering, downscaling, train/test split
3. **Pose Estimation** → VGGT outputs COLMAP format (intrinsics/extrinsics/sparse points)
4. **Optional Refinement** → COLMAP bundle adjustment
5. **GS Training** → gsplat trains with densification, spherical harmonics (SH degree 3)
6. **Evaluation** → PSNR/SSIM metrics on test set
7. **Visualization** → turntable videos, novel view synthesis

### Data Format
- **Input:** COLMAP-format dataset with `sparse/0/` directory containing binary files
- **Expected structure:**
  ```
  data/campus_scene/
    images/           # cleaned, downscaled frames
    splits/           # train.txt, test.txt
    sparse/0/         # COLMAP: cameras.bin, images.bin, points3D.bin
  ```

### Gaussian Representation
Each 3D Gaussian is parameterized by:
- Position (x, y, z)
- Scale (sx, sy, sz)
- Rotation (quaternion)
- Opacity (α)
- Color (RGB or spherical harmonics coefficients)

### Training Details
- Resolution: 1000-1500px typical
- Iterations: 5,000-10,000 (default: 10,000)
- Features: CUDA rasterizer, adaptive densification, exposure compensation
- Spherical harmonics: starts at degree 0, ramps to degree 3
- Loss: weighted combination of L1 + SSIM (lambda=0.2)

## Key Files

- `scripts/dataset_prep.py`: Video preprocessing, blur culling, train/test split
- `scripts/run_vggt.py`: VGGT pose estimation wrapper (currently stub)
- `scripts/run_gsplat.sh`: Shell script to launch standard gsplat training
- `scripts/train_spectral_gs.py`: **Spectral-GS training script** (reduces needle artifacts)
- `scripts/run_spectral_gs.sh`: Shell wrapper for Spectral-GS training
- `src/baseline/train_gsplat.py`: Python wrapper for gsplat training with validation
- `src/spectral_gs/`: **Spectral-GS implementation** (entropy-based densification)
  - `spectral_entropy.py`: Core entropy computation and splitting logic
  - `spectral_strategy.py`: Custom densification strategy
  - `filtering.py`: 2D view-consistent filtering
- `requirements.txt`: Core dependencies (imageio, gsplat, pycolmap, torch, opencv)

## Development Notes

### Current Status
- Dataset preparation pipeline is complete
- VGGT integration is stubbed (needs implementation in `scripts/run_vggt.py`)
- gsplat training wrapper is complete but calls `examples/simple_trainer.py` (assumes gsplat repo structure)

### Known Issues
- VGGT stub needs real implementation to write COLMAP binaries
- Training script assumes gsplat examples are available at `examples/simple_trainer.py` (may need path adjustment)
- Optional COLMAP bundle adjustment refinement is not yet integrated

### Spectral-GS Implementation
**NEW**: We've implemented Spectral-GS (SIGGRAPH Asia 2025) to reduce needle artifacts:
- **Spectral entropy**: H(Σ) = -∑(sᵢ²/tr(Σ) * ln(sᵢ²/tr(Σ))) measures Gaussian shape
- **Shape-aware splitting**: Splits needle-like Gaussians (entropy < 0.5) even with low gradients
- **Anisotropic reduction**: Scales reduced by 1.6x during splitting
- **View-consistent filtering**: Optional 2D filtering to reduce aliasing

**Key parameters:**
- `--spectral_threshold 0.5`: Lower = more aggressive needle splitting
- `--enable_spectral_splitting`: Enable spectral-based densification
- `--enable_filtering`: Apply 2D Gaussian filtering to rendered images

See `src/spectral_gs/README.md` for detailed documentation.

### Tuning Parameters
If reconstructions have issues:
- **Needle artifacts**: Use Spectral-GS with `--spectral_threshold 0.4` (lower = more splitting)
- **Ghosting/blur**: Enable densification, tune opacity/scale regularizers
- **Slow training**: Reduce resolution with `--data_factor`, decrease iterations
- **Lighting variation**: Enable exposure compensation in gsplat
- **VGGT pose errors**: Run COLMAP bundle adjustment refinement

## References
- Kerbl et al., "3D Gaussian Splatting", SIGGRAPH 2023
- Fanello et al., "VGGT", CVPR 2025
- Kerbl et al., "gsplat", JMLR 2025
- Schönberger & Frahm, "COLMAP", CVPR 2016
- Huang et al., "Spectral-GS: Taming 3D Gaussian Splatting with Spectral Entropy", SIGGRAPH Asia 2025
