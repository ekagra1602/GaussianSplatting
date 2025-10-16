# Baseline 3D Gaussian Splatting Training Guide

**Understanding `train_spectral_gs.py` with Spectral Splitting DISABLED**

This document explains the standard 3D Gaussian Splatting training pipeline implemented in our script when you run WITHOUT the `--enable_spectral_splitting` flag.

---

## Table of Contents

1. [What This Script Does](#what-this-script-does)
2. [Input Requirements](#input-requirements)
3. [Training Pipeline Overview](#training-pipeline-overview)
4. [Step-by-Step Breakdown](#step-by-step-breakdown)
5. [Key Components Explained](#key-components-explained)
6. [Training Parameters](#training-parameters)
7. [Output Files](#output-files)
8. [Common Issues](#common-issues)

---

## What This Script Does

The script trains a 3D Gaussian Splatting model to reconstruct a 3D scene from:
- **Input:** Multiple 2D images with known camera poses
- **Output:** A set of 3D Gaussians that render to match the input images

Think of it as: **Learning to place and shape 3D blobs so they look right from all camera angles**

---

## Input Requirements

### Required Directory Structure

```
data/scene_name/
├── images/              # Your scene images
│   ├── frame_00001.jpg
│   ├── frame_00002.jpg
│   └── ...
├── sparse/0/            # COLMAP camera data (from VGGT)
│   ├── cameras.bin      # Camera intrinsics (focal length, etc.)
│   ├── images.bin       # Camera poses (where each photo was taken)
│   └── points3D.bin     # Sparse 3D points (initialization)
└── splits/              # Train/test split
    ├── train.txt
    └── test.txt
```

### How You Get This

1. **Extract frames** from video using `dataset_prep.py`
2. **Run VGGT** to get camera poses: `demo_colmap.py --use_ba`

---

## Training Pipeline Overview

```
┌──────────────────────────────────────────────────────┐
│  1. SETUP PHASE                                      │
│     - Load COLMAP data (camera poses + points)       │
│     - Initialize 3D Gaussians from sparse points     │
│     - Create optimizers for all Gaussian parameters  │
│     - Initialize densification strategy              │
└──────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│  2. TRAINING LOOP (repeat for max_steps)             │
│     ┌──────────────────────────────────────────┐    │
│     │  For each training image:                │    │
│     │                                           │    │
│     │  a) FORWARD PASS                          │    │
│     │     - Rasterize Gaussians to image       │    │
│     │                                           │    │
│     │  b) COMPUTE LOSS                          │    │
│     │     - Compare rendered vs real image     │    │
│     │                                           │    │
│     │  c) BACKWARD PASS                         │    │
│     │     - Compute gradients                  │    │
│     │                                           │    │
│     │  d) DENSIFICATION (every 100 steps)       │    │
│     │     - Split/clone high-gradient Gaussians│    │
│     │     - Prune low-opacity Gaussians        │    │
│     │                                           │    │
│     │  e) OPTIMIZER STEP                        │    │
│     │     - Update Gaussian parameters         │    │
│     │                                           │    │
│     │  f) HARD CAP (every 500 steps)            │    │
│     │     - Keep only top-K Gaussians          │    │
│     └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│  3. SAVE PHASE                                       │
│     - Save .pt checkpoint (for resuming)             │
│     - Save .ply file (for viewing)                   │
└──────────────────────────────────────────────────────┘
```

---

## Step-by-Step Breakdown

### Phase 1: Setup (Lines 160-220)

#### 1.1 Load Dataset

```python
# Read COLMAP data
parser = Parser(
    data_dir=args.data_dir,     # e.g., "data/scene"
    factor=args.data_factor      # 1 = full res, 2 = half res, etc.
)

# Create PyTorch dataset
trainset = ColmapDataset(parser, split="train")
train_loader = DataLoader(trainset, batch_size=1, shuffle=True)
```

**What this does:**
- Reads `sparse/0/*.bin` files to get camera poses and sparse points
- Loads image file paths
- Creates data loader that serves one image at a time

#### 1.2 Initialize Gaussians

```python
# Get scene scale (CRITICAL for proper learning rates!)
scene_scale = parser.scene_scale * 1.1

# Initialize from COLMAP sparse points
params = init_gaussians(
    points=parser.points,              # [N, 3] 3D positions
    colors=parser.points_rgb / 255.0,  # [N, 3] RGB colors (normalized!)
    init_scale=args.init_scale         # Scale multiplier (default: 1.0)
)
```

**What `init_gaussians` does:**

```python
def init_gaussians(points, colors, init_scale=1.0):
    N = points.shape[0]  # Number of points (e.g., 100,000)

    # 1. Initialize scales based on nearest neighbors
    # (Gaussians should be sized to cover gaps between points)
    distances = knn(points, 4)[:, 1:]  # Distance to 3 nearest neighbors
    avg_dist = distances.mean(dim=-1)  # Average distance
    scales = torch.log(avg_dist * init_scale).repeat(1, 3)  # [N, 3]
    # Stored in LOG space for optimization!

    # 2. Initialize rotations (quaternions)
    quats = torch.rand((N, 4))  # Random orientations
    quats = quats / torch.norm(quats, dim=-1, keepdim=True)  # Normalize

    # 3. Initialize opacities
    opacities = torch.logit(torch.full((N,), 0.1))  # All 10% opacity
    # Stored in LOGIT space for optimization!

    # 4. Initialize colors using Spherical Harmonics
    sh0 = rgb_to_sh(colors).unsqueeze(1)  # [N, 1, 3]
    # DC component (degree 0) of SH
    shN = torch.zeros((N, 0, 3))  # Empty (we only use degree 0)

    return {
        "means": points,      # [N, 3]
        "scales": scales,     # [N, 3] in log space
        "quats": quats,       # [N, 4]
        "opacities": opacities,  # [N] in logit space
        "sh0": sh0,           # [N, 1, 3] SH DC component
        "shN": shN            # [N, 0, 3] empty
    }
```

**Key transformations:**
- **Scales:** Stored as `log(scale)` → optimizer updates are multiplicative
- **Opacities:** Stored as `logit(opacity)` → ensures [0,1] range after sigmoid
- **Colors:** Stored as SH coefficients → better than raw RGB

#### 1.3 Create Optimizers

```python
optimizers = create_optimizers(params, lr_scale=scene_scale)
```

```python
def create_optimizers(params, lr_scale=1.0):
    return {
        "means": Adam([params["means"]], lr=1.6e-4 * lr_scale),  # Scaled by scene!
        "scales": Adam([params["scales"]], lr=5e-3),
        "quats": Adam([params["quats"]], lr=1e-3),
        "opacities": Adam([params["opacities"]], lr=5e-2),
        "sh0": Adam([params["sh0"]], lr=2.5e-3),
        "shN": Adam([params["shN"]], lr=2.5e-3 / 20)
    }
```

**Why different learning rates?**
- `means`: Slow (1.6e-4) - positions should move carefully
- `opacities`: Fast (5e-2) - quick to adjust visibility
- `scales`: Medium (5e-3) - size adjustments
- Scene scale: Means LR multiplied by scene normalization (typically 1.5-3x)

#### 1.4 Initialize Strategy

```python
strategy = SpectralStrategy(
    refine_start_iter=500,
    refine_stop_iter=15000,
    refine_every=100,
    # ... other params
    enable_spectral_splitting=False  # DISABLED for baseline
)
strategy_state = strategy.initialize_state()
```

**What this creates:**
- A densification controller
- Tracks gradients and decides when to split/clone/prune
- With spectral splitting disabled, uses only standard gradient-based densification

---

### Phase 2: Training Loop (Lines 226-350)

#### 2.1 Load Image Batch

```python
for step, batch in enumerate(train_loader):
    if step >= args.max_steps:
        break

    # Get data
    image = batch["image"][0].to(device) / 255.0  # [H, W, 3] normalized!
    K = batch["K"][0].to(device)                  # [3, 3] camera intrinsics
    camtoworld = batch["camtoworld"][0].to(device)  # [4, 4] camera pose
    H, W = image.shape[:2]
```

**What's in the batch:**
- `image`: The ground truth photo (what we want to match)
- `K`: Camera intrinsics matrix (focal length, principal point)
- `camtoworld`: Where the camera was when this photo was taken

#### 2.2 Prepare Gaussians for Rendering

```python
# Convert from optimization space to rendering space
colors = torch.cat([params["sh0"], params["shN"]], 1)  # [N, 1, 3]
opacities = torch.sigmoid(params["opacities"])         # [N] → [0, 1]
scales = torch.exp(params["scales"])                   # [N, 3] → positive
quats = params["quats"]                                # [N, 4] (normalized by gsplat)
```

**Why these transformations?**
- Optimizer works on unbounded values (log, logit)
- Rasterizer needs actual values (scales, opacities)

#### 2.3 Rasterize (Render) the Gaussians

```python
# Prepare camera matrices
viewmat = torch.linalg.inv(camtoworld).unsqueeze(0)  # World → camera
K_batched = K.unsqueeze(0)

# Render!
renders, alphas, render_info = rasterization(
    means=params["means"],      # 3D positions
    quats=quats,                # Rotations
    scales=scales,              # Sizes
    opacities=opacities,        # Transparencies
    colors=colors,              # SH coefficients
    viewmats=viewmat,           # Camera view matrix
    Ks=K_batched,              # Camera intrinsics
    width=W,
    height=H,
    sh_degree=0,               # Using only DC component
    absgrad=strategy.absgrad    # Track gradients for densification
)

rendered_image = renders[0]  # [H, W, 3]
```

**What rasterization does:**
1. Projects each 3D Gaussian onto the image plane
2. Computes its 2D splat (ellipse on screen)
3. Alpha-blends all splats in depth order
4. Returns the final rendered image

#### 2.4 Compute Loss

```python
# L1 loss (simple pixel difference)
l1_loss = F.l1_loss(rendered_image, image)
loss = l1_loss

# Compute PSNR for logging
mse = F.mse_loss(rendered_image, image)
psnr = -10.0 * torch.log10(mse)
```

**Loss function:**
- Compares rendered image vs real image pixel-by-pixel
- L1 = average absolute difference
- PSNR = Peak Signal-to-Noise Ratio (higher is better, 25-35 is good)

#### 2.5 Backward Pass

```python
# Zero gradients
for optimizer in optimizers.values():
    optimizer.zero_grad()

# Backpropagate
loss.backward()

# Now all parameters have .grad populated
```

#### 2.6 Densification (Every 100 Steps)

```python
if step >= 500 and step % 100 == 0 and step < 15000:
    strategy.step_post_backward(
        params, optimizers, strategy_state, step,
        render_info, packed=False
    )
```

**What densification does:**

```python
# Inside DefaultStrategy.step_post_backward():

# 1. Accumulate gradients
grad2d = render_info["means2d"].grad  # 2D screen-space gradients
state["grad2d"] += grad2d
state["count"] += 1

if step % 100 == 0:
    avg_grad2d = state["grad2d"] / state["count"]

    # 2. CLONE small Gaussians with high gradients
    # (need more coverage in this area)
    is_small = scales.max(dim=1)[0] < 0.01
    has_high_grad = avg_grad2d > 0.0002
    to_clone = is_small & has_high_grad

    for idx in to_clone:
        # Create duplicate Gaussian at same location
        params["means"] = torch.cat([params["means"], params["means"][idx]])
        # ... duplicate all parameters

    # 3. SPLIT large Gaussians with high gradients
    # (this region needs finer detail)
    is_large = scales.max(dim=1)[0] >= 0.01
    to_split = is_large & has_high_grad

    for idx in to_split:
        # Create 2 smaller Gaussians
        new_scale = params["scales"][idx] / 1.6  # Smaller
        new_means = sample_around(params["means"][idx])  # Nearby positions
        # ... add both children

    # 4. PRUNE low-opacity Gaussians
    # (these aren't contributing to the image)
    to_prune = opacities < 0.005

    for key in params.keys():
        params[key] = params[key][~to_prune]  # Keep only non-pruned

    # 5. Reset gradient accumulators
    state["grad2d"] = 0
    state["count"] = 0
```

**The three operations:**
1. **Clone:** Duplicate Gaussian (for small ones with high gradient)
2. **Split:** Create 2 smaller Gaussians (for large ones with high gradient)
3. **Prune:** Remove transparent Gaussians (opacity < threshold)

#### 2.7 Optimizer Step

```python
for optimizer in optimizers.values():
    optimizer.step()
```

Updates all parameters using their gradients:
- Means move toward better positions
- Scales adjust sizes
- Opacities adjust transparency
- Rotations adjust orientations

#### 2.8 Hard Cap (Every 500 Steps)

```python
if step % 500 == 0 and len(params["means"]) > args.max_gaussians:
    # Too many Gaussians! Keep only the best ones
    opacities = torch.sigmoid(params["opacities"])
    _, indices = torch.topk(opacities, args.max_gaussians)

    # Prune all parameters
    for key in params.keys():
        params[key] = torch.nn.Parameter(params[key][indices])

    # Reinitialize optimizers
    optimizers = create_optimizers(params, lr_scale=scene_scale)
    strategy_state = strategy.initialize_state()
```

**Why we need this:**
- Densification can explode Gaussian count
- GPU memory is limited
- Keeps only top-K most opaque (most visible) Gaussians

---

### Phase 3: Save (Lines 380-390)

```python
# Save PyTorch checkpoint
save_checkpoint(params, result_dir / "final.pt")

# Save PLY for viewing
save_ply(params, result_dir / "final.ply")
```

**Output files:**
- `final.pt`: Can resume training from this
- `final.ply`: Can view in 3D viewer

---

## Key Components Explained

### 1. Gaussian Parameterization

Each 3D Gaussian is defined by:

```python
mean = [x, y, z]           # 3D position
scale = [sx, sy, sz]       # Size along 3 axes
quat = [w, x, y, z]        # Rotation (quaternion)
opacity = α                # Transparency [0, 1]
sh0 = [r, g, b]           # Color (DC component of SH)
```

**Covariance matrix:**
```
Σ = R · S · Sᵀ · Rᵀ

where:
  R = rotation matrix from quaternion
  S = diagonal matrix of scales
```

### 2. Rasterization Process

```
For each pixel (x, y):
  1. Find all Gaussians visible at this pixel
  2. Sort by depth (front to back)
  3. For each Gaussian:
     - Compute 2D Gaussian weight at (x, y)
     - Multiply by opacity and color
     - Alpha blend with previous color
  4. Result = final pixel color
```

### 3. Learning Rate Scaling

**Critical:** The `scene_scale` adjustment!

```python
scene_scale = parser.scene_scale * 1.1
# For a scene normalized to [-1, 1], scene_scale ≈ 2.5

optimizers["means"] = Adam(..., lr=1.6e-4 * 2.5)  # = 4e-4
```

**Why this matters:**
- COLMAP normalizes scenes to unit scale
- If your scene is 10 meters wide → normalized to ~2 units
- Without scaling, Gaussians move too slowly (LR too small for scale)

### 4. Parameter Spaces

| Parameter | Storage Space | Rendering Space | Why? |
|-----------|--------------|-----------------|------|
| Scales | `log(s)` | `exp(log(s)) = s` | Ensures positive, multiplicative updates |
| Opacities | `logit(α)` | `sigmoid(logit(α)) = α` | Ensures [0,1] range |
| Colors | SH coefficients | Evaluate SH | View-dependent effects |
| Quaternions | Raw | Normalized by gsplat | Unit quaternions |

---

## Training Parameters

### Essential Parameters

```bash
python train_spectral_gs.py \
  --data_dir data/scene \          # Input COLMAP dataset
  --result_dir results/baseline \  # Where to save outputs
  --max_steps 10000                # How many iterations
```

### Optional Tuning

```bash
  --data_factor 1 \          # Image downsampling (1=full, 2=half)
  --init_scale 1.0 \         # Initial Gaussian size multiplier
  --max_gaussians 300000 \   # Hard cap on count
  --log_every 100            # Print metrics frequency
```

### Defaults (Usually Good)

```python
refine_start_iter = 500     # Start densification at step 500
refine_stop_iter = 15000    # Stop densification at step 15000
refine_every = 100          # Densify every 100 steps
```

---

## Output Files

### `final.pt` (PyTorch Checkpoint)

```python
checkpoint = {
    "means": [N, 3],
    "scales": [N, 3],
    "quats": [N, 4],
    "opacities": [N],
    "sh0": [N, 1, 3],
    "shN": [N, 0, 3]
}
```

**Use case:** Resume training

```python
params = torch.load("results/baseline/final.pt")
# Continue training from here
```

### `final.ply` (Point Cloud for Viewing)

Standard PLY format with Gaussian splatting properties:

```
ply
format binary_little_endian 1.0
element vertex N
property float x, y, z          # Position
property float nx, ny, nz       # Normals (unused)
property float f_dc_0, 1, 2     # SH DC coefficients
property float opacity          # Opacity
property float scale_0, 1, 2    # Scales
property float rot_0, 1, 2, 3   # Quaternion
```

**View at:** https://antimatter15.com/splat/

---

## Common Issues

### Issue 1: PSNR < 5 (Black Screen)

**Symptoms:**
```
Step 100 | Loss: 0.65 | PSNR: 2.3
Step 500 | Loss: 0.65 | PSNR: 2.3  (not improving!)
```

**Causes & Fixes:**

1. **Forgot to normalize images**
   ```python
   # WRONG:
   image = batch["image"][0].to(device)  # [0, 255]

   # CORRECT:
   image = batch["image"][0].to(device) / 255.0  # [0, 1]
   ```

2. **Missing scene_scale**
   ```python
   # WRONG:
   optimizers["means"] = Adam(..., lr=1.6e-4)

   # CORRECT:
   scene_scale = parser.scene_scale * 1.1
   optimizers["means"] = Adam(..., lr=1.6e-4 * scene_scale)
   ```

3. **Wrong color initialization**
   ```python
   # WRONG:
   sh0 = torch.logit(colors)

   # CORRECT:
   sh0 = rgb_to_sh(colors).unsqueeze(1)
   ```

### Issue 2: Gaussian Count Exploding

**Symptoms:**
```
Step 1000: 120K Gaussians
Step 2000: 350K Gaussians
Step 3000: 800K Gaussians  (OOM crash!)
```

**Fixes:**

1. **Reduce max_gaussians**
   ```bash
   --max_gaussians 200000  # Instead of 300000
   ```

2. **More frequent hard caps**
   ```python
   # Every 500 steps → every 300 steps
   if step % 300 == 0 and len(params["means"]) > max_gaussians:
   ```

### Issue 3: Slow Training

**Symptoms:**
- 5-10 iterations per second (should be 30-50)

**Fixes:**

1. **Lower resolution**
   ```bash
   --data_factor 2  # Half resolution
   ```

2. **Fewer Gaussians**
   ```bash
   --max_gaussians 150000
   ```

3. **Check GPU utilization**
   ```bash
   nvidia-smi  # Should show >80% GPU usage
   ```

### Issue 4: Blurry Results

**Symptoms:**
- PSNR ~25-27 but images look soft/blurry

**Causes:**

1. **Camera poses inaccurate**
   - Re-run VGGT with `--use_ba` (Bundle Adjustment)

2. **Not enough iterations**
   - Increase `--max_steps 30000`

3. **Input images low quality**
   - Check `--min_sharpness` in dataset prep
   - Use `-q:v 1` in ffmpeg extraction

---

## Performance Expectations

### Training Time

| Steps | Resolution | Gaussians | Time (A100) | Time (T4) |
|-------|-----------|-----------|-------------|-----------|
| 10K   | 1600px    | 300K      | ~25 min     | ~60 min   |
| 30K   | 1600px    | 300K      | ~75 min     | ~180 min  |
| 10K   | 800px     | 150K      | ~8 min      | ~20 min   |

### Quality Metrics

| Dataset Quality | Expected PSNR | Expected SSIM |
|----------------|---------------|---------------|
| Excellent (VGGT+BA, 100 sharp frames) | 30-35 dB | 0.92-0.96 |
| Good (VGGT+BA, 70 frames) | 27-30 dB | 0.88-0.92 |
| Fair (VGGT no BA, 50 frames) | 23-27 dB | 0.80-0.88 |
| Poor (bad poses) | <20 dB | <0.75 |

---

## Complete Example

```bash
# 1. Prepare dataset
python scripts/dataset_prep.py \
  --video campus.mp4 \
  --out data/campus \
  --target_frames 100 \
  --min_sharpness 60 \
  --width 1600

# Output:
# data/campus/images/       (100 frames)
# data/campus/splits/       (train.txt, test.txt)

# 2. Get camera poses with VGGT
python vggt/demo_colmap.py \
  --scene_dir data/campus \
  --use_ba \
  --fine_tracking

# Output:
# data/campus/sparse/0/cameras.bin
# data/campus/sparse/0/images.bin
# data/campus/sparse/0/points3D.bin

# 3. Train baseline 3D-GS
python scripts/train_spectral_gs.py \
  --data_dir data/campus \
  --result_dir results/campus_baseline \
  --max_steps 10000 \
  --log_every 100 \
  --verbose

# Output during training:
# Step 00100 | Loss: 0.1234 | PSNR: 18.52 | Gaussians: 105234
# Step 00500 | Loss: 0.0543 | PSNR: 24.12 | Gaussians: 145678
# Step 01000 | Loss: 0.0234 | PSNR: 28.34 | Gaussians: 234567
# ...
# Step 10000 | Loss: 0.0087 | PSNR: 31.23 | Gaussians: 289456

# 4. View results
# Download results/campus_baseline/final.ply
# Open at https://antimatter15.com/splat/
```

---

## Summary

**The baseline training script does:**

1. ✅ Load COLMAP dataset (camera poses + images)
2. ✅ Initialize Gaussians from sparse points
3. ✅ Optimize Gaussian parameters to match images
4. ✅ Adaptively add/remove Gaussians (densification)
5. ✅ Save viewable 3D model

**It does NOT do:**
- ❌ Spectral splitting (unless `--enable_spectral_splitting`)
- ❌ Camera pose estimation (that's VGGT's job)
- ❌ Video frame extraction (that's `dataset_prep.py`)

**Key to success:**
- Good camera poses (VGGT with Bundle Adjustment)
- Sharp input frames (blur filtering)
- Proper scene_scale (for LR scaling)
- Normalized images (/255.0)

---

**Document Version:** 1.0
**Last Updated:** October 15, 2025
**Script:** `train_spectral_gs.py` (baseline mode)
