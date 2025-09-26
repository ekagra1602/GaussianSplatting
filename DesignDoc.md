

# Design Document

**Project:** Gaussian Splatting for Campus Reconstruction
**Author:** Ekagra Gupta
**Advisor:** Prof. [Name]
**Date:** [Insert Date]

---

## 1. Background

3D Gaussian Splatting (Kerbl et al., 2023) is a state-of-the-art method for novel-view synthesis and 3D reconstruction. It represents a scene as a set of 3D anisotropic Gaussians, each parameterized by position, scale, orientation, opacity, and color. Rendering is performed by projecting Gaussians into screen space and alpha-compositing them as ellipses.

The `live-it` project demonstrated that combining prebuilt research libraries (VGGT + gsplat) can accelerate the pipeline while maintaining high quality. VGGT (Visual Geometry Grounded Transformer, CVPR 2025) provides accurate camera poses and an initial 3D point cloud from video or images. The gsplat library (Nerfstudio team, JMLR 2025) implements a CUDA-optimized Gaussian Splatting trainer and renderer.

This project applies the pipeline to reconstruct a chosen portion of the ASU campus within 15 days, while also implementing a **Tiny CPU Rasterizer** from scratch to demonstrate conceptual understanding.

---

## 2. Goals

* **Reconstruct** a campus scene (e.g., courtyard, statue) from captured video/images.
* **Use prebuilt libraries** (VGGT for pose estimation, gsplat for Gaussian Splatting training) to achieve high-quality results.
* **Implement from scratch** a Tiny Rasterizer (CPU, low-res) to re-render a subset of the scene, proving understanding of the math and rendering process.
* **Evaluate** the reconstruction quantitatively (PSNR, SSIM) and qualitatively (videos, turntables).
* **Deliver** a reproducible pipeline, demo video, and presentation.

---

## 3. System Overview

### Pipeline Flow

1. **Data Capture**: Record video / capture frames of chosen campus scene.
2. **Pose Estimation**: Use VGGT to estimate intrinsics, extrinsics, depth, and sparse point cloud. Export COLMAP-format files.
3. **Optional Refinement**: Run COLMAP Bundle Adjustment (BA) on VGGT output to refine camera poses.
4. **Gaussian Splatting Training (gsplat)**: Train 3D Gaussian Splatting model using provided poses and points. Includes densification, SH colors, and regularizers.
5. **Custom Module (Tiny Rasterizer)**: CPU implementation of 3D→2D Gaussian projection and alpha compositing for ≤5k Gaussians at 256×256 resolution. Used for crops and comparison.
6. **Evaluation**: Compare baseline gsplat render vs custom rasterizer (patches). Compute PSNR/SSIM on held-out views.
7. **Visualization**: Export turntable and novel view renders; visualize final scene with Three.js viewer.

---

## 4. Design Details

### 4.1 Data Representation

Each Gaussian parameterized as:

* **Position**: $x \in \mathbb{R}^3$
* **Scale**: $s = (s_x, s_y, s_z)$
* **Rotation**: quaternion $q \in \mathbb{R}^4$, normalized
* **Opacity**: $\alpha \in (0,1)$
* **Color**: RGB or SH coefficients

### 4.2 Pose Estimation (VGGT)

* Inputs: N frames of the scene.
* Outputs: Camera intrinsics, extrinsics, depth maps, sparse 3D points.
* Export: COLMAP-compatible files (`cameras.bin`, `images.bin`, `points3D.bin`).
* Optional: COLMAP BA refinement for improved accuracy.

### 4.3 Gaussian Splatting Training (gsplat)

* Inputs: dataset in COLMAP format.
* Features: CUDA rasterizer, densification, spherical harmonics (SH=0→2), exposure compensation, scale/opacity regularizers.
* Training: 3–10k iterations, resolution 1–1.5k px.
* Outputs: optimized Gaussian parameters; renders.

### 4.4 Custom Module (Tiny Rasterizer)

* **Projection**: World → camera (R, t), then to image via intrinsics K.
* **Covariance projection**:

  $$
  \Sigma_{2D} \approx J \Sigma_{3D} J^T, \quad
  \Sigma_{3D} = R_g \,\text{diag}(s_x^2, s_y^2, s_z^2)\, R_g^T
  $$
* **Gaussian evaluation**:

  $$
  w(p) = \exp\left(-\tfrac{1}{2}(p-\mu)^T \Sigma_{2D}^{-1}(p-\mu)\right)
  $$
* **Alpha compositing (front-to-back)**:

  $$
  C_\text{out} = C_\text{in}(1-\alpha w) + c(\alpha w)
  $$
* Scope: CPU, low-res (≤256×256), ≤5k Gaussians, crop render.

### 4.5 Evaluation Metrics

* PSNR / SSIM between gsplat renders and ground truth held-out views.
* PSNR between Tiny Rasterizer patch and gsplat’s equivalent patch.
* Qualitative: turntable videos, zoomed crops, side-by-side visuals.

---

## 5. Implementation Plan

### Tools & Libraries

* **VGGT**: pretrained transformer (FacebookResearch).
* **gsplat**: Nerfstudio CUDA GS library.
* **COLMAP**: optional BA.
* **PyTorch**: tensor ops, training backend.
* **NumPy**: Tiny Rasterizer.
* **Three.js**: visualization of final `.glb`.

### Repo Structure

```
repo/
  data/campus_scene/       # frames, COLMAP outputs
  src/
    baseline/              # gsplat training & render
    custom/                # tiny rasterizer
    utils/                 # camera projection, colmap io
  scripts/
    run_vggt.sh
    run_gsplat.sh
    run_rasterizer.sh
  slides/
  report/
  README.md
```

---

## 6. Risks & Mitigations

* **VGGT pose errors** → run COLMAP BA refinement.
* **Training too slow** → downscale images, reduce iterations.
* **Tiny Rasterizer too slow** → restrict to crop & few Gaussians.
* **Numerical instability in Σ₂D inversion** → add ε to diagonal, fallback to axis-aligned.

---

## 7. Timeline

| Days  | Task                                             | Deliverable                         |
| ----- | ------------------------------------------------ | ----------------------------------- |
| 1–2   | Setup env, repo, notes; stubs for rasterizer     | Repo skeleton, success criteria     |
| 3–4   | Capture campus video/frames; run VGGT (→ COLMAP) | Dataset + poses                     |
| 5–6   | Baseline gsplat training (~2–3k iters)           | First renders + metrics             |
| 7–9   | Full gsplat training (→10k iters), tuning        | High-quality render video           |
| 10–11 | Implement Tiny Rasterizer                        | Patch render + comparison           |
| 12    | Ablations (VGGT vs VGGT+BA, gsplat vs custom)    | Metrics table + visuals             |
| 13    | Draft slides                                     | Slide deck draft                    |
| 14    | Dry run (scripts, demo pipeline)                 | Working pipeline run                |
| 15    | Final polish                                     | Final slides + video + repo release |

---

## 8. Deliverables

* **Reconstruction**: gsplat-trained model of campus scene.
* **Custom module**: Tiny CPU Rasterizer.
* **Demo video**: turntable + novel-view sweeps.
* **Slides**: ~12 slides, with pipeline, math, results.
* **Report**: 4–6 pages (method, results, discussion).
* **Repo**: scripts, code, README with reproduction steps.

---