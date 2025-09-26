# Gaussian Splatting for Campus Reconstruction

This project implements **3D Gaussian Splatting** to reconstruct a portion of the ASU campus.
It combines **VGGT** (Visual Geometry Grounded Transformer, CVPR 2025) for fast pose estimation with **gsplat** (Nerfstudio’s CUDA Gaussian Splatting library, JMLR 2025) for training and rendering.

In addition, a **Tiny CPU Rasterizer** is implemented from scratch to demonstrate the math behind Gaussian Splatting by re-rendering a cropped scene patch.

---

## 📌 Project Goals

* Reconstruct a real-world campus scene from video/images.
* Achieve high-quality results using **VGGT + gsplat**.
* Implement a **custom Tiny Rasterizer** (CPU, NumPy) to prove conceptual understanding.
* Evaluate reconstructions quantitatively (PSNR/SSIM) and qualitatively (videos, turntables).
* Deliver reproducible code, a demo video, and presentation.

---

## 🏗️ System Overview

**Pipeline:**

1. **Data Capture** → Record campus video or 120–200 images.
2. **Pose Estimation (VGGT)** → Estimate intrinsics, extrinsics, depth, sparse cloud; export COLMAP files.
3. **Optional BA (COLMAP)** → Refine poses for accuracy.
4. **Gaussian Splatting (gsplat)** → Train CUDA-optimized GS model with densification, SH colors, regularizers.
5. **Tiny Rasterizer (Custom)** → CPU-only rasterizer for patch renders (≤5k Gaussians, 256×256).
6. **Evaluation** → PSNR/SSIM metrics + side-by-side visuals.
7. **Visualization** → Export turntable + novel views, `.glb` viewer in Three.js.

---

## 📂 Repository Structure

```
repo/
  data/campus_scene/       # input frames, COLMAP outputs
  src/
    baseline/              # gsplat training & rendering
    custom/                # tiny rasterizer implementation
    utils/                 # camera projection, COLMAP IO
  scripts/
    run_vggt.sh            # run VGGT pose estimation
    run_gsplat.sh          # train gsplat on dataset
    run_rasterizer.sh      # run tiny rasterizer demo
  slides/                  # presentation slides
  report/                  # written report
  README.md
```

---

## ⚙️ Setup

### Environment

* Python 3.11+
* CUDA-capable GPU (tested with NVIDIA RTX series)
* [PyTorch](https://pytorch.org) (install first)
* [VGGT](https://github.com/facebookresearch/vggt) (pose estimation)
* [gsplat](https://github.com/nerfstudio-project/gsplat) (Gaussian Splatting)
* [COLMAP](https://colmap.github.io/) (optional bundle adjustment)

Install gsplat and dependencies:

```bash
pip install torch torchvision torchaudio
pip install gsplat
pip install numpy pillow matplotlib
```

Clone VGGT:

```bash
git clone https://github.com/facebookresearch/vggt.git
```

---

## 🚀 Usage

### 1. Data Capture

Record ~150 frames of your scene (high overlap, steady exposure).
Extract frames if starting from video:

```bash
ffmpeg -i input.mp4 -vf fps=3 data/campus_scene/frames/frame_%04d.png
```

### 2. Pose Estimation with VGGT

```bash
bash scripts/run_vggt.sh data/campus_scene/frames/ data/campus_scene/colmap_out/
```

This produces `cameras.bin`, `images.bin`, `points3D.bin`.

(Optional) Refine with COLMAP BA:

```bash
colmap bundle_adjuster --input_path data/campus_scene/colmap_out
```

### 3. Train Gaussian Splatting Model

```bash
bash scripts/run_gsplat.sh data/campus_scene/colmap_out
```

This trains for ~5–10k iterations and outputs renderings + checkpoints.

### 4. Tiny Rasterizer (Custom)

Run the CPU-only rasterizer for a patch render:

```bash
bash scripts/run_rasterizer.sh
```

Outputs `custom_patch.png` for comparison with gsplat.

---

## 📊 Evaluation

* **Quantitative**: PSNR, SSIM on held-out frames.
* **Qualitative**: turntable video, novel view sweeps, patch comparisons.
* Compare gsplat vs Tiny Rasterizer outputs on crops.

---

## 📅 Timeline

| Days  | Task                                      |
| ----- | ----------------------------------------- |
| 1–2   | Setup, stubs, notes                       |
| 3–4   | Data capture + VGGT poses                 |
| 5–6   | Baseline gsplat training (2–3k iters)     |
| 7–9   | Full gsplat training (→10k iters, tuning) |
| 10–11 | Tiny Rasterizer implementation            |
| 12    | Ablations & evaluation                    |
| 13    | Draft slides                              |
| 14    | Dry run (pipeline end-to-end)             |
| 15    | Final polish (slides + demo video)        |

---

## 📝 Deliverables

* ✅ High-quality Gaussian Splatting reconstruction of campus scene.
* ✅ Custom Tiny CPU Rasterizer (educational).
* ✅ Demo video (turntable, novel-view sweeps).
* ✅ Slide deck (~12 slides).
* ✅ Report (4–6 pages).
* ✅ Public repo with reproducible scripts.

---

## 🔍 References

* Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023.
* Kerbl et al., *gsplat: A Modular CUDA Library for Gaussian Splatting*, JMLR 2025.
* Fanello et al., *Visual Geometry Grounded Transformer (VGGT)*, CVPR 2025.
* Schönberger & Frahm, *Structure-from-Motion Revisited (COLMAP)*, CVPR 2016.

---
