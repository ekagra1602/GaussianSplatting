# Gaussian Splatting for Campus Reconstruction

This project reconstructs a portion of the ASU campus using **3D Gaussian Splatting**.  
It combines **VGGT** (Visual Geometry Grounded Transformer, CVPR 2025) for pose estimation with **gsplat** (Nerfstudio’s CUDA Gaussian Splatting library, JMLR 2025) for training and rendering.

## 📌 Goals
- Capture campus scene with video/images.
- Estimate camera poses & sparse 3D points with VGGT (optional COLMAP BA refine).
- Train Gaussian Splatting model using gsplat.
- Evaluate with PSNR/SSIM and produce demo videos.

## 🏗️ Pipeline
1. Data Capture → campus frames.
2. Pose Estimation → VGGT → COLMAP-format export.
3. (Optional) COLMAP BA refinement.
4. Gaussian Splatting Training → gsplat (with densification, SH colors).
5. Evaluation & Visualization → metrics + turntable video.

## ⚙️ Setup
- Python 3.11+, CUDA GPU
- PyTorch
- [VGGT](https://github.com/facebookresearch/vggt)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [COLMAP](https://colmap.github.io/) (optional)

Install:
```bash
pip install torch torchvision torchaudio
pip install gsplat
pip install numpy pillow matplotlib
git clone https://github.com/facebookresearch/vggt.git
````

## 🚀 Usage

1. Extract frames:

```bash
ffmpeg -i input.mp4 -vf fps=3 data/campus_scene/frames/frame_%04d.png
```

2. Run VGGT:

```bash
bash scripts/run_vggt.sh data/campus_scene/frames/ data/campus_scene/colmap_out/
```

3. Train gsplat:

```bash
bash scripts/run_gsplat.sh data/campus_scene/colmap_out
```

4. Render results → images & turntable video.

## 📊 Evaluation

* Metrics: PSNR, SSIM on held-out frames.
* Visuals: turntable video, novel-view sweeps.

## 🔍 References

* Kerbl et al., *3D Gaussian Splatting*, SIGGRAPH 2023
* Fanello et al., *VGGT*, CVPR 2025
* Kerbl et al., *gsplat*, JMLR 2025
* Schönberger & Frahm, *COLMAP*, CVPR 2016

``