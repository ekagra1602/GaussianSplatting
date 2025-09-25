Design Document

Project: Gaussian Splatting
Author: Ekagra Gupta
Advisor: Prof. Heni Ben Amor
Date: 9 September 2025

⸻

1. Background

3D Gaussian Splatting (Kerbl et al., 2023) is a recent technique for real-time novel view synthesis. Instead of dense voxel grids or slow NeRF-style MLPs, it represents scenes as a set of anisotropic 3D Gaussians with position, scale, opacity, and color parameters. These Gaussians are projected into 2D screen space and rasterized as ellipses with alpha compospositing.

This project implements a miniature Gaussian Splatting pipeline to reconstruct a chosen portion of the ASU campus. A baseline implementation will be set up using existing PyTorch-based GS code. To demonstrate deeper understanding, I will build a Tiny CPU Rasterizer from scratch, without relying on heavy libraries, to re-render a subset of the scene.

⸻

2. Goals
	•	Reconstruct a campus scene (e.g., statue or courtyard) using Gaussian Splatting.
	•	Implement from scratch a minimal CPU-based Gaussian rasterizer capable of rendering a small set of Gaussians.
	•	Compare baseline GPU/optimized GS results with my custom rasterizer on accuracy and speed.
	•	Deliver a demo video, reproducible code, and a presentation.

⸻

3. System Overview

Pipeline
	1.	Data Capture: Collect images/video of campus scene.
	2.	Camera Poses: Use COLMAP for structure-from-motion and bundle adjustment.
	3.	Baseline GS: Train a standard Gaussian Splatting model to reconstruct the scene.
	4.	Custom Tiny Rasterizer:
	•	Input: Gaussian parameters (position, scale, rotation, color, opacity).
	•	Output: 2D image patch (low-res) rendered with CPU-only routines.
	5.	Comparison: Render same view using both baseline and custom rasterizer.
	6.	Evaluation: Compare outputs visually and quantitatively (PSNR, SSIM).

⸻

4. Design Details

4.1 Data Representation

Each Gaussian will be parameterized as:
	•	Position: \mathbf{x} \in \mathbb{R}^3
	•	Scale: \mathbf{s} \in \mathbb{R}^3
	•	Rotation: R \in SO(3) (represented by quaternion)
	•	Opacity: \alpha \in (0,1)
	•	Color: c \in [0,1]^3

4.2 Camera Model
	•	Intrinsics: K (focal length, principal point)
	•	Extrinsics: [R|t] (rotation + translation from COLMAP)
	•	Projection: \mathbf{u} = K \cdot (R\mathbf{x} + t)

4.3 Rasterization in Custom Module
	1.	Projection: Transform 3D Gaussian center to screen space.
	2.	Covariance Projection: Compute 2D ellipse covariance:
\Sigma_{2D} \approx J \Sigma_{3D} J^T
where J is Jacobian of projection.
	3.	Evaluation: For each pixel p in a local 3σ bounding box, compute weight:
w(p) = \exp\left(-\tfrac{1}{2}(p-\mu)^T \Sigma^{-1}(p-\mu)\right)
	4.	Compositing: Use front-to-back alpha blending:
C_{\text{out}} = C_{\text{in}}(1 - \alpha w) + c(\alpha w)

4.4 Performance Constraints
	•	Resolution limited to ≤ 256×256 for feasibility.
	•	Small subset of Gaussians (≤ 5k) for testing.
	•	Brute-force rasterization acceptable (no acceleration structures).

⸻

5. Implementation Plan

Baseline
	•	Use reference PyTorch GS implementation.
	•	Train model for ~3–5k iterations on captured dataset.

Tiny Rasterizer (Custom Piece)
	•	Implement in Python + NumPy (no PyTorch autograd).
	•	Components:
	•	project_point() → 2D coordinates
	•	compute_covariance() → ellipse covariance
	•	render_gaussian() → per-Gaussian contribution to pixel buffer
	•	alpha_composite() → combine Gaussians front-to-back
	•	Test on synthetic toy scene (3 Gaussians) before campus data.

⸻

6. Evaluation Metrics
	•	Qualitative: Visual side-by-side of baseline render vs. Tiny Rasterizer render.
	•	Quantitative:
	•	PSNR between two images.
	•	SSIM structural similarity.
	•	Performance: Time to render 256×256 frame.

⸻

7. Risks and Mitigations
	•	COLMAP fails to register images → reduce resolution, take more photos, ensure texture-rich surfaces.
	•	Tiny Rasterizer too slow → restrict to cropped patches or fewer Gaussians.
	•	Numerical instability in covariance inversion → add ε to diagonals, use Cholesky.
	•	Time constraint → baseline by Day 6, rasterizer by Day 10.

⸻

8. Timeline

Days	Task	Deliverable
1–2	Read paper, set up repo, select scene	Repo skeleton, success criteria
3–4	Capture images, run COLMAP	Poses + dataset split
5–6	Train baseline GS	First render + metrics
7–10	Implement Tiny Rasterizer	Working toy render + campus patch render
11–12	Ablations & comparison	Metrics table, visual side-by-sides
13	Slides & report scaffold	Draft slides
14	Dry runs & integration	Final demo run
15	Final polish	Video, slides, repo release


⸻

9. Deliverables
	•	Code repo with baseline/ and custom/ modules.
	•	Demo video: turntable render.
	•	Slides: project background, method, results.
	•	Report (4–6 pages).

⸻
