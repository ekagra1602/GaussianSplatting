# Gaussian Splatting - Reconstruction of a part of Campus

A focused implementation to reconstruct a chosen part of campus using 3D Gaussian Splatting (Kerbl et al., 2023). One core component will be implemented from scratch (no external libraries), even if it is an optimization submodule, to demonstrate understanding.

## Objectives
- Reconstruct any specific area of campus from multi-view images.
- Implement one key aspect without libraries (e.g., minimal CPU/NumPy splat rasterizer, simple optimization loop, or visibility/sorting micro-impl).
- Explain the approach clearly in the 9 Oct meeting and prepare for a later presentation.

## Paper Focus
- Primary reference: 3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., 2023).
- From-scratch candidate modules (to finalize after initial experiments):
  - Minimal Gaussian splat rasterizer with alpha compositing, or
  - Simple optimization loop updating positions/scales/colors without DL frameworks, or
  - Tile-based binning and sorting for a small image.
  
## Repo Structure (proposed)
- data/ — images and calibration
- src/ — splatting and optimization code
- notebooks/ — experiments/visualizations
- results/ — renderings and videos
