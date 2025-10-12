#!/usr/bin/env python3
"""
Spectral-GS Training Script
Train 3D Gaussian Splatting with Spectral Entropy-based densification.

Based on:
- Spectral-GS: Taming 3D Gaussian Splatting with Spectral Entropy (SIGGRAPH Asia 2025)
- gsplat library (Nerfstudio)

Usage:
    python scripts/train_spectral_gs.py --data_dir data/campus_scene --result_dir results/spectral_gs
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import imageio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import gsplat
try:
    from gsplat.rendering import rasterization
    from gsplat.utils import  knn
    from gsplat import quat_scale_to_covar_preci
except ImportError:
    print("Error: gsplat not installed. Please install with: pip install gsplat")
    sys.exit(1)

# Import spectral-gs components
from spectral_gs import SpectralStrategy, apply_view_consistent_filter, compute_spectral_entropy


class COLMAPDataset(torch.utils.data.Dataset):
    """Simple COLMAP dataset loader."""

    def __init__(self, data_dir: str, split: str = "train", data_factor: int = 1):
        """
        Load COLMAP dataset.

        Args:
            data_dir: Path to dataset directory (should contain images/ and sparse/0/)
            split: "train" or "test"
            data_factor: Downsample factor for images
        """
        import pycolmap

        self.data_dir = Path(data_dir)
        self.split = split
        self.data_factor = data_factor

        # Load COLMAP reconstruction
        sparse_dir = self.data_dir / "sparse" / "0"
        if not sparse_dir.exists():
            raise ValueError(f"COLMAP sparse directory not found: {sparse_dir}")

        reconstruction = pycolmap.Reconstruction(str(sparse_dir))

        # Get images
        self.images = []
        self.cameras = []
        self.image_names = []

        # Load train/test split if available
        split_file = self.data_dir / "splits" / f"{split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                valid_names = set(line.strip() for line in f)
        else:
            # Use all images
            valid_names = None

        for image_id, image in reconstruction.images.items():
            if valid_names is not None and image.name not in valid_names:
                continue

            camera = reconstruction.cameras[image.camera_id]

            # Load image
            image_path = self.data_dir / "images" / image.name
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue

            img = imageio.imread(image_path)

            # Downsample if needed
            if data_factor > 1:
                img = img[::data_factor, ::data_factor]

            self.images.append(torch.from_numpy(img).float() / 255.0)
            self.cameras.append((camera, image, data_factor))
            self.image_names.append(image.name)

        if len(self.images) == 0:
            raise ValueError(f"No images loaded from {data_dir}")

        print(f"Loaded {len(self.images)} images for {split} split")

        # Load sparse points for initialization
        self.points = []
        self.point_colors = []
        for point_id, point in reconstruction.points3D.items():
            self.points.append(point.xyz)
            self.point_colors.append(point.color / 255.0)

        self.points = np.array(self.points)
        self.point_colors = np.array(self.point_colors)

        print(f"Loaded {len(self.points)} sparse points")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns:
            image: [H, W, 3] RGB image
            K: [3, 3] camera intrinsics
            camtoworld: [4, 4] camera-to-world transformation
        """
        image = self.images[idx]
        camera, colmap_image, factor = self.cameras[idx]

        # Build intrinsics matrix
        fx = camera.focal_length_x / factor
        fy = camera.focal_length_y / factor
        cx = camera.principal_point_x / factor
        cy = camera.principal_point_y / factor

        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32)

        # Get camera-to-world transformation
        qvec = colmap_image.qvec
        tvec = colmap_image.tvec

        # Convert to rotation matrix
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()

        # Build camera-to-world matrix
        camtoworld = np.eye(4)
        camtoworld[:3, :3] = rotation.T  # Transpose because COLMAP uses world-to-camera
        camtoworld[:3, 3] = -rotation.T @ tvec

        camtoworld = torch.from_numpy(camtoworld).float()

        return {
            "image": image,
            "K": K,
            "camtoworld": camtoworld,
            "image_name": self.image_names[idx],
        }


def init_gaussians(points: np.ndarray, colors: np.ndarray, init_scale: float = 1.0) -> Dict[str, torch.nn.Parameter]:
    """
    Initialize Gaussian parameters from sparse point cloud.

    Args:
        points: [N, 3] 3D points
        colors: [N, 3] RGB colors
        init_scale: Scale initialization factor

    Returns:
        params: Dictionary of parameters
    """
    N = points.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    points = torch.from_numpy(points).float().to(device)
    colors = torch.from_numpy(colors).float().to(device)

    # Initialize scales based on KNN distances
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    scales = torch.log(torch.sqrt(dist2_avg) * init_scale).unsqueeze(-1).repeat(1, 3)

    # Random quaternions
    quats = torch.rand((N, 4), device=device)
    quats = quats / torch.norm(quats, dim=-1, keepdim=True)

    # Initialize opacities (in logit space)
    opacities = torch.logit(torch.ones((N, 1), device=device) * 0.1)

    # Spherical harmonics (degree 0 = RGB)
    sh0 = torch.logit(colors).unsqueeze(1)  # [N, 1, 3]

    # Create parameter dict
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(points),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0),
    })

    return params


def create_optimizers(params: torch.nn.ParameterDict, lr_scale: float = 1.0) -> Dict[str, torch.optim.Optimizer]:
    """Create optimizers for Gaussian parameters."""
    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=1.6e-4 * lr_scale, eps=1e-15),
        "scales": torch.optim.Adam([params["scales"]], lr=5e-3 * lr_scale, eps=1e-15),
        "quats": torch.optim.Adam([params["quats"]], lr=1e-3 * lr_scale, eps=1e-15),
        "opacities": torch.optim.Adam([params["opacities"]], lr=5e-2 * lr_scale, eps=1e-15),
        "sh0": torch.optim.Adam([params["sh0"]], lr=2.5e-3 * lr_scale, eps=1e-15),
    }
    return optimizers


def compute_loss(
    pred_image: torch.Tensor,
    gt_image: torch.Tensor,
    ssim_lambda: float = 0.2
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute loss: L = (1 - λ) * L1 + λ * (1 - SSIM)

    Args:
        pred_image: [H, W, 3] predicted image
        gt_image: [H, W, 3] ground truth image
        ssim_lambda: Weight for SSIM loss

    Returns:
        loss: Total loss
        metrics: Dictionary of metric values
    """
    # L1 loss
    l1_loss = F.l1_loss(pred_image, gt_image)

    # SSIM loss
    # Add batch and channel dimensions for SSIM
    pred = pred_image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    gt = gt_image.permute(2, 0, 1).unsqueeze(0)

    # Simple SSIM approximation (for full SSIM, use pytorch_msssim or similar)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_pred = F.avg_pool2d(pred, 11, stride=1, padding=5)
    mu_gt = F.avg_pool2d(gt, 11, stride=1, padding=5)

    sigma_pred = F.avg_pool2d(pred ** 2, 11, stride=1, padding=5) - mu_pred ** 2
    sigma_gt = F.avg_pool2d(gt ** 2, 11, stride=1, padding=5) - mu_gt ** 2
    sigma_pred_gt = F.avg_pool2d(pred * gt, 11, stride=1, padding=5) - mu_pred * mu_gt

    ssim_map = ((2 * mu_pred * mu_gt + c1) * (2 * sigma_pred_gt + c2)) / \
               ((mu_pred ** 2 + mu_gt ** 2 + c1) * (sigma_pred + sigma_gt + c2))

    ssim_loss = 1 - ssim_map.mean()

    # Combined loss
    loss = (1 - ssim_lambda) * l1_loss + ssim_lambda * ssim_loss

    # PSNR
    mse = F.mse_loss(pred_image, gt_image)
    psnr = -10 * torch.log10(mse)

    metrics = {
        "loss": loss.item(),
        "l1": l1_loss.item(),
        "ssim": 1 - ssim_loss.item(),
        "psnr": psnr.item(),
    }

    return loss, metrics


def train(args):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    train_dataset = COLMAPDataset(args.data_dir, split="train", data_factor=args.data_factor)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    # Initialize Gaussians
    print("\nInitializing Gaussians...")
    params = init_gaussians(train_dataset.points, train_dataset.point_colors, init_scale=args.init_scale)
    print(f"Initialized {len(params['means'])} Gaussians")

    # Create optimizers
    optimizers = create_optimizers(params)

    # Create strategy
    print("\nInitializing Spectral-GS strategy...")
    strategy = SpectralStrategy(
        prune_opa=args.prune_opa,
        grow_grad2d=args.grow_grad2d,
        grow_scale3d=args.grow_scale3d,
        prune_scale3d=args.prune_scale3d,
        refine_start_iter=args.refine_start_iter,
        refine_stop_iter=args.refine_stop_iter,
        reset_every=args.reset_every,
        refine_every=args.refine_every,
        spectral_threshold=args.spectral_threshold,
        enable_spectral_splitting=args.enable_spectral_splitting,
        spectral_split_factor=args.spectral_split_factor,
        verbose=args.verbose,
    )

    # Initialize strategy state
    strategy_state = strategy.initialize_state()

    # Training loop
    print(f"\nStarting training for {args.max_steps} steps...")
    print("=" * 80)

    step = 0
    pbar = tqdm(total=args.max_steps)

    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break

            # Move batch to device
            image = batch["image"][0].to(device)  # [H, W, 3]
            K = batch["K"][0].to(device)  # [3, 3]
            camtoworld = batch["camtoworld"][0].to(device)  # [4, 4]

            H, W = image.shape[:2]

            # Rasterize
            colors = torch.sigmoid(params["sh0"])  # [N, 1, 3]
            opacities = torch.sigmoid(params["opacities"])  # [N, 1]
            scales = torch.exp(params["scales"])  # [N, 3]
            quats = F.normalize(params["quats"], dim=-1)  # [N, 4]

            # Prepare for rendering
            viewmat = torch.linalg.inv(camtoworld).unsqueeze(0)  # [1, 4, 4]
            K_batched = K.unsqueeze(0)  # [1, 3, 3]

            # Call strategy pre-backward
            info = {}
            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

            # Rasterize
            renders, alphas, render_info = rasterization(
                means=params["means"],
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmat,
                Ks=K_batched,
                width=W,
                height=H,
                packed=False,
                absgrad=(step < args.refine_stop_iter and strategy.absgrad),
            )

            rendered_image = renders[0]  # [H, W, 3]

            # Apply view-consistent filtering (optional, only during evaluation)
            if args.enable_filtering and step % args.log_every == 0:
                with torch.no_grad():
                    filtered_image = apply_view_consistent_filter(
                        rendered_image.permute(2, 0, 1),  # [3, H, W]
                        focal_length=K[0, 0].item(),
                        filter_type=args.filter_type,
                    )
                    filtered_image = filtered_image.permute(1, 2, 0)  # [H, W, 3]
            else:
                filtered_image = rendered_image

            # Compute loss
            loss, metrics = compute_loss(filtered_image, image, ssim_lambda=args.ssim_lambda)

            # Backward
            for optimizer in optimizers.values():
                optimizer.zero_grad()

            loss.backward()

            # Call strategy post-backward
            info.update(render_info)
            strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=False)

            # Optimizer step
            for optimizer in optimizers.values():
                optimizer.step()

            # Logging
            if step % args.log_every == 0:
                n_gaussians = len(params["means"])

                # Compute spectral entropy statistics
                with torch.no_grad():
                    spectral_entropies = compute_spectral_entropy(scales, quats)
                    mean_entropy = spectral_entropies.mean().item()
                    low_entropy = (spectral_entropies < args.spectral_threshold).sum().item()

                pbar.set_description(
                    f"Step {step:05d} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"PSNR: {metrics['psnr']:.2f} | "
                    f"Gaussians: {n_gaussians} | "
                    f"Entropy: {mean_entropy:.3f} | "
                    f"Needles: {low_entropy}"
                )

            # Save checkpoint
            if args.save_ply and step > 0 and step % args.save_every == 0:
                save_checkpoint(params, result_dir / f"checkpoint_{step:06d}.pt")

            step += 1
            pbar.update(1)

            if step >= args.max_steps:
                break

    pbar.close()
    print("\n" + "=" * 80)
    print("Training completed!")

    # Save final model
    if args.save_ply:
        print(f"\nSaving final checkpoint to {result_dir}/final.pt")
        save_checkpoint(params, result_dir / "final.pt")

        # Also save as PLY
        save_ply(params, result_dir / "final.ply")

    print(f"\nResults saved to: {result_dir}")


def save_checkpoint(params: torch.nn.ParameterDict, path: Path):
    """Save model checkpoint."""
    checkpoint = {k: v.detach().cpu() for k, v in params.items()}
    torch.save(checkpoint, path)


def save_ply(params: torch.nn.ParameterDict, path: Path):
    """Save Gaussians as PLY file."""
    # This is a simplified PLY export
    # For full export, use gsplat's export utilities
    print(f"PLY export not implemented yet. Saved checkpoint to {path.parent}")


def main():
    parser = argparse.ArgumentParser(description="Train Spectral-GS on COLMAP dataset")

    # Data args
    parser.add_argument("--data_dir", type=str, required=True, help="Path to COLMAP dataset")
    parser.add_argument("--result_dir", type=str, required=True, help="Path to save results")
    parser.add_argument("--data_factor", type=int, default=1, help="Downsample factor for images")

    # Training args
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--init_scale", type=float, default=1.0, help="Initial scale factor")
    parser.add_argument("--ssim_lambda", type=float, default=0.2, help="SSIM loss weight")

    # Strategy args
    parser.add_argument("--prune_opa", type=float, default=0.005, help="Opacity pruning threshold")
    parser.add_argument("--grow_grad2d", type=float, default=0.0002, help="2D gradient grow threshold")
    parser.add_argument("--grow_scale3d", type=float, default=0.01, help="3D scale grow threshold")
    parser.add_argument("--prune_scale3d", type=float, default=0.1, help="3D scale prune threshold")
    parser.add_argument("--refine_start_iter", type=int, default=500, help="Start refinement iteration")
    parser.add_argument("--refine_stop_iter", type=int, default=15000, help="Stop refinement iteration")
    parser.add_argument("--reset_every", type=int, default=3000, help="Reset frequency")
    parser.add_argument("--refine_every", type=int, default=100, help="Refinement frequency")

    # Spectral-GS args
    parser.add_argument("--spectral_threshold", type=float, default=0.5, help="Spectral entropy threshold (τ)")
    parser.add_argument("--enable_spectral_splitting", action="store_true", default=True, help="Enable spectral splitting")
    parser.add_argument("--spectral_split_factor", type=float, default=1.6, help="Scale reduction factor for splitting")

    # Filtering args
    parser.add_argument("--enable_filtering", action="store_true", help="Enable view-consistent filtering")
    parser.add_argument("--filter_type", type=str, default="gaussian", choices=["gaussian", "box", "combined"], help="Filter type")

    # Logging args
    parser.add_argument("--log_every", type=int, default=100, help="Logging frequency")
    parser.add_argument("--save_every", type=int, default=5000, help="Checkpoint save frequency")
    parser.add_argument("--save_ply", action="store_true", default=True, help="Save PLY files")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Validate
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Run training
    train(args)


if __name__ == "__main__":
    main()
