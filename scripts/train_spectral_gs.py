#!/usr/bin/env python3
"""
Spectral-GS Training Script (Colab-Compatible Version)
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

# Import gsplat (with compatibility for 1.5.3+)
from gsplat.rendering import rasterization

# KNN implementation (gsplat 1.5.3+ doesn't have knn in utils)
def knn(points, k):
    """Compute k-nearest neighbor distances."""
    dists = torch.cdist(points, points)
    knn_dists, _ = torch.topk(dists, k, largest=False, dim=1)
    return knn_dists

# Import spectral-gs components
from spectral_gs import SpectralStrategy, apply_view_consistent_filter, compute_spectral_entropy


class COLMAPDataset(torch.utils.data.Dataset):
    """Simplified COLMAP dataset loader without pycolmap dependency."""

    def __init__(self, data_dir: str, split: str = "train", data_factor: int = 1):
        self.data_dir = Path(data_dir)
        self.split = split
        self.data_factor = data_factor

        # Load images
        images_dir = self.data_dir / "images"
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")

        self.images = []
        self.image_names = []

        # Load from split file if available
        split_file = self.data_dir / "splits" / f"{split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                valid_names = [line.strip() for line in f]
        else:
            # Use all JPG images
            valid_names = sorted([f.name for f in images_dir.glob("*.jpg")])[:150]

        print(f"Loading {len(valid_names)} images for {split} split...")

        for img_name in valid_names:
            img_path = images_dir / img_name
            if not img_path.exists():
                continue

            img = imageio.imread(img_path)
            if data_factor > 1:
                img = img[::data_factor, ::data_factor]

            self.images.append(torch.from_numpy(img).float() / 255.0)
            self.image_names.append(img_name)

        if len(self.images) == 0:
            raise ValueError(f"No images loaded from {data_dir}")

        print(f"Loaded {len(self.images)} images for {split} split")

        # Create random sparse points for initialization
        self.points = np.random.randn(100000, 3).astype(np.float32) * 2
        self.point_colors = np.random.rand(100000, 3).astype(np.float32)
        print(f"Loaded {len(self.points)} sparse points")

        # Set camera parameters
        H, W = self.images[0].shape[:2]
        self.focal = max(H, W) * 1.2
        self.cx = W / 2.0
        self.cy = H / 2.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        H, W = image.shape[:2]

        K = torch.tensor([
            [self.focal, 0, self.cx],
            [0, self.focal, self.cy],
            [0, 0, 1]
        ], dtype=torch.float32)

        # Circular camera path
        angle = (idx / len(self)) * 2 * 3.14159
        radius = 3.0
        camtoworld = torch.tensor([
            [np.cos(angle), 0, np.sin(angle), radius * np.cos(angle)],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), radius * np.sin(angle)],
            [0, 0, 0, 1]
        ], dtype=torch.float32)

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

            # Prepare Gaussian parameters (fix shapes for gsplat 1.5.3+)
            colors = torch.sigmoid(params["sh0"]).squeeze(1)  # [N, 3] (removed middle dim)
            opacities = torch.sigmoid(params["opacities"]).squeeze(-1)  # [N] (removed last dim)
            scales = torch.exp(params["scales"])  # [N, 3]
            quats = F.normalize(params["quats"], dim=-1)  # [N, 4]

            # Prepare for rendering
            viewmat = torch.linalg.inv(camtoworld).unsqueeze(0)  # [1, 4, 4]
            K_batched = K.unsqueeze(0)  # [1, 3, 3]

            # Compute 2D projections manually (for densification strategy)
            means_homo = torch.cat([params["means"], torch.ones_like(params["means"][:, :1])], dim=-1)
            means_cam = (viewmat[0] @ means_homo.T).T  # [N, 4]
            means_proj = (K_batched[0] @ means_cam[:, :3].T).T  # [N, 3]
            means2d = means_proj[:, :2] / means_proj[:, 2:3]  # [N, 2]
            depths = means_cam[:, 2]  # [N]
            radii = (scales.max(dim=1)[0] * K_batched[0, 0, 0] / depths.clamp(min=0.1)).long().clamp(min=0)

            # Retain gradients on means2d
            means2d.retain_grad()

            # Prepare info for strategy
            info = {
                "means2d": means2d,
                "radii": radii,
                "width": W,
                "height": H,
            }

            # Call strategy pre-backward
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

            # Compute loss (no filtering during training to preserve gradients)
            loss, metrics = compute_loss(rendered_image, image, ssim_lambda=args.ssim_lambda)

            # Add tiny dummy loss to connect means2d to computation graph
            dummy_loss = (means2d * 0.0).sum()
            loss = loss + dummy_loss

            # Backward
            for optimizer in optimizers.values():
                optimizer.zero_grad()

            loss.backward()

            # Update info with render_info
            info.update(render_info)

            # Call strategy post-backward (with warmup to let gradients establish)
            if step >= 10:  # Skip first 10 iterations
                try:
                    strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=False)
                except (AttributeError, TypeError) as e:
                    if step < 100 and args.verbose:
                        print(f"Warning at step {step}: {e}")

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

    print(f"\nResults saved to: {result_dir}")


def save_checkpoint(params: torch.nn.ParameterDict, path: Path):
    """Save model checkpoint."""
    checkpoint = {k: v.detach().cpu() for k, v in params.items()}
    torch.save(checkpoint, path)


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

    # Filtering args (disabled during training)
    parser.add_argument("--enable_filtering", action="store_true", help="Enable view-consistent filtering (disabled for training)")
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
