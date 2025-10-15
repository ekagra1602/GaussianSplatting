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

# Import gsplat (with compatibility for 1.5.3+)
from gsplat.rendering import rasterization

# Import gsplat's COLMAP loader (must run from gsplat/examples directory)
from datasets.colmap import Dataset as ColmapDataset, Parser

# Note: Using gsplat's KNN from utils (sklearn-based)

# Import spectral-gs components
from spectral_gs import SpectralStrategy, apply_view_consistent_filter, compute_spectral_entropy

# Import gsplat utils
from utils import rgb_to_sh, knn as knn_sklearn


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

    # Initialize scales based on KNN distances (using gsplat's KNN)
    dist2_avg = (knn_sklearn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    scales = torch.log(torch.sqrt(dist2_avg) * init_scale).unsqueeze(-1).repeat(1, 3)

    # Random quaternions
    quats = torch.rand((N, 4), device=device)
    quats = quats / torch.norm(quats, dim=-1, keepdim=True)

    # Initialize opacities (in logit space) - [N] not [N, 1] to match baseline
    opacities = torch.logit(torch.full((N,), 0.1, device=device))

    # Spherical harmonics (degree 0 = RGB) - use rgb_to_sh like baseline
    sh0 = rgb_to_sh(colors).unsqueeze(1)  # [N, 1, 3] in SH space
    shN = torch.zeros((N, 0, 3), device=device)  # Empty higher-order SH (degree 0 only)

    # Create parameter dict
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(points),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shN),
    })

    return params


def create_optimizers(params: torch.nn.ParameterDict, lr_scale: float = 1.0) -> Dict[str, torch.optim.Optimizer]:
    """Create optimizers for Gaussian parameters.

    Args:
        params: Parameter dictionary
        lr_scale: Scale factor for means learning rate (should be scene_scale)
    """
    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=1.6e-4 * lr_scale, eps=1e-15),  # Only means LR is scaled!
        "scales": torch.optim.Adam([params["scales"]], lr=5e-3, eps=1e-15),
        "quats": torch.optim.Adam([params["quats"]], lr=1e-3, eps=1e-15),
        "opacities": torch.optim.Adam([params["opacities"]], lr=5e-2, eps=1e-15),
        "sh0": torch.optim.Adam([params["sh0"]], lr=2.5e-3, eps=1e-15),
        "shN": torch.optim.Adam([params["shN"]], lr=2.5e-3 / 20, eps=1e-15),
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

    # Load dataset using gsplat's COLMAP loader (exactly like baseline)
    print(f"\nLoading dataset from {args.data_dir}...")
    parser = Parser(
        data_dir=args.data_dir,
        factor=args.data_factor,
        normalize=True,
        test_every=8,
    )
    trainset = ColmapDataset(
        parser,
        split="train",
    )

    train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)

    # Get scene scale from parser (CRITICAL for proper LR scaling!)
    scene_scale = parser.scene_scale * 1.1
    print(f"Scene scale: {scene_scale}")

    # Initialize Gaussians from COLMAP sparse points
    print("\nInitializing Gaussians...")
    params = init_gaussians(parser.points, parser.points_rgb / 255.0, init_scale=args.init_scale)
    print(f"Initialized {len(params['means'])} Gaussians")

    # Create optimizers with scene_scale adjusted learning rates
    optimizers = create_optimizers(params, lr_scale=scene_scale)

    # Create strategy with aggressive pruning
    print("\nInitializing Spectral-GS strategy...")
    print(f"   Max Gaussians: {args.max_gaussians}")
    print(f"   Spectral splitting: {'ENABLED' if args.enable_spectral_splitting else 'DISABLED'}")
    print(f"   Opacity pruning threshold: {args.prune_opa}")
    print(f"   Scale pruning threshold: {args.prune_scale3d}")

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

            # Move batch to device (image is in [0, 255] range from gsplat's Dataset)
            image = batch["image"][0].to(device) / 255.0  # [H, W, 3] normalize to [0, 1]
            K = batch["K"][0].to(device)  # [3, 3]
            camtoworld = batch["camtoworld"][0].to(device)  # [4, 4]

            H, W = image.shape[:2]

            # Prepare Gaussian parameters for rendering (match baseline exactly)
            # Concatenate sh0 and shN like baseline (shN is empty for degree 0)
            colors = torch.cat([params["sh0"], params["shN"]], 1)  # [N, 1, 3] in SH space
            opacities = torch.sigmoid(params["opacities"])  # [N]
            scales = torch.exp(params["scales"])  # [N, 3]
            quats = params["quats"]  # [N, 4] - gsplat normalizes internally!

            # Prepare for rendering
            viewmat = torch.linalg.inv(camtoworld).unsqueeze(0)  # [1, 4, 4]
            K_batched = K.unsqueeze(0)  # [1, 3, 3]

            # Compute placeholder 2D projections for strategy pre-backward check
            means_homo = torch.cat([params["means"], torch.ones_like(params["means"][:, :1])], dim=-1)
            means_cam = (viewmat[0] @ means_homo.T).T
            means_proj = (K_batched[0] @ means_cam[:, :3].T).T
            means2d_placeholder = means_proj[:, :2] / means_proj[:, 2:3]
            depths = means_cam[:, 2]
            radii_placeholder = (scales.max(dim=1)[0] * K_batched[0, 0, 0] / depths.clamp(min=0.1)).long().clamp(min=0).unsqueeze(0)

            # Prepare info for strategy
            info = {
                "means2d": means2d_placeholder,
                "radii": radii_placeholder,
                "width": W,
                "height": H,
            }

            # Call strategy pre-backward
            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

            # Rasterize (must pass sh_degree when colors are in SH format [N, K, 3])
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
                sh_degree=0,  # We have degree 0 (K=1, DC component only)
                packed=False,
                absgrad=(step < args.refine_stop_iter and strategy.absgrad),
            )

            rendered_image = renders[0]  # [H, W, 3]

            # Update info with render_info (contains means2d with proper gradients)
            info.update(render_info)

            # Retain gradients on the ACTUAL means2d from rasterization (before backward)
            if "means2d" in info and info["means2d"] is not None:
                info["means2d"].retain_grad()

            # Compute loss - TEMPORARILY USE PURE L1 TO DEBUG
            l1loss = F.l1_loss(renders[0], image)
            loss = l1loss

            mse = F.mse_loss(renders[0], image)
            psnr = -10.0 * torch.log10(mse)

            metrics = {
                "loss": loss.item(),
                "l1": l1loss.item(),
                "psnr": psnr.item(),
            }

            # Backward
            for optimizer in optimizers.values():
                optimizer.zero_grad()

            loss.backward()

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

            # Hard cap on number of Gaussians (AFTER strategy and optimizer)
            n_gaussians = len(params["means"])
            if n_gaussians > args.max_gaussians:
                # Sort by opacity and keep top max_gaussians
                with torch.no_grad():
                    opacities_sorted = torch.sigmoid(params["opacities"])  # [N]
                    _, indices = torch.topk(opacities_sorted, args.max_gaussians)

                    # Prune to top-k
                    for key in params.keys():
                        params[key] = torch.nn.Parameter(params[key][indices])

                    # Reset optimizers with new parameters (with scene_scale!)
                    optimizers = create_optimizers(params, lr_scale=scene_scale)

                    # Reset strategy state to match new Gaussian count
                    strategy_state = strategy.initialize_state()

                    if args.verbose:
                        print(f"  [Cap] Pruned {n_gaussians - args.max_gaussians} Gaussians (keeping top {args.max_gaussians})")

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

            step += 1
            pbar.update(1)

            if step >= args.max_steps:
                break

    pbar.close()
    print("\n" + "=" * 80)
    print("Training completed!")

    # Save final model
    print(f"\n💾 Saving final model...")
    save_checkpoint(params, result_dir / "final.pt")
    save_ply(params, result_dir / "final.ply")

    print(f"\n✅ Results saved to: {result_dir}")
    print(f"   - final.pt: PyTorch checkpoint")
    print(f"   - final.ply: Viewable 3D Gaussian model")
    print(f"\n📌 View your scene at: https://antimatter15.com/splat/")
    print(f"   Just drag and drop the final.ply file!")


def save_checkpoint(params: torch.nn.ParameterDict, path: Path):
    """Save model checkpoint."""
    checkpoint = {k: v.detach().cpu() for k, v in params.items()}
    torch.save(checkpoint, path)


def save_ply(params: torch.nn.ParameterDict, path: Path):
    """
    Save Gaussian model as PLY file for viewing in 3DGS viewers.
    Uses gsplat's export_splats for correct formatting.
    """
    from gsplat import export_splats

    # export_splats expects raw parameters (log-space scales, logit-space opacities)
    means = params["means"]  # [N, 3]
    scales = params["scales"]  # [N, 3] in log space
    quats = params["quats"]  # [N, 4]
    opacities = params["opacities"]  # [N] in logit space
    sh0 = params["sh0"]  # [N, 1, 3]
    shN = params["shN"]  # [N, 0, 3] for degree 0

    export_splats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        format="ply",
        save_to=str(path),
    )

    print(f"✅ Saved PLY with {len(means)} Gaussians to {path}")
    print(f"   File size: {path.stat().st_size / (1024*1024):.1f} MB")


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
    parser.add_argument("--prune_opa", type=float, default=0.05, help="Opacity pruning threshold (higher = more aggressive)")
    parser.add_argument("--grow_grad2d", type=float, default=0.0002, help="2D gradient grow threshold")
    parser.add_argument("--grow_scale3d", type=float, default=0.01, help="3D scale grow threshold")
    parser.add_argument("--prune_scale3d", type=float, default=0.5, help="3D scale prune threshold (higher = more aggressive)")
    parser.add_argument("--refine_start_iter", type=int, default=500, help="Start refinement iteration")
    parser.add_argument("--refine_stop_iter", type=int, default=15000, help="Stop refinement iteration")
    parser.add_argument("--reset_every", type=int, default=3000, help="Reset frequency")
    parser.add_argument("--refine_every", type=int, default=100, help="Refinement frequency")
    parser.add_argument("--max_gaussians", type=int, default=300000, help="Hard cap on number of Gaussians")

    # Spectral-GS args
    parser.add_argument("--spectral_threshold", type=float, default=0.3, help="Spectral entropy threshold (lower = more needles split)")
    parser.add_argument("--enable_spectral_splitting", action="store_true", default=False, help="Enable spectral splitting")
    parser.add_argument("--spectral_split_factor", type=float, default=1.6, help="Scale reduction factor for splitting")

    # Filtering args (disabled during training)
    parser.add_argument("--enable_filtering", action="store_true", help="Enable view-consistent filtering (disabled for training)")
    parser.add_argument("--filter_type", type=str, default="gaussian", choices=["gaussian", "box", "combined"], help="Filter type")

    # Logging args
    parser.add_argument("--log_every", type=int, default=100, help="Logging frequency")
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
