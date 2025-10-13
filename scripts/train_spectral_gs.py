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

# Import gsplat's COLMAP loader (must run from gsplat/examples directory)
from datasets.colmap import Dataset as ColmapDataset, Parser

# KNN implementation (gsplat 1.5.3+ doesn't have knn in utils)
def knn(points, k):
    """Compute k-nearest neighbor distances."""
    dists = torch.cdist(points, points)
    knn_dists, _ = torch.topk(dists, k, largest=False, dim=1)
    return knn_dists

# Import spectral-gs components
from spectral_gs import SpectralStrategy, apply_view_consistent_filter, compute_spectral_entropy


def read_colmap_bin_array(path):
    """Read COLMAP binary file into numpy array."""
    with open(path, "rb") as fid:
        width, height, channels = np.fromfile(fid, np.uint64, 3)
        return np.fromfile(fid, np.float64, int(width * height * channels)).reshape((int(height), int(width), int(channels)))


def read_cameras_binary(path_to_model_file):
    """Read COLMAP cameras.bin file."""
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = np.fromfile(fid, np.uint64, 1)[0]
        for _ in range(int(num_cameras)):
            camera_id = np.fromfile(fid, np.uint32, 1)[0]
            model_id = np.fromfile(fid, np.int32, 1)[0]
            width = np.fromfile(fid, np.uint64, 1)[0]
            height = np.fromfile(fid, np.uint64, 1)[0]
            params = np.fromfile(fid, np.float64, 4)  # fx, fy, cx, cy for PINHOLE
            cameras[camera_id] = {
                'model': 'PINHOLE',
                'width': int(width),
                'height': int(height),
                'params': params  # [fx, fy, cx, cy]
            }
    return cameras


def read_images_binary(path_to_model_file):
    """Read COLMAP images.bin file."""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_images = np.fromfile(fid, np.uint64, 1)[0]
        for _ in range(int(num_images)):
            image_id = np.fromfile(fid, np.uint32, 1)[0]
            qw, qx, qy, qz = np.fromfile(fid, np.float64, 4)
            tx, ty, tz = np.fromfile(fid, np.float64, 3)
            camera_id = np.fromfile(fid, np.uint32, 1)[0]

            # Read image name
            image_name = ""
            while True:
                char = np.fromfile(fid, np.uint8, 1)[0]
                if char == 0:
                    break
                image_name += chr(char)

            # Skip 2D points
            num_points2D = np.fromfile(fid, np.uint64, 1)[0]
            np.fromfile(fid, np.float64, int(num_points2D) * 3)  # x, y, point3D_id

            images[image_id] = {
                'qvec': np.array([qw, qx, qy, qz]),
                'tvec': np.array([tx, ty, tz]),
                'camera_id': int(camera_id),
                'name': image_name
            }
    return images


def read_points3D_binary(path_to_model_file):
    """Read COLMAP points3D.bin file."""
    points3D = []
    colors = []
    with open(path_to_model_file, "rb") as fid:
        num_points = np.fromfile(fid, np.uint64, 1)[0]
        for _ in range(int(num_points)):
            point_id = np.fromfile(fid, np.uint64, 1)[0]
            xyz = np.fromfile(fid, np.float64, 3)
            rgb = np.fromfile(fid, np.uint8, 3)
            error = np.fromfile(fid, np.float64, 1)[0]

            # Skip track
            track_length = np.fromfile(fid, np.uint64, 1)[0]
            np.fromfile(fid, np.uint32, int(track_length) * 2)  # image_id, point2D_idx

            points3D.append(xyz)
            colors.append(rgb / 255.0)  # Normalize to [0, 1]

    return np.array(points3D, dtype=np.float32), np.array(colors, dtype=np.float32)


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])


class COLMAPDataset(torch.utils.data.Dataset):
    """COLMAP dataset loader that reads actual binary files."""

    def __init__(self, data_dir: str, split: str = "train", data_factor: int = 1):
        self.data_dir = Path(data_dir)
        self.split = split
        self.data_factor = data_factor

        # Load COLMAP sparse reconstruction
        sparse_dir = self.data_dir / "sparse" / "0"
        print(f"Loading COLMAP data from {sparse_dir}...")

        self.cameras = read_cameras_binary(str(sparse_dir / "cameras.bin"))
        self.colmap_images = read_images_binary(str(sparse_dir / "images.bin"))
        self.points, self.point_colors = read_points3D_binary(str(sparse_dir / "points3D.bin"))

        print(f"Loaded {len(self.cameras)} cameras")
        print(f"Loaded {len(self.colmap_images)} images")
        print(f"Loaded {len(self.points)} sparse points")

        # Load actual images
        images_dir = self.data_dir / "images"
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")

        # Filter images based on split
        split_file = self.data_dir / "splits" / f"{split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                valid_names = set(line.strip() for line in f)
        else:
            valid_names = None  # Use all images

        # Load images and their corresponding COLMAP data
        self.images = []
        self.image_names = []
        self.image_data = []  # Store COLMAP image data

        for img_id, img_data in self.colmap_images.items():
            img_name = img_data['name']

            # Check if in split
            if valid_names is not None and img_name not in valid_names:
                continue

            img_path = images_dir / img_name
            if not img_path.exists():
                print(f"Warning: Image {img_name} not found, skipping")
                continue

            # Load image
            img = imageio.imread(img_path)
            if data_factor > 1:
                img = img[::data_factor, ::data_factor]

            self.images.append(torch.from_numpy(img).float() / 255.0)
            self.image_names.append(img_name)
            self.image_data.append(img_data)

        if len(self.images) == 0:
            raise ValueError(f"No images loaded from {data_dir}")

        print(f"Loaded {len(self.images)} images for {split} split")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        img_data = self.image_data[idx]

        # Get camera parameters
        camera = self.cameras[img_data['camera_id']]
        fx, fy, cx, cy = camera['params']

        # Apply downsampling to intrinsics
        if self.data_factor > 1:
            fx /= self.data_factor
            fy /= self.data_factor
            cx /= self.data_factor
            cy /= self.data_factor

        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32)

        # Convert COLMAP pose (world-to-camera) to camera-to-world
        qvec = img_data['qvec']
        tvec = img_data['tvec']

        R = qvec2rotmat(qvec)
        # COLMAP stores world-to-camera: x_cam = R * x_world + t
        # We need camera-to-world: x_world = R^T * (x_cam - t) = R^T * x_cam - R^T * t
        R_inv = R.T
        t_inv = -R_inv @ tvec

        camtoworld = np.eye(4, dtype=np.float32)
        camtoworld[:3, :3] = R_inv
        camtoworld[:3, 3] = t_inv

        return {
            "image": image,
            "K": K,
            "camtoworld": torch.from_numpy(camtoworld),
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

    # Load dataset using gsplat's COLMAP loader (known to work with baseline)
    print(f"\nLoading dataset from {args.data_dir}...")
    parser = Parser(
        data_dir=args.data_dir,
        factor=args.data_factor,
        normalize=True,
        test_every=8,
    )
    scene_data = ColmapDataset(
        parser,
        split="train",
    )

    train_loader = DataLoader(scene_data, batch_size=1, shuffle=True, num_workers=0)

    # Initialize Gaussians from COLMAP sparse points
    print("\nInitializing Gaussians...")
    params = init_gaussians(parser.points, parser.points_rgb, init_scale=args.init_scale)
    print(f"Initialized {len(params['means'])} Gaussians")

    # Create optimizers
    optimizers = create_optimizers(params)

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

            # Compute placeholder 2D projections for strategy pre-backward check
            means_homo = torch.cat([params["means"], torch.ones_like(params["means"][:, :1])], dim=-1)
            means_cam = (viewmat[0] @ means_homo.T).T
            means_proj = (K_batched[0] @ means_cam[:, :3].T).T
            means2d_placeholder = means_proj[:, :2] / means_proj[:, 2:3]
            depths = means_cam[:, 2]
            radii_placeholder = (scales.max(dim=1)[0] * K_batched[0, 0, 0] / depths.clamp(min=0.1)).long().clamp(min=0)

            # Prepare info for strategy
            info = {
                "means2d": means2d_placeholder,
                "radii": radii_placeholder,
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

            # Update info with render_info (contains means2d with proper gradients)
            info.update(render_info)

            # Retain gradients on the ACTUAL means2d from rasterization (before backward)
            if "means2d" in info and info["means2d"] is not None:
                info["means2d"].retain_grad()

            # Compute loss (no filtering during training to preserve gradients)
            loss, metrics = compute_loss(rendered_image, image, ssim_lambda=args.ssim_lambda)

            # Backward
            for optimizer in optimizers.values():
                optimizer.zero_grad()

            loss.backward()

            # Call strategy post-backward (with warmup to let gradients establish)
            if step >= 10:  # Skip first 10 iterations
                try:
                    strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=False)

                    # Hard cap on number of Gaussians
                    n_gaussians = len(params["means"])
                    if n_gaussians > args.max_gaussians:
                        # Sort by opacity and keep top max_gaussians
                        with torch.no_grad():
                            opacities = torch.sigmoid(params["opacities"]).squeeze(-1)
                            _, indices = torch.topk(opacities, args.max_gaussians)

                            # Prune to top-k
                            for key in params.keys():
                                params[key] = torch.nn.Parameter(params[key][indices])

                            # Reset optimizers with new parameters
                            optimizers = create_optimizers(params)

                            if args.verbose:
                                print(f"  [Cap] Pruned {n_gaussians - args.max_gaussians} Gaussians (keeping top {args.max_gaussians})")

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

    Format compatible with https://antimatter15.com/splat/ and other viewers.
    """
    import struct

    # Move to CPU and convert to numpy
    means = params["means"].detach().cpu().numpy()
    scales = torch.exp(params["scales"]).detach().cpu().numpy()
    quats = F.normalize(params["quats"], dim=-1).detach().cpu().numpy()
    opacities = torch.sigmoid(params["opacities"]).detach().cpu().numpy().flatten()

    # Convert RGB colors to SH DC coefficients (proper format for 3DGS viewers)
    rgb = torch.sigmoid(params["sh0"]).squeeze(1)  # [N, 3] in [0, 1]
    C0 = 0.28209479177387814  # SH constant: 0.5 / sqrt(pi)
    sh_dc = ((rgb - 0.5) / C0).detach().cpu().numpy()  # [N, 3]

    N = means.shape[0]

    # Create PLY header
    header = f"""ply
format binary_little_endian 1.0
element vertex {N}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""

    # Write PLY file
    with open(path, 'wb') as f:
        f.write(header.encode('utf-8'))

        # Write binary data
        for i in range(N):
            # Position
            f.write(struct.pack('fff', means[i, 0], means[i, 1], means[i, 2]))

            # Normals (set to 0)
            f.write(struct.pack('fff', 0.0, 0.0, 0.0))

            # SH DC coefficients (not RGB - in SH space)
            f.write(struct.pack('fff', sh_dc[i, 0], sh_dc[i, 1], sh_dc[i, 2]))

            # Opacity
            f.write(struct.pack('f', opacities[i]))

            # Scales
            f.write(struct.pack('fff', scales[i, 0], scales[i, 1], scales[i, 2]))

            # Rotation (quaternion)
            f.write(struct.pack('ffff', quats[i, 0], quats[i, 1], quats[i, 2], quats[i, 3]))

    print(f"✅ Saved PLY with {N} Gaussians to {path}")
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
