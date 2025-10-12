"""
Spectral entropy computation for 3D Gaussians.
Based on: Spectral-GS: Taming 3D Gaussian Splatting with Spectral Entropy (SIGGRAPH Asia 2025)
"""

import torch
from typing import Tuple

def quat_to_rotation_matrix(quats: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices.

    Args:
        quats: [..., 4] quaternions (w, x, y, z)

    Returns:
        rotmats: [..., 3, 3] rotation matrices
    """
    # Normalize quaternions
    quats = quats / (torch.norm(quats, dim=-1, keepdim=True) + 1e-10)

    w, x, y, z = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]

    # Compute rotation matrix elements
    R = torch.stack([
        torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=-1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=-1)
    ], dim=-2)

    return R


def scales_quats_to_covariance(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """
    Convert scales and quaternions to 3D covariance matrices.
    Covariance = R @ S @ S^T @ R^T where R is rotation, S is scale matrix.

    Args:
        scales: [..., 3] scale factors
        quats: [..., 4] quaternions

    Returns:
        covariances: [..., 3, 3] covariance matrices
    """
    # Get rotation matrices
    R = quat_to_rotation_matrix(quats)  # [..., 3, 3]

    # Create diagonal scale matrices
    # S @ S^T = diag(s)^2
    S_squared = torch.diag_embed(scales ** 2)  # [..., 3, 3]

    # Compute covariance: R @ S^2 @ R^T
    covariance = torch.matmul(torch.matmul(R, S_squared), R.transpose(-2, -1))

    return covariance


def compute_spectral_entropy(
    scales: torch.Tensor,
    quats: torch.Tensor,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Compute spectral entropy from 3D Gaussian covariance matrix.

    Formula: H(Σ) = -∑(sᵢ²/tr(Σ) * ln(sᵢ²/tr(Σ)))
    where sᵢ² are the eigenvalues of the covariance matrix.

    For efficiency, since Σ = RSS^TR^T and eigenvalues are rotation-invariant,
    the eigenvalues of Σ are just the squared scales: [s₀², s₁², s₂²]

    Args:
        scales: [N, 3] scale factors for N Gaussians
        quats: [N, 4] quaternions (not used in computation but kept for API consistency)
        eps: Small epsilon for numerical stability

    Returns:
        entropy: [N] spectral entropy values
    """
    # Eigenvalues of covariance matrix = squared scales (rotation doesn't affect eigenvalues)
    eigenvalues = scales ** 2  # [N, 3]

    # Clamp for numerical stability
    eigenvalues = torch.clamp(eigenvalues, min=eps)

    # Compute trace
    trace = eigenvalues.sum(dim=-1, keepdim=True)  # [N, 1]

    # Normalized eigenvalues (probability distribution)
    normalized = eigenvalues / trace  # [N, 3]

    # Spectral entropy: -∑(pᵢ * ln(pᵢ))
    # Add small epsilon to prevent log(0)
    log_normalized = torch.log(normalized + eps)
    entropy = -(normalized * log_normalized).sum(dim=-1)  # [N]

    return entropy


def compute_covariance_from_scales_quats(
    scales: torch.Tensor,
    quats: torch.Tensor
) -> torch.Tensor:
    """
    Compute full 3D covariance matrices from scales and quaternions.
    This is useful for debugging and visualization.

    Args:
        scales: [N, 3] scale factors
        quats: [N, 4] quaternions

    Returns:
        covariances: [N, 3, 3] covariance matrices
    """
    return scales_quats_to_covariance(scales, quats)


def split_gaussian_spectral(
    mean: torch.Tensor,
    scale: torch.Tensor,
    quat: torch.Tensor,
    opacity: torch.Tensor,
    scale_factor: float = 1.6,
    num_splits: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split a single Gaussian with spectral awareness.

    The splitting strategy:
    1. Reduce covariance anisotropically (divide scales by scale_factor)
    2. Sample new positions along the principal axis (largest scale direction)

    Args:
        mean: [3] center position
        scale: [3] scale factors
        quat: [4] quaternion
        opacity: [] opacity value
        scale_factor: Factor to reduce scales by (default: 1.6 per paper)
        num_splits: Number of new Gaussians to create (default: 2)

    Returns:
        new_means: [num_splits, 3]
        new_scales: [num_splits, 3]
        new_quats: [num_splits, 4]
        new_opacities: [num_splits]
    """
    # Reduce scales anisotropically
    new_scales = scale / scale_factor

    # Find principal direction (direction of largest scale)
    principal_idx = torch.argmax(scale)
    R = quat_to_rotation_matrix(quat)  # [3, 3]
    principal_dir = R[:, principal_idx]  # [3]

    # Sample positions along principal direction
    std_dev = scale[principal_idx]
    offsets = torch.linspace(-0.5, 0.5, num_splits, device=mean.device)
    new_means = mean.unsqueeze(0) + offsets.unsqueeze(-1) * principal_dir.unsqueeze(0) * std_dev

    # All children get same reduced scales and rotation
    new_scales = new_scales.unsqueeze(0).repeat(num_splits, 1)
    new_quats = quat.unsqueeze(0).repeat(num_splits, 1)

    # Reduce opacity (split mass among children)
    new_opacities = (opacity / num_splits).unsqueeze(0).repeat(num_splits)

    return new_means, new_scales, new_quats, new_opacities


def duplicate_gaussian(
    mean: torch.Tensor,
    scale: torch.Tensor,
    quat: torch.Tensor,
    opacity: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Duplicate a Gaussian (used for small Gaussians with high gradients).

    Args:
        mean: [3] center position
        scale: [3] scale factors
        quat: [4] quaternion
        opacity: [] opacity value

    Returns:
        new_mean: [3]
        new_scale: [3]
        new_quat: [4]
        new_opacity: []
    """
    # Simply return the same parameters (duplication)
    return mean, scale, quat, opacity
