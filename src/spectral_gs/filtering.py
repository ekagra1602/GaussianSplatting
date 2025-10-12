"""
2D view-consistent filtering for rendered images.
Based on: Spectral-GS: Taming 3D Gaussian Splatting with Spectral Entropy (SIGGRAPH Asia 2025)
"""

import torch
import torch.nn.functional as F
import math


def compute_adaptive_kernel_size(
    focal_length: float,
    depth: torch.Tensor,
    s0: float = 1.0,
    min_size: int = 3,
    max_size: int = 11
) -> int:
    """
    Compute adaptive filter kernel size based on focal length and depth.

    Formula from paper: s = s₀ * (focal²/depth²)

    Args:
        focal_length: Camera focal length in pixels
        depth: [H, W] depth map
        s0: Base kernel size parameter
        min_size: Minimum kernel size (must be odd)
        max_size: Maximum kernel size (must be odd)

    Returns:
        kernel_size: Odd integer kernel size
    """
    # Use median depth for simplicity
    median_depth = torch.median(depth[depth > 0])

    # Compute kernel size
    kernel_size = s0 * (focal_length ** 2) / (median_depth ** 2)
    kernel_size = int(torch.ceil(torch.tensor(kernel_size)).item())

    # Clamp to valid range
    kernel_size = max(min_size, min(max_size, kernel_size))

    # Ensure odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    return kernel_size


def create_gaussian_kernel(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel.

    Args:
        kernel_size: Size of the kernel (must be odd)
        sigma: Standard deviation
        device: Device to create kernel on

    Returns:
        kernel: [kernel_size, kernel_size] normalized Gaussian kernel
    """
    # Create 1D Gaussian
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
    gauss = torch.exp(-0.5 * (ax / sigma) ** 2)

    # Create 2D kernel
    kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)

    # Normalize
    kernel = kernel / kernel.sum()

    return kernel


def apply_gaussian_blur_2d(
    image: torch.Tensor,
    kernel_size: int,
    sigma: float
) -> torch.Tensor:
    """
    Apply 2D Gaussian blur to an image.

    Args:
        image: [C, H, W] or [B, C, H, W] image tensor
        kernel_size: Odd integer kernel size
        sigma: Standard deviation for Gaussian

    Returns:
        blurred: Blurred image with same shape as input
    """
    # Handle both [C, H, W] and [B, C, H, W]
    needs_squeeze = False
    if image.ndim == 3:
        image = image.unsqueeze(0)  # Add batch dimension
        needs_squeeze = True

    B, C, H, W = image.shape

    # Create Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size, sigma, image.device)

    # Reshape for conv2d: [out_channels, in_channels/groups, kH, kW]
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)

    # Apply depthwise convolution (each channel independently)
    padding = kernel_size // 2
    blurred = F.conv2d(image, kernel, padding=padding, groups=C)

    if needs_squeeze:
        blurred = blurred.squeeze(0)

    return blurred


def apply_box_filter(
    image: torch.Tensor,
    kernel_size: int
) -> torch.Tensor:
    """
    Apply box filter (mean filter) to an image.

    Args:
        image: [C, H, W] or [B, C, H, W] image tensor
        kernel_size: Odd integer kernel size

    Returns:
        filtered: Filtered image with same shape as input
    """
    # Handle both [C, H, W] and [B, C, H, W]
    needs_squeeze = False
    if image.ndim == 3:
        image = image.unsqueeze(0)
        needs_squeeze = True

    B, C, H, W = image.shape

    # Create uniform kernel
    kernel = torch.ones((kernel_size, kernel_size), device=image.device) / (kernel_size ** 2)
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)

    # Apply depthwise convolution
    padding = kernel_size // 2
    filtered = F.conv2d(image, kernel, padding=padding, groups=C)

    if needs_squeeze:
        filtered = filtered.squeeze(0)

    return filtered


def apply_view_consistent_filter(
    rendered_image: torch.Tensor,
    depth_map: torch.Tensor = None,
    focal_length: float = 1000.0,
    s0: float = 1.0,
    filter_type: str = "gaussian",
    fixed_kernel_size: int = None
) -> torch.Tensor:
    """
    Apply 2D view-consistent filtering to reduce aliasing artifacts.

    The paper combines box filter and Gaussian blur with adaptive kernel sizing
    based on focal length and depth.

    Args:
        rendered_image: [3, H, W] or [B, 3, H, W] rendered RGB image
        depth_map: [H, W] or [B, H, W] depth map (optional, used for adaptive sizing)
        focal_length: Camera focal length in pixels
        s0: Base kernel size parameter
        filter_type: "gaussian", "box", or "combined"
        fixed_kernel_size: If provided, use this instead of adaptive sizing

    Returns:
        filtered_image: Filtered image with same shape as input
    """
    # Determine kernel size
    if fixed_kernel_size is not None:
        kernel_size = fixed_kernel_size
    elif depth_map is not None:
        # Adaptive kernel based on depth
        if depth_map.ndim == 3:
            depth_map = depth_map[0]  # Use first in batch
        kernel_size = compute_adaptive_kernel_size(focal_length, depth_map, s0)
    else:
        # Default kernel size
        kernel_size = 5

    # Apply filtering
    if filter_type == "gaussian":
        sigma = kernel_size / 6.0  # Standard rule: kernel_size ≈ 6*sigma
        filtered = apply_gaussian_blur_2d(rendered_image, kernel_size, sigma)

    elif filter_type == "box":
        filtered = apply_box_filter(rendered_image, kernel_size)

    elif filter_type == "combined":
        # Apply both sequentially (box then Gaussian)
        filtered = apply_box_filter(rendered_image, kernel_size)
        sigma = kernel_size / 6.0
        filtered = apply_gaussian_blur_2d(filtered, kernel_size, sigma)

    else:
        raise ValueError(f"Unknown filter_type: {filter_type}. Use 'gaussian', 'box', or 'combined'.")

    return filtered
