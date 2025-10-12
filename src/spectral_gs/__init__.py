"""
Spectral-GS: Taming 3D Gaussian Splatting with Spectral Entropy
Implementation based on the SIGGRAPH Asia 2025 paper by Huang et al.
"""

from .spectral_entropy import compute_spectral_entropy
from .spectral_strategy import SpectralStrategy
from .filtering import apply_view_consistent_filter

__all__ = [
    "compute_spectral_entropy",
    "SpectralStrategy",
    "apply_view_consistent_filter",
]
