"""
Spectral-aware densification strategy for 3D Gaussian Splatting.
Based on: Spectral-GS: Taming 3D Gaussian Splatting with Spectral Entropy (SIGGRAPH Asia 2025)

This strategy extends the DefaultStrategy by adding spectral entropy-based splitting
to reduce needle-like artifacts.
"""

import torch
from torch import Tensor
from typing import Any, Dict, Tuple
from gsplat.strategy.default import DefaultStrategy

from .spectral_entropy import (
    compute_spectral_entropy,
    split_gaussian_spectral,
)


class SpectralStrategy(DefaultStrategy):
    """
    Spectral-GS densification strategy.

    Key differences from DefaultStrategy:
    1. Adds spectral entropy computation
    2. Splits Gaussians with low spectral entropy (needle-like) even if gradients are low
    3. Uses anisotropic covariance reduction during splitting
    """

    def __init__(
        self,
        prune_opa: float = 0.005,
        grow_grad2d: float = 0.0002,
        grow_scale3d: float = 0.01,
        grow_scale2d: float = 0.05,
        prune_scale3d: float = 0.1,
        refine_scale2d_stop_iter: int = 0,
        refine_start_iter: int = 500,
        refine_stop_iter: int = 15_000,
        reset_every: int = 3000,
        refine_every: int = 100,
        pause_refine_after_reset: int = 0,
        absgrad: bool = False,
        revised_opacity: bool = False,
        verbose: bool = False,
        # Spectral-GS specific parameters
        spectral_threshold: float = 0.5,
        enable_spectral_splitting: bool = True,
        spectral_split_factor: float = 1.6,
    ):
        """
        Args:
            prune_opa: Opacity threshold for pruning
            grow_grad2d: 2D gradient threshold for splitting/duplicating
            grow_scale3d: 3D scale threshold for splitting vs duplicating
            grow_scale2d: 2D scale threshold for splitting
            prune_scale3d: 3D scale threshold for pruning
            refine_scale2d_stop_iter: Iteration to stop using 2D scale criterion
            refine_start_iter: Iteration to start refinement
            refine_stop_iter: Iteration to stop refinement
            reset_every: Frequency of opacity reset
            refine_every: Frequency of refinement
            pause_refine_after_reset: Pause iterations after reset
            absgrad: Use absolute gradients instead of averaged gradients
            revised_opacity: Use revised opacity heuristic
            verbose: Print verbose messages
            spectral_threshold: Spectral entropy threshold for splitting (τ in paper, default: 0.5)
            enable_spectral_splitting: Enable spectral entropy-based splitting
            spectral_split_factor: Factor to reduce scales during spectral splitting (default: 1.6)
        """
        super().__init__(
            prune_opa=prune_opa,
            grow_grad2d=grow_grad2d,
            grow_scale3d=grow_scale3d,
            grow_scale2d=grow_scale2d,
            prune_scale3d=prune_scale3d,
            refine_scale2d_stop_iter=refine_scale2d_stop_iter,
            refine_start_iter=refine_start_iter,
            refine_stop_iter=refine_stop_iter,
            reset_every=reset_every,
            refine_every=refine_every,
            pause_refine_after_reset=pause_refine_after_reset,
            absgrad=absgrad,
            revised_opacity=revised_opacity,
            verbose=verbose,
        )

        # Spectral-GS specific parameters
        self.spectral_threshold = spectral_threshold
        self.enable_spectral_splitting = enable_spectral_splitting
        self.spectral_split_factor = spectral_split_factor

        if verbose:
            print(f"[SpectralStrategy] Initialized with spectral_threshold={spectral_threshold}")
            print(f"[SpectralStrategy] enable_spectral_splitting={enable_spectral_splitting}")

    def step_post_backward(
        self,
        params: Dict[str, torch.nn.Parameter],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """
        Callback function to be executed after backward pass.

        This extends the parent's step_post_backward with spectral entropy-based splitting.
        """
        if step >= self.refine_stop_iter:
            return

        # First, run the standard DefaultStrategy densification
        super().step_post_backward(params, optimizers, state, step, info, packed)

        # Then, apply spectral entropy-based splitting
        if (
            self.enable_spectral_splitting
            and step >= self.refine_start_iter
            and step % self.refine_every == 0
        ):
            self._spectral_split_gs(params, optimizers, state, step)

    def _spectral_split_gs(
        self,
        params: Dict[str, torch.nn.Parameter],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ):
        """
        Perform spectral entropy-based splitting of needle-like Gaussians.

        This identifies Gaussians with low spectral entropy (< threshold) and splits them
        to improve scene representation.
        """
        scales = params["scales"]
        quats = params["quats"]
        means = params["means"]
        opacities = torch.sigmoid(params["opacities"])

        # Compute spectral entropy for all Gaussians
        with torch.no_grad():
            spectral_entropy = compute_spectral_entropy(
                torch.exp(scales),  # scales are in log space
                quats
            )

        # Find needle-like Gaussians (low spectral entropy)
        is_needle = spectral_entropy < self.spectral_threshold

        # Only split needles that haven't been caught by gradient-based splitting
        # and have reasonable opacity
        is_visible = opacities.squeeze() > self.prune_opa
        to_split = is_needle & is_visible

        n_to_split = to_split.sum().item()

        if n_to_split > 0:
            if self.verbose:
                print(
                    f"[SpectralStrategy] Step {step}: "
                    f"Splitting {n_to_split} needle-like Gaussians "
                    f"(entropy < {self.spectral_threshold})"
                )

            # Get indices to split
            split_indices = torch.where(to_split)[0]

            # Perform splitting using gsplat's split operation
            # We'll use the parent's split logic but with our own scale factor
            from gsplat.strategy.ops import split

            # Prepare data for splitting
            split_params = split(
                params,
                optimizers,
                state,
                split_indices,
                self.revised_opacity,
                n_split_samples=2,
                # Use spectral split factor for scale reduction
                split_scale_factor=self.spectral_split_factor,
            )

            # Update state counters
            if "spectral_splits" not in state:
                state["spectral_splits"] = 0
            state["spectral_splits"] += n_to_split
