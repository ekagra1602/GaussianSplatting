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

        # Apply spectral entropy-based splitting BEFORE standard densification
        # This ensures our indices are valid
        if (
            self.enable_spectral_splitting
            and step >= self.refine_start_iter
            and step % self.refine_every == 0
        ):
            self._spectral_split_gs(params, optimizers, state, step)

        # Then, run the standard DefaultStrategy densification
        super().step_post_backward(params, optimizers, state, step, info, packed)

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
        N = len(params["means"])

        if N == 0:
            return

        scales = torch.exp(params["scales"])  # [N, 3] actual scales
        quats = params["quats"]  # [N, 4]

        # Compute spectral entropy for all Gaussians
        spectral_entropy = compute_spectral_entropy(scales, quats)

        # Find needle-like Gaussians (low spectral entropy)
        is_needle = spectral_entropy < self.spectral_threshold

        # Split needles with reasonable visibility
        opacities = torch.sigmoid(params["opacities"])  # [N]
        is_visible = opacities > 0.005  # Reasonable threshold
        candidate_split = is_needle & is_visible

        # Limit splitting to 10% of candidates per iteration for stability
        # This prevents massive parameter changes that break optimizer state
        n_candidates = candidate_split.sum().item()
        max_splits_per_iter = max(1000, int(0.1 * n_candidates))  # At least 1000, or 10% of candidates

        if n_candidates > max_splits_per_iter:
            # Select top max_splits_per_iter needles by lowest entropy
            candidate_indices = torch.where(candidate_split)[0]
            candidate_entropies = spectral_entropy[candidate_indices]
            _, sorted_idx = torch.sort(candidate_entropies)
            top_indices = candidate_indices[sorted_idx[:max_splits_per_iter]]

            to_split = torch.zeros(N, dtype=torch.bool, device=scales.device)
            to_split[top_indices] = True
        else:
            to_split = candidate_split

        n_to_split = to_split.sum().item()

        if n_to_split > 0 and self.verbose:
            print(
                f"[SpectralStrategy] Step {step}: "
                f"Splitting {n_to_split} needle-like Gaussians "
                f"(entropy < {self.spectral_threshold})"
            )

        if n_to_split > 0:
            # Use the proper spectral splitting function from spectral_entropy.py
            device = params["means"].device

            # Get indices of needles to split
            split_indices = torch.where(to_split)[0]
            n_needles = len(split_indices)

            # Collect split results
            new_means_list = []
            new_scales_list = []
            new_quats_list = []
            new_opacities_list = []
            new_sh0_list = []
            new_shN_list = []

            # Split each needle using proper spectral splitting
            for idx in split_indices:
                mean = params["means"][idx]  # [3]
                scale = torch.exp(params["scales"][idx])  # [3] actual scales
                quat = params["quats"][idx]  # [4]
                opacity = torch.sigmoid(params["opacities"][idx])  # scalar

                # Call the correct splitting function!
                split_means, split_scales, split_quats, split_opacities = split_gaussian_spectral(
                    mean, scale, quat, opacity,
                    scale_factor=self.spectral_split_factor,
                    num_splits=2
                )

                # Store in log/logit space for parameters
                new_means_list.append(split_means)  # [2, 3]
                new_scales_list.append(torch.log(split_scales))  # [2, 3]
                new_quats_list.append(split_quats)  # [2, 4]
                new_opacities_list.append(torch.logit(split_opacities.clamp(1e-6, 1-1e-6)))  # [2]

                # Duplicate SH coefficients for both children
                new_sh0_list.append(params["sh0"][idx].unsqueeze(0).repeat(2, 1, 1))  # [2, 1, 3]
                if params["shN"].numel() > 0:
                    new_shN_list.append(params["shN"][idx].unsqueeze(0).repeat(2, 1, 1))

            # Concatenate all new Gaussians
            new_means = torch.cat(new_means_list, dim=0)  # [2*n_needles, 3]
            new_scales = torch.cat(new_scales_list, dim=0)  # [2*n_needles, 3]
            new_quats = torch.cat(new_quats_list, dim=0)  # [2*n_needles, 4]
            new_opacities = torch.cat(new_opacities_list, dim=0)  # [2*n_needles]
            new_sh0 = torch.cat(new_sh0_list, dim=0)  # [2*n_needles, 1, 3]

            # Update parameters
            params["means"] = torch.nn.Parameter(
                torch.cat([params["means"], new_means], dim=0)
            )
            params["scales"] = torch.nn.Parameter(
                torch.cat([params["scales"], new_scales], dim=0)
            )
            params["quats"] = torch.nn.Parameter(
                torch.cat([params["quats"], new_quats], dim=0)
            )
            params["opacities"] = torch.nn.Parameter(
                torch.cat([params["opacities"], new_opacities], dim=0)
            )
            params["sh0"] = torch.nn.Parameter(
                torch.cat([params["sh0"], new_sh0], dim=0)
            )

            # Handle shN (may be empty)
            if params["shN"].numel() > 0:
                new_shN = torch.cat(new_shN_list, dim=0)
                params["shN"] = torch.nn.Parameter(
                    torch.cat([params["shN"], new_shN], dim=0)
                )
            else:
                # Expand empty shN to match new count
                new_shape = list(params["shN"].shape)
                new_shape[0] = len(params["means"])
                params["shN"] = torch.nn.Parameter(
                    torch.zeros(new_shape, dtype=params["shN"].dtype, device=device)
                )

            # Reinitialize optimizers for all parameters
            for key, optimizer in optimizers.items():
                param_group = optimizer.param_groups[0]
                new_optimizer = type(optimizer)(
                    [params[key]],
                    lr=param_group['lr'],
                    eps=param_group.get('eps', 1e-8),
                    betas=param_group.get('betas', (0.9, 0.999)),
                )
                optimizers[key] = new_optimizer

            # Extend strategy state (2 new Gaussians per split)
            n_new_gaussians = 2 * n_needles

            if "grad2d" in state:
                state["grad2d"] = torch.cat([
                    state["grad2d"],
                    torch.zeros(n_new_gaussians, device=device)
                ], dim=0)

            if "count" in state:
                state["count"] = torch.cat([
                    state["count"],
                    torch.zeros(n_new_gaussians, dtype=state["count"].dtype, device=device)
                ], dim=0)

            if "radii" in state:
                if state["radii"].ndim == 1:
                    state["radii"] = torch.cat([
                        state["radii"],
                        torch.zeros(n_new_gaussians, dtype=state["radii"].dtype, device=device)
                    ], dim=0)
                else:
                    state["radii"] = torch.cat([
                        state["radii"],
                        torch.zeros(state["radii"].shape[0], n_new_gaussians, dtype=state["radii"].dtype, device=device)
                    ], dim=1)

            if self.verbose:
                print(f"[SpectralStrategy] Split {n_needles} needles into {n_new_gaussians} children, now have {len(params['means'])} Gaussians")
