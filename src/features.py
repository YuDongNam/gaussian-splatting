"""Feature extraction module for Gaussian Splatting statistical analysis.

This module provides functions to extract statistical features from
Gaussian Splatting models, including geometric properties and rendering metrics.
"""

from typing import Dict, Optional, Tuple

import torch

from src.gs_adapter import Gaussians, accumulate_alpha


def eigen_stats(
    scaling: torch.Tensor,
    rotation: Optional[torch.Tensor] = None,
    epsilon: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Compute eigenvalue-based statistics from scaling values.
    
    Instead of constructing full covariance matrices, this function directly
    uses the scaling values to compute volume and anisotropy metrics.
    
    Args:
        scaling: Scale values after exp activation (N, 3) torch.Tensor
        rotation: Rotation quaternions (N, 4) torch.Tensor. Not used in current
                  implementation but kept for API consistency.
        epsilon: Small value to prevent numerical instability (default: 1e-8)
        
    Returns:
        Dictionary containing:
            - volume: Volume of each Gaussian (N,) torch.Tensor
                     Computed as prod(scaling) = scaling[:, 0] * scaling[:, 1] * scaling[:, 2]
            - anisotropy: Anisotropy ratio (N,) torch.Tensor
                         Computed as max(scaling) / min(scaling) with epsilon protection
            - log_volume: Logarithm of volume for numerical stability (N,) torch.Tensor
            - max_scale: Maximum scale value per Gaussian (N,) torch.Tensor
            - min_scale: Minimum scale value per Gaussian (N,) torch.Tensor
            
    Raises:
        ValueError: If scaling tensor has invalid shape
    """
    if scaling.dim() != 2 or scaling.shape[1] != 3:
        raise ValueError(
            f"scaling must be (N, 3) tensor, got {scaling.shape}"
        )
    
    # Ensure scaling values are positive (add epsilon to prevent zeros)
    scaling_safe = scaling.clamp(min=epsilon)
    
    # Compute volume: product of all three scale dimensions
    volume = scaling_safe[:, 0] * scaling_safe[:, 1] * scaling_safe[:, 2]
    
    # Compute log volume for numerical stability
    log_volume = (
        torch.log(scaling_safe[:, 0] + epsilon) +
        torch.log(scaling_safe[:, 1] + epsilon) +
        torch.log(scaling_safe[:, 2] + epsilon)
    )
    
    # Compute max and min scales
    max_scale = torch.max(scaling_safe, dim=1)[0]
    min_scale = torch.min(scaling_safe, dim=1)[0]
    
    # Compute anisotropy: max / min ratio
    # Add epsilon to denominator to prevent division by zero
    anisotropy = max_scale / (min_scale + epsilon)
    
    # Clamp anisotropy to reasonable range to avoid extreme values
    anisotropy = torch.clamp(anisotropy, min=1.0, max=1e6)
    
    return {
        "volume": volume,
        "anisotropy": anisotropy,
        "log_volume": log_volume,
        "max_scale": max_scale,
        "min_scale": min_scale,
    }


def estimate_coverage_overlap(
    gs: Gaussians,
    viewmat: torch.Tensor,
    projmat: torch.Tensor,
    image_height: int = 800,
    image_width: int = 800,
    alpha_threshold: float = 0.01,
    fovx: Optional[float] = None,
    fovy: Optional[float] = None,
    campos: Optional[torch.Tensor] = None,
    bg_color: Optional[torch.Tensor] = None,
    sh_degree: int = 3,
    scale_modifier: float = 1.0,
) -> Dict[str, float]:
    """Estimate coverage and overlap metrics from rendered alpha map.
    
    Uses accumulate_alpha to render the scene and computes:
    - Coverage: Ratio of pixels with alpha > threshold
    - Overlap: Mean alpha value or total energy normalized by screen area
    
    Args:
        gs: Gaussians dataclass with model parameters
        viewmat: View matrix (4, 4) torch.Tensor on GPU
        projmat: Projection matrix (4, 4) torch.Tensor on GPU
        image_height: Rendering height in pixels (default: 800)
        image_width: Rendering width in pixels (default: 800)
        alpha_threshold: Threshold for coverage calculation (default: 0.01)
        fovx: Field of view in x direction (radians). If None, computed from projmat
        fovy: Field of view in y direction (radians). If None, computed from projmat
        campos: Camera position (3,) torch.Tensor. If None, extracted from viewmat
        bg_color: Background color tensor (3,) or (3, 1, 1). Defaults to black
        sh_degree: Spherical harmonics degree (default: 3)
        scale_modifier: Scale modifier for rendering (default: 1.0)
        
    Returns:
        Dictionary containing:
            - coverage: Ratio of pixels with alpha > threshold (0.0 to 1.0)
            - mean_alpha: Mean alpha value across all pixels
            - total_energy: Sum of all alpha values
            - energy_density: Total energy normalized by screen area (total_energy / (H * W))
            - max_alpha: Maximum alpha value in the map
            - min_alpha: Minimum alpha value in the map
            
    Raises:
        ValueError: If alpha_threshold is not in valid range [0, 1]
        RuntimeError: If rendering fails
    """
    if not 0.0 <= alpha_threshold <= 1.0:
        raise ValueError(
            f"alpha_threshold must be in [0, 1], got {alpha_threshold}"
        )
    
    # Get alpha map from rendering
    try:
        alpha_map = accumulate_alpha(
            gs=gs,
            viewmat=viewmat,
            projmat=projmat,
            image_height=image_height,
            image_width=image_width,
            fovx=fovx,
            fovy=fovy,
            campos=campos,
            bg_color=bg_color,
            sh_degree=sh_degree,
            scale_modifier=scale_modifier,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to compute alpha map: {e}") from e
    
    # Ensure alpha_map is on CPU for computation
    if alpha_map.is_cuda:
        alpha_map_cpu = alpha_map.cpu()
    else:
        alpha_map_cpu = alpha_map
    
    # Compute coverage: ratio of pixels with alpha > threshold
    coverage_mask = alpha_map_cpu > alpha_threshold
    num_covered_pixels = coverage_mask.sum().item()
    total_pixels = alpha_map_cpu.numel()
    coverage = num_covered_pixels / total_pixels if total_pixels > 0 else 0.0
    
    # Compute mean alpha value (overlap approximation)
    mean_alpha = alpha_map_cpu.mean().item()
    
    # Compute total energy (sum of all alpha values)
    total_energy = alpha_map_cpu.sum().item()
    
    # Compute energy density: total energy normalized by screen area
    screen_area = float(image_height * image_width)
    energy_density = total_energy / screen_area if screen_area > 0 else 0.0
    
    # Compute min/max alpha for additional statistics
    max_alpha = alpha_map_cpu.max().item()
    min_alpha = alpha_map_cpu.min().item()
    
    return {
        "coverage": coverage,
        "mean_alpha": mean_alpha,
        "total_energy": total_energy,
        "energy_density": energy_density,
        "max_alpha": max_alpha,
        "min_alpha": min_alpha,
    }

