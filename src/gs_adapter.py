"""Gaussian Splatting adapter for statistical analysis.

This module provides interfaces to load Gaussian Splatting models and
extract rendering information using the official diff-gaussian-rasterization library.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from plyfile import PlyData

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
except ImportError as e:
    raise ImportError(
        "diff_gaussian_rasterization is not installed. "
        "Please install it from submodules/diff-gaussian-rasterization"
    ) from e


@dataclass
class Gaussians:
    """Gaussian Splatting model data structure.
    
    Attributes:
        xyz: 3D positions of Gaussians (N, 3) torch.Tensor
        opacity: Opacity values after sigmoid activation (N, 1) torch.Tensor
        features: Spherical harmonics features (N, C, SH_coeffs) torch.Tensor
        scaling: Scale values after exp activation (N, 3) torch.Tensor
        rotation: Rotation quaternions after normalization (N, 4) torch.Tensor
    """
    xyz: torch.Tensor
    opacity: torch.Tensor
    features: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor


def load_gaussians(ckpt_path: Path, max_sh_degree: int = 3) -> Gaussians:
    """Load Gaussian Splatting model from .ply checkpoint file.
    
    Args:
        ckpt_path: Path to .ply checkpoint file
        max_sh_degree: Maximum spherical harmonics degree (default: 3)
        
    Returns:
        Gaussians dataclass with all tensors on GPU (cuda)
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If ply file format is invalid
    """
    ckpt_path = Path(ckpt_path)
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    
    if not ckpt_path.suffix == ".ply":
        raise ValueError(f"Expected .ply file, got {ckpt_path.suffix}")
    
    try:
        plydata = PlyData.read(str(ckpt_path))
    except Exception as e:
        raise ValueError(f"Failed to read ply file: {e}") from e
    
    if len(plydata.elements) == 0:
        raise ValueError("Ply file has no elements")
    
    vertex_data = plydata.elements[0]
    
    # Extract xyz coordinates
    try:
        xyz = np.stack((
            np.asarray(vertex_data["x"]),
            np.asarray(vertex_data["y"]),
            np.asarray(vertex_data["z"]),
        ), axis=1)
    except KeyError as e:
        raise ValueError(f"Missing coordinate field in ply file: {e}") from e
    
    # Extract opacity (will be sigmoid applied later)
    try:
        opacity_raw = np.asarray(vertex_data["opacity"])[..., np.newaxis]
    except KeyError as e:
        raise ValueError(f"Missing opacity field in ply file: {e}") from e
    
    # Extract features (DC component)
    try:
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(vertex_data["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(vertex_data["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(vertex_data["f_dc_2"])
    except KeyError as e:
        raise ValueError(f"Missing DC feature fields in ply file: {e}") from e
    
    # Extract rest features (SH coefficients)
    extra_f_names = [
        p.name for p in vertex_data.properties 
        if p.name.startswith("f_rest_")
    ]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    
    expected_rest_features = 3 * ((max_sh_degree + 1) ** 2 - 1)
    if len(extra_f_names) != expected_rest_features:
        raise ValueError(
            f"Expected {expected_rest_features} rest features, "
            f"got {len(extra_f_names)}. Check max_sh_degree."
        )
    
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(vertex_data[attr_name])
    
    # Reshape to (N, 3, SH_coeffs - 1)
    features_extra = features_extra.reshape((
        features_extra.shape[0], 
        3, 
        (max_sh_degree + 1) ** 2 - 1
    ))
    
    # Extract scaling (will be exp applied later)
    scale_names = [
        p.name for p in vertex_data.properties 
        if p.name.startswith("scale_")
    ]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    
    if len(scale_names) != 3:
        raise ValueError(f"Expected 3 scale fields, got {len(scale_names)}")
    
    scales_raw = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales_raw[:, idx] = np.asarray(vertex_data[attr_name])
    
    # Extract rotation (will be normalized later)
    rot_names = [
        p.name for p in vertex_data.properties 
        if p.name.startswith("rot")
    ]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    
    if len(rot_names) != 4:
        raise ValueError(f"Expected 4 rotation fields, got {len(rot_names)}")
    
    rots_raw = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots_raw[:, idx] = np.asarray(vertex_data[attr_name])
    
    # Convert to torch tensors and move to GPU
    xyz_tensor = torch.tensor(xyz, dtype=torch.float32, device="cuda")
    opacity_raw_tensor = torch.tensor(opacity_raw, dtype=torch.float32, device="cuda")
    features_dc_tensor = torch.tensor(
        features_dc, dtype=torch.float32, device="cuda"
    ).transpose(1, 2).contiguous()
    features_extra_tensor = torch.tensor(
        features_extra, dtype=torch.float32, device="cuda"
    ).transpose(1, 2).contiguous()
    scales_raw_tensor = torch.tensor(
        scales_raw, dtype=torch.float32, device="cuda"
    )
    rots_raw_tensor = torch.tensor(
        rots_raw, dtype=torch.float32, device="cuda"
    )
    
    # Apply activations: Scale -> exp, Opacity -> sigmoid, Rotation -> normalize
    scaling = torch.exp(scales_raw_tensor)
    opacity = torch.sigmoid(opacity_raw_tensor)
    rotation = torch.nn.functional.normalize(rots_raw_tensor, dim=1)
    
    # Concatenate features: DC + rest
    # After transpose: features_dc_tensor is (N, 1, 3), features_extra_tensor is (N, SH_coeffs-1, 3)
    # Concatenate along dim=1 to get (N, SH_coeffs, 3)
    # This matches GaussianModel.get_features() which uses dim=1
    features = torch.cat((features_dc_tensor, features_extra_tensor), dim=1)
    
    return Gaussians(
        xyz=xyz_tensor,
        opacity=opacity,
        features=features,
        scaling=scaling,
        rotation=rotation,
    )


def accumulate_alpha(
    gs: Gaussians,
    viewmat: torch.Tensor,
    projmat: torch.Tensor,
    image_height: int = 800,
    image_width: int = 800,
    fovx: Optional[float] = None,
    fovy: Optional[float] = None,
    campos: Optional[torch.Tensor] = None,
    bg_color: Optional[torch.Tensor] = None,
    sh_degree: int = 3,
    scale_modifier: float = 1.0,
) -> torch.Tensor:
    """Accumulate alpha values from Gaussian Splatting rendering.
    
    Performs forward pass through GaussianRasterizer and extracts alpha/density
    information from the rendered output.
    
    Args:
        gs: Gaussians dataclass with model parameters
        viewmat: View matrix (4, 4) torch.Tensor on GPU
        projmat: Projection matrix (4, 4) torch.Tensor on GPU
        image_height: Rendering height in pixels (default: 800)
        image_width: Rendering width in pixels (default: 800)
        fovx: Field of view in x direction (radians). If None, computed from projmat
        fovy: Field of view in y direction (radians). If None, computed from projmat
        campos: Camera position (3,) torch.Tensor. If None, extracted from viewmat
        bg_color: Background color tensor (3,) or (3, 1, 1). Defaults to black
        sh_degree: Spherical harmonics degree (default: 3)
        scale_modifier: Scale modifier for rendering (default: 1.0)
        
    Returns:
        Alpha accumulation map (image_height, image_width) torch.Tensor
        representing the density/opacity at each pixel
        
    Raises:
        ValueError: If input tensors have invalid shapes
        RuntimeError: If rendering fails
    """
    # Validate input shapes
    if viewmat.shape != (4, 4):
        raise ValueError(f"viewmat must be (4, 4), got {viewmat.shape}")
    if projmat.shape != (4, 4):
        raise ValueError(f"projmat must be (4, 4), got {projmat.shape}")
    
    # Set default background color (black)
    if bg_color is None:
        bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
    
    # Ensure bg_color has correct shape
    if bg_color.dim() == 1:
        bg_color = bg_color.view(3, 1, 1)
    elif bg_color.dim() == 2:
        bg_color = bg_color.view(3, 1, 1)
    
    # Compute FOV if not provided
    if fovx is None or fovy is None:
        # Extract FOV from projection matrix
        # proj[0,0] = 1 / (tan(fovx/2) * aspect)
        # proj[1,1] = 1 / tan(fovy/2)
        if fovx is None:
            # Approximate: assume square pixels
            fovx = 2.0 * math.atan(1.0 / projmat[0, 0].item())
        if fovy is None:
            fovy = 2.0 * math.atan(1.0 / projmat[1, 1].item())
    
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)
    
    # Extract camera position from view matrix if not provided
    if campos is None:
        # Camera position is the translation component of view matrix inverse
        viewmat_inv = torch.inverse(viewmat)
        campos = viewmat_inv[:3, 3]
    
    # Create screenspace points (required by rasterizer)
    screenspace_points = torch.zeros_like(
        gs.xyz, 
        dtype=gs.xyz.dtype, 
        requires_grad=False, 
        device="cuda"
    )
    
    # Set up rasterization settings
    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scale_modifier,
        viewmatrix=viewmat,
        projmatrix=projmat,
        sh_degree=sh_degree,
        campos=campos,
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )
    
    # Initialize rasterizer
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Prepare inputs for rasterizer
    means3D = gs.xyz
    means2D = screenspace_points
    opacities = gs.opacity
    
    # Use scales and rotations (not precomputed covariance)
    scales = gs.scaling
    rotations = gs.rotation
    
    # Use SH features (not precomputed colors)
    # gs.features is (N, SH_coeffs, 3) after concat: (N, 16, 3) for SH degree 3
    # Rasterizer expects (N, SH_coeffs, 3) format when separate_sh=False
    # Use features directly (same as gaussian_renderer when separate_sh=False)
    shs = gs.features.contiguous()  # (N, SH_coeffs, 3)
    
    try:
        # Forward pass through rasterizer
        # Note: GaussianRasterizer.forward() does not accept 'dc' parameter
        # It only accepts 'shs' parameter with all SH coefficients concatenated
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
    except Exception as e:
        raise RuntimeError(f"Rendering failed: {e}") from e
    
    # Extract alpha/density from rendered image
    # The rendered image is RGB (3, H, W), we can compute alpha as:
    # 1. Use the opacity values directly accumulated
    # 2. Or compute density from the rendered image intensity
    # For statistical analysis, we'll compute a density map from the rendered image
    # Alpha accumulation can be approximated as: 1 - (background contribution)
    # Since we use black background, we can use the luminance as density proxy
    
    # Convert RGB to grayscale (luminance) as alpha proxy
    # Using standard luminance weights: 0.299*R + 0.587*G + 0.114*B
    alpha_map = (
        0.299 * rendered_image[0, :, :] +
        0.587 * rendered_image[1, :, :] +
        0.114 * rendered_image[2, :, :]
    )
    
    # Alternative: use max channel as density
    # alpha_map = torch.max(rendered_image, dim=0)[0]
    
    return alpha_map

