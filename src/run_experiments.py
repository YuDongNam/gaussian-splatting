"""Experiment runner for Gaussian Splatting statistical analysis.

This module provides functions to process checkpoints and run experiments
across multiple scenes, collecting coverage and overlap metrics.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from src.config import CFG
from src.features import estimate_coverage_overlap
from src.gs_adapter import Gaussians, load_gaussians
from src.io_utils import ensure_dir, save_csv


def load_cameras_from_json(cameras_path: Path) -> List[Dict]:
    """Load camera information from JSON file.
    
    Args:
        cameras_path: Path to cameras.json file
        
    Returns:
        List of camera dictionaries with keys: id, img_name, width, height,
        position, rotation, fx, fy
        
    Raises:
        FileNotFoundError: If cameras file doesn't exist
        ValueError: If JSON format is invalid
    """
    cameras_path = Path(cameras_path)
    
    if not cameras_path.exists():
        raise FileNotFoundError(f"Cameras file not found: {cameras_path}")
    
    try:
        with open(cameras_path, 'r', encoding='utf-8') as f:
            cameras = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in cameras file: {e}") from e
    
    if not isinstance(cameras, list):
        raise ValueError("Cameras file must contain a list of camera dictionaries")
    
    return cameras


def build_view_matrix(position: List[float], rotation: List[List[float]]) -> torch.Tensor:
    """Build view matrix from camera position and rotation.
    
    Args:
        position: Camera position [x, y, z]
        rotation: Camera rotation matrix (3x3)
        
    Returns:
        View matrix (4, 4) torch.Tensor on GPU
    """
    # Convert to numpy arrays
    pos = np.array(position, dtype=np.float32)
    rot = np.array(rotation, dtype=np.float32)
    
    # Build world-to-camera transform
    # Rotation is already camera-to-world, so we need to invert it
    w2c = np.zeros((4, 4), dtype=np.float32)
    w2c[:3, :3] = rot.T  # Transpose rotation to get world-to-camera
    w2c[:3, 3] = -rot.T @ pos  # Translation in camera space
    w2c[3, 3] = 1.0
    
    # Convert to torch tensor and transpose (OpenGL convention)
    viewmat = torch.tensor(w2c, dtype=torch.float32, device="cuda").transpose(0, 1)
    
    return viewmat


def build_projection_matrix(
    fx: float,
    fy: float,
    width: int,
    height: int,
    znear: float = 0.01,
    zfar: float = 100.0,
) -> torch.Tensor:
    """Build projection matrix from camera intrinsics.
    
    Args:
        fx: Focal length in x direction
        fy: Focal length in y direction
        width: Image width
        height: Image height
        znear: Near clipping plane (default: 0.01)
        zfar: Far clipping plane (default: 100.0)
        
    Returns:
        Projection matrix (4, 4) torch.Tensor on GPU
    """
    # Convert focal lengths to FOV
    fovx = 2.0 * math.atan(width / (2.0 * fx))
    fovy = 2.0 * math.atan(height / (2.0 * fy))
    
    # Build projection matrix (OpenGL perspective projection)
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)
    
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = 1.0 / tanfovx
    proj[1, 1] = 1.0 / tanfovy
    proj[2, 2] = -(zfar + znear) / (zfar - znear)
    proj[2, 3] = -1.0
    proj[3, 2] = -(2.0 * zfar * znear) / (zfar - znear)
    
    # Convert to torch tensor and transpose (OpenGL convention)
    projmat = torch.tensor(proj, dtype=torch.float32, device="cuda").transpose(0, 1)
    
    return projmat


def process_ckpt(
    scene_id: str,
    ckpt_path: Path,
    cameras: List[Dict],
    cfg: Optional[CFG] = None,
    alpha_threshold: float = 0.01,
) -> List[Dict]:
    """Process a single checkpoint across all camera views.
    
    Args:
        scene_id: Identifier for the scene
        ckpt_path: Path to .ply checkpoint file
        cameras: List of camera dictionaries from JSON
        cfg: Configuration object. If None, uses defaults
        alpha_threshold: Threshold for coverage calculation (default: 0.01)
        
    Returns:
        List of dictionaries containing results for each view:
        - scene_id: Scene identifier
        - view_id: View/camera identifier
        - view_name: Image name for the view
        - coverage: Coverage metric
        - mean_alpha: Mean alpha value
        - energy_density: Energy density metric
        - total_energy: Total energy
        - max_alpha: Maximum alpha value
        - min_alpha: Minimum alpha value
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If processing fails
    """
    if cfg is None:
        cfg = CFG()
    
    # Load Gaussian Splatting model
    try:
        gs = load_gaussians(ckpt_path, max_sh_degree=3)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {ckpt_path}: {e}") from e
    
    results = []
    
    # Process each camera view
    for cam_info in cameras:
        try:
            # Extract camera parameters
            view_id = cam_info.get('id', -1)
            view_name = cam_info.get('img_name', f'view_{view_id}')
            width = cam_info.get('width', cfg.render_width)
            height = cam_info.get('height', cfg.render_height)
            position = cam_info.get('position', [0.0, 0.0, 0.0])
            rotation = cam_info.get('rotation', [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            fx = cam_info.get('fx', width / 2.0)
            fy = cam_info.get('fy', height / 2.0)
            
            # Build view and projection matrices
            viewmat = build_view_matrix(position, rotation)
            projmat = build_projection_matrix(fx, fy, width, height)
            
            # Compute FOV from intrinsics
            fovx = 2.0 * math.atan(width / (2.0 * fx))
            fovy = 2.0 * math.atan(height / (2.0 * fy))
            
            # Estimate coverage and overlap
            metrics = estimate_coverage_overlap(
                gs=gs,
                viewmat=viewmat,
                projmat=projmat,
                image_height=height,
                image_width=width,
                alpha_threshold=alpha_threshold,
                fovx=fovx,
                fovy=fovy,
                sh_degree=3,
                scale_modifier=1.0,
            )
            
            # Store results
            result = {
                'scene_id': scene_id,
                'view_id': view_id,
                'view_name': view_name,
                'coverage': metrics['coverage'],
                'mean_alpha': metrics['mean_alpha'],
                'energy_density': metrics['energy_density'],
                'total_energy': metrics['total_energy'],
                'max_alpha': metrics['max_alpha'],
                'min_alpha': metrics['min_alpha'],
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Warning: Failed to process view {view_id} ({view_name}): {e}")
            # Continue with next view
            continue
    
    return results


def run_all(
    scenes: List[Dict[str, str]],
    output_path: Optional[Path] = None,
    cfg: Optional[CFG] = None,
    alpha_threshold: float = 0.01,
) -> pd.DataFrame:
    """Run experiments for all scenes and save results to CSV.
    
    Args:
        scenes: List of scene dictionaries with keys:
            - scene_id: Scene identifier
            - ckpt_path: Path to checkpoint .ply file
            - cameras_path: Path to cameras.json file
        output_path: Path to output CSV file. If None, uses cfg.output_path
        cfg: Configuration object. If None, uses defaults
        alpha_threshold: Threshold for coverage calculation (default: 0.01)
        
    Returns:
        pandas DataFrame with all results
        
    Raises:
        ValueError: If scenes list is empty or invalid
    """
    if not scenes:
        raise ValueError("scenes list cannot be empty")
    
    if cfg is None:
        cfg = CFG()
    
    if output_path is None:
        output_path = cfg.output_path / "experiment_results.csv"
    
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    
    all_results = []
    
    # Process each scene
    for scene_info in scenes:
        scene_id = scene_info.get('scene_id')
        ckpt_path = Path(scene_info.get('ckpt_path'))
        cameras_path = Path(scene_info.get('cameras_path'))
        
        if not scene_id:
            print(f"Warning: Skipping scene with missing scene_id")
            continue
        
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint not found for scene {scene_id}: {ckpt_path}")
            continue
        
        if not cameras_path.exists():
            print(f"Warning: Cameras file not found for scene {scene_id}: {cameras_path}")
            continue
        
        print(f"Processing scene: {scene_id}")
        
        try:
            # Load cameras
            cameras = load_cameras_from_json(cameras_path)
            print(f"  Loaded {len(cameras)} cameras")
            
            # Process checkpoint
            results = process_ckpt(
                scene_id=scene_id,
                ckpt_path=ckpt_path,
                cameras=cameras,
                cfg=cfg,
                alpha_threshold=alpha_threshold,
            )
            
            all_results.extend(results)
            print(f"  Processed {len(results)} views")
            
        except Exception as e:
            print(f"Error processing scene {scene_id}: {e}")
            continue
    
    # Create DataFrame
    if not all_results:
        print("Warning: No results to save")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    save_csv(
        data=all_results,
        filepath=output_path,
        fieldnames=list(df.columns),
    )
    
    print(f"\nResults saved to: {output_path}")
    print(f"Total rows: {len(df)}")
    
    return df

