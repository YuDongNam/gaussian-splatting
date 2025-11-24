"""Main entry point for Gaussian Splatting statistical analysis experiments.

This module provides a CLI interface to run experiments on multiple scenes.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

from src.config import CFG
from src.io_utils import load_splits
from src.run_experiments import run_all


def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run Gaussian Splatting statistical analysis experiments"
    )
    
    parser.add_argument(
        '--scenes',
        type=str,
        required=True,
        help='Path to JSON file containing scene definitions or comma-separated scene configs'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: outputs/experiment_results.csv)'
    )
    
    parser.add_argument(
        '--alpha-threshold',
        type=float,
        default=0.01,
        help='Alpha threshold for coverage calculation (default: 0.01)'
    )
    
    parser.add_argument(
        '--render-height',
        type=int,
        default=800,
        help='Rendering height in pixels (default: 800)'
    )
    
    parser.add_argument(
        '--render-width',
        type=int,
        default=800,
        help='Rendering width in pixels (default: 800)'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Base data path (default: data/)'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Base output path (default: outputs/)'
    )
    
    return parser.parse_args()


def load_scenes_from_json(scenes_path: Path) -> List[Dict[str, str]]:
    """Load scene definitions from JSON file.
    
    Expected JSON format:
    [
        {
            "scene_id": "scene1",
            "ckpt_path": "path/to/point_cloud.ply",
            "cameras_path": "path/to/cameras.json"
        },
        ...
    ]
    
    Args:
        scenes_path: Path to JSON file with scene definitions
        
    Returns:
        List of scene dictionaries
        
    Raises:
        FileNotFoundError: If scenes file doesn't exist
        ValueError: If JSON format is invalid
    """
    scenes_path = Path(scenes_path)
    
    if not scenes_path.exists():
        raise FileNotFoundError(f"Scenes file not found: {scenes_path}")
    
    try:
        with open(scenes_path, 'r', encoding='utf-8') as f:
            scenes = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in scenes file: {e}") from e
    
    if not isinstance(scenes, list):
        raise ValueError("Scenes file must contain a list of scene dictionaries")
    
    # Validate scene structure
    required_keys = {'scene_id', 'ckpt_path', 'cameras_path'}
    for scene in scenes:
        if not isinstance(scene, dict):
            raise ValueError("Each scene must be a dictionary")
        missing_keys = required_keys - set(scene.keys())
        if missing_keys:
            raise ValueError(
                f"Scene {scene.get('scene_id', 'unknown')} missing keys: {missing_keys}"
            )
    
    return scenes


def main():
    """Main entry point for the experiment runner."""
    args = parse_args()
    
    # Create configuration
    cfg = CFG(
        render_height=args.render_height,
        render_width=args.render_width,
        random_seed=args.random_seed,
    )
    
    if args.data_path:
        cfg.data_path = Path(args.data_path)
    if args.output_path:
        cfg.output_path = Path(args.output_path)
    
    # Set random seed
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    
    # Load scenes
    scenes_path = Path(args.scenes)
    
    if scenes_path.suffix == '.json':
        # Load from JSON file
        scenes = load_scenes_from_json(scenes_path)
    else:
        # Try to parse as comma-separated scene configs
        # Format: "scene1:ckpt1:cameras1,scene2:ckpt2:cameras2"
        scenes = []
        for scene_config in args.scenes.split(','):
            parts = scene_config.split(':')
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid scene config format: {scene_config}. "
                    "Expected: scene_id:ckpt_path:cameras_path"
                )
            scenes.append({
                'scene_id': parts[0],
                'ckpt_path': parts[1],
                'cameras_path': parts[2],
            })
    
    if not scenes:
        raise ValueError("No scenes to process")
    
    print(f"Loaded {len(scenes)} scene(s)")
    print(f"Configuration:")
    print(f"  Render size: {cfg.render_width}x{cfg.render_height}")
    print(f"  Alpha threshold: {args.alpha_threshold}")
    print(f"  Random seed: {cfg.random_seed}")
    print()
    
    # Run experiments
    output_path = Path(args.output) if args.output else None
    
    try:
        df = run_all(
            scenes=scenes,
            output_path=output_path,
            cfg=cfg,
            alpha_threshold=args.alpha_threshold,
        )
        
        if not df.empty:
            print("\nSummary statistics:")
            print(df.groupby('scene_id').agg({
                'coverage': ['mean', 'std'],
                'mean_alpha': ['mean', 'std'],
                'energy_density': ['mean', 'std'],
            }))
        
        print("\nExperiment completed successfully!")
        
    except Exception as e:
        print(f"\nError running experiments: {e}")
        raise


if __name__ == "__main__":
    main()

