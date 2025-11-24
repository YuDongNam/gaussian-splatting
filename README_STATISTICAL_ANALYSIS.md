# Gaussian Splatting Statistical Analysis

This project provides tools for statistical analysis of 3D Gaussian Splatting models. It extracts geometric features and rendering metrics from trained Gaussian Splatting checkpoints to enable quantitative analysis across multiple scenes and viewpoints.

## Overview

The statistical analysis pipeline includes:

- **Geometric Feature Extraction**: Volume, anisotropy, and scale statistics from Gaussian parameters
- **Rendering Metrics**: Coverage and overlap estimation from alpha accumulation maps
- **Batch Processing**: Automated processing of multiple scenes and camera viewpoints
- **Results Export**: CSV output for further analysis

## Project Structure

```
.
├── src/
│   ├── config.py          # Configuration dataclass
│   ├── io_utils.py        # File I/O utilities
│   ├── gs_adapter.py      # Gaussian Splatting model loader and renderer
│   ├── features.py        # Feature extraction functions
│   ├── run_experiments.py # Experiment runner
│   └── main.py            # CLI entry point
├── data/                  # Input data directory
│   ├── ckpts/             # Checkpoint .ply files
│   └── tables/            # Additional data tables
├── outputs/               # Output directory for results
└── requirements.txt       # Python dependencies
```

## Installation

### 1. Install Core Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Submodules

The project requires CUDA-enabled extensions from submodules:

```bash
# Install diff-gaussian-rasterization
pip install submodules/diff-gaussian-rasterization

# Install simple-knn (if needed)
pip install submodules/simple-knn

# Install fused-ssim (if needed)
pip install submodules/fused-ssim
```

**Note**: These submodules require CUDA and appropriate GPU drivers. Make sure you have:
- CUDA toolkit (11.6 or compatible)
- PyTorch with CUDA support
- Compatible GPU

### 3. Verify Installation

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "from diff_gaussian_rasterization import GaussianRasterizer; print('OK')"
```

## Data Preparation

### 1. Prepare Checkpoint Files

Place your trained Gaussian Splatting checkpoint files (`.ply` format) in the `data/ckpts/` directory or specify custom paths.

### 2. Prepare Camera Information

For each scene, you need a `cameras.json` file containing camera parameters. The JSON format should be:

```json
[
  {
    "id": 0,
    "img_name": "0000.png",
    "width": 800,
    "height": 800,
    "position": [0.0, 0.0, 0.0],
    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "fx": 400.0,
    "fy": 400.0
  },
  ...
]
```

**Note**: Camera JSON files are typically generated during training. If you have a trained model, check the model output directory for `cameras.json`.

### 3. Create Scene Configuration

Create a JSON file (e.g., `data/scenes.json`) listing all scenes to process:

```json
[
  {
    "scene_id": "scene1",
    "ckpt_path": "data/ckpts/scene1/point_cloud/iteration_30000/point_cloud.ply",
    "cameras_path": "data/ckpts/scene1/cameras.json"
  },
  {
    "scene_id": "scene2",
    "ckpt_path": "data/ckpts/scene2/point_cloud/iteration_30000/point_cloud.ply",
    "cameras_path": "data/ckpts/scene2/cameras.json"
  }
]
```

## Usage

### Basic Usage

Run experiments using a scene configuration file:

```bash
python src/main.py --scenes data/scenes.json --output outputs/results.csv
```

### Command Line Options

```bash
python src/main.py \
  --scenes data/scenes.json \
  --output outputs/results.csv \
  --alpha-threshold 0.01 \
  --render-height 800 \
  --render-width 800 \
  --random-seed 42
```

**Arguments:**
- `--scenes`: Path to JSON file with scene definitions (required)
- `--output`: Output CSV file path (default: `outputs/experiment_results.csv`)
- `--alpha-threshold`: Threshold for coverage calculation (default: 0.01)
- `--render-height`: Rendering height in pixels (default: 800)
- `--render-width`: Rendering width in pixels (default: 800)
- `--random-seed`: Random seed for reproducibility (default: 42)
- `--data-path`: Base data path (default: `data/`)
- `--output-path`: Base output path (default: `outputs/`)

### Alternative: Command Line Scene Definition

You can also specify scenes directly on the command line:

```bash
python src/main.py \
  --scenes "scene1:data/ckpt1.ply:data/cameras1.json,scene2:data/ckpt2.ply:data/cameras2.json"
```

## Output Format

The output CSV file contains the following columns:

- `scene_id`: Scene identifier
- `view_id`: Camera/view identifier
- `view_name`: Image name for the view
- `coverage`: Ratio of pixels with alpha > threshold (0.0 to 1.0)
- `mean_alpha`: Mean alpha value across all pixels
- `energy_density`: Total energy normalized by screen area
- `total_energy`: Sum of all alpha values
- `max_alpha`: Maximum alpha value in the map
- `min_alpha`: Minimum alpha value in the map

## Example Workflow

1. **Train or obtain Gaussian Splatting models** for your scenes
2. **Locate checkpoint files** (`.ply` files) and camera JSON files
3. **Create scene configuration** JSON file
4. **Run analysis**:
   ```bash
   python src/main.py --scenes data/scenes.json --output outputs/results.csv
   ```
5. **Analyze results** using pandas or other data analysis tools:
   ```python
   import pandas as pd
   df = pd.read_csv('outputs/results.csv')
   print(df.groupby('scene_id')['coverage'].mean())
   ```

## Module Reference

### `src/gs_adapter.py`
- `load_gaussians(ckpt_path)`: Load Gaussian Splatting model from .ply file
- `accumulate_alpha(gs, viewmat, projmat, ...)`: Render and extract alpha map

### `src/features.py`
- `eigen_stats(scaling, rotation)`: Compute volume and anisotropy statistics
- `estimate_coverage_overlap(gs, viewmat, projmat, ...)`: Compute coverage and overlap metrics

### `src/run_experiments.py`
- `process_ckpt(scene_id, ckpt_path, cameras)`: Process single checkpoint
- `run_all(scenes)`: Process all scenes and save results

## Troubleshooting

### CUDA Errors
- Ensure CUDA toolkit is properly installed
- Verify GPU drivers are up to date
- Check PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

### Import Errors
- Make sure submodules are installed: `pip install submodules/diff-gaussian-rasterization`
- Check that all dependencies are installed: `pip install -r requirements.txt`

### File Not Found Errors
- Verify checkpoint paths in scene configuration
- Ensure camera JSON files exist and are readable
- Check file permissions

## License

This statistical analysis tool is built on top of the official 3D Gaussian Splatting implementation. Please refer to the original project's LICENSE.md for licensing information.

## Citation

If you use this statistical analysis tool, please cite the original 3D Gaussian Splatting paper:

```bibtex
@Article{kerbl3Dgaussians,
  author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal      = {ACM Transactions on Graphics},
  number       = {4},
  volume       = {42},
  month        = {July},
  year         = {2023},
  url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

