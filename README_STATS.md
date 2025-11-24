# Gaussian Splatting Statistical Analysis

This project provides tools for statistical analysis of 3D Gaussian Splatting models.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
```

### 2. Prepare Data

- Place checkpoint `.ply` files in `data/ckpts/` or specify custom paths
- Ensure `cameras.json` files exist for each scene (typically generated during training)

### 3. Create Scene Configuration

Create `data/scenes.json`:

```json
[
  {
    "scene_id": "scene1",
    "ckpt_path": "path/to/point_cloud.ply",
    "cameras_path": "path/to/cameras.json"
  }
]
```

### 4. Run Analysis

```bash
python src/main.py --scenes data/scenes.json --output outputs/results.csv
```

## Output

Results are saved as CSV with columns:
- `scene_id`, `view_id`, `view_name`
- `coverage`: Ratio of pixels with alpha > threshold
- `mean_alpha`: Mean alpha value
- `energy_density`: Total energy / screen area
- `max_alpha`, `min_alpha`: Alpha value range

## Requirements

- Python 3.7+
- CUDA-enabled GPU
- PyTorch with CUDA support
- See `requirements.txt` for full list

