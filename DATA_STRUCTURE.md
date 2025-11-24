# Data Structure for Training

This document describes the required data structure for running `train.py` in the Gaussian Splatting project.

## Overview

The `train.py` script supports two types of datasets:
1. **COLMAP format** (most common)
2. **NeRF Synthetic (Blender) format**

## COLMAP Format (Recommended)

### Directory Structure

```
data/
└── <scene_name>/
    ├── images/              # Input images (required)
    │   ├── IMG_001.jpg
    │   ├── IMG_002.jpg
    │   └── ...
    └── sparse/
        └── 0/
            ├── cameras.bin      # Camera intrinsics (binary format)
            ├── images.bin        # Camera extrinsics (binary format)
            ├── points3D.bin     # 3D point cloud (binary format)
            └── project.ini      # COLMAP project file (optional)
            
            # Alternative: text format (if .bin files don't exist)
            ├── cameras.txt      # Camera intrinsics (text format)
            ├── images.txt      # Camera extrinsics (text format)
            └── points3D.txt     # 3D point cloud (text format)
```

### Required Files

1. **`images/` folder**: Contains all input images (JPG, PNG, etc.)
2. **`sparse/0/cameras.bin`** (or `cameras.txt`): Camera intrinsic parameters
3. **`sparse/0/images.bin`** (or `images.txt`): Camera poses and image metadata
4. **`sparse/0/points3D.bin`** (or `points3D.txt`): Sparse 3D point cloud from COLMAP reconstruction

### Optional Files

- **`sparse/0/depth_params.json`**: Depth regularization parameters (if using depth supervision)
- **`depths/` folder**: Depth maps for each image (if using depth supervision)
- **`sparse/0/test.txt`**: List of test image names (if using custom train/test split)

### Example Command

```bash
python train.py -s data/scenes/train --model_path outputs/train_model
```

## NeRF Synthetic (Blender) Format

### Directory Structure

```
data/
└── <scene_name>/
    ├── transforms_train.json   # Camera poses and metadata (required)
    ├── transforms_test.json    # Test camera poses (optional)
    └── <images>/               # Image files
        ├── 0000.png
        ├── 0001.png
        └── ...
```

### Required Files

1. **`transforms_train.json`**: Contains camera parameters in NeRF format
   ```json
   {
     "camera_angle_x": 0.8575560450553894,
     "frames": [
       {
         "file_path": "./train/r_0",
         "transform_matrix": [[...], [...], [...], [...]]
       },
       ...
     ]
   }
   ```

2. **Image files**: Referenced in `transforms_train.json`

### Example Command

```bash
python train.py -s data/scenes/nerf_scene --model_path outputs/nerf_model --white_background
```

## Data Preparation Steps

### For COLMAP Format:

1. **Run COLMAP reconstruction** on your images:
   ```bash
   colmap feature_extractor --database_path database.db --image_path images/
   colmap exhaustive_matcher --database_path database.db
   colmap mapper --database_path database.db --image_path images/ --output_path sparse/
   ```

2. **Organize the output**:
   - Move `sparse/0/` folder to `data/<scene_name>/sparse/0/`
   - Move images to `data/<scene_name>/images/`

3. **Verify structure**:
   ```bash
   ls data/<scene_name>/sparse/0/  # Should contain cameras.bin, images.bin, points3D.bin
   ls data/<scene_name>/images/    # Should contain all image files
   ```

### For NeRF Synthetic Format:

1. **Export from Blender** or use existing NeRF datasets
2. **Place `transforms_train.json`** in scene directory
3. **Ensure image paths** in JSON match actual file locations

## Training Output Structure

After training, the model will be saved in the `--model_path` directory:

```
outputs/
└── <model_name>/
    ├── cameras.json              # Camera parameters (generated during training)
    ├── input.ply                 # Initial point cloud
    ├── cfg_args                  # Training configuration
    └── point_cloud/
        └── iteration_<N>/
            └── point_cloud.ply   # Trained model checkpoint
```

## Common Issues

### Missing Files
- **Error**: "Could not recognize scene type!"
  - **Solution**: Ensure `sparse/0/` folder exists (COLMAP) or `transforms_train.json` exists (Blender)

### Image Path Issues
- **Error**: "Image not found"
  - **Solution**: Check that image paths in COLMAP files or JSON match actual file locations

### Point Cloud Issues
- **Error**: "points3D.bin not found"
  - **Solution**: Ensure COLMAP reconstruction completed successfully and generated points3D.bin

## Notes

- The `--images` flag can specify a different subdirectory name (default: `images`)
- The `--eval` flag uses a MipNeRF360-style train/test split
- Depth supervision requires `--depths` flag pointing to depth map folder
- For high-resolution datasets, use `--data_device cpu` to reduce VRAM usage

