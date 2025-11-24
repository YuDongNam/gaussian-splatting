"""I/O utility functions for Gaussian Splatting analysis."""

import csv
from pathlib import Path
from typing import List, Dict, Any, Optional


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Path to directory
        
    Returns:
        Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(data: List[Dict[str, Any]], filepath: Path, fieldnames: Optional[List[str]] = None) -> None:
    """Save data to CSV file.
    
    Args:
        data: List of dictionaries to save
        filepath: Path to output CSV file
        fieldnames: Optional list of field names. If None, uses keys from first dict
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    if not data:
        raise ValueError("Data list is empty")
    
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def load_splits(filepath: Path) -> Dict[str, List[str]]:
    """Load train/test/val splits from file.
    
    Expected format: JSON-like structure or CSV with split information.
    For now, assumes a simple text file with one item per line,
    or a more complex format can be added later.
    
    Args:
        filepath: Path to splits file
        
    Returns:
        Dictionary with keys 'train', 'test', 'val' and lists of items
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Splits file not found: {filepath}")
    
    # Simple implementation: assumes JSON format
    # Can be extended to support other formats
    import json
    
    with open(filepath, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    return splits

