"""Configuration module for Gaussian Splatting statistical analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CFG:
    """Configuration class for Gaussian Splatting analysis.
    
    Attributes:
        render_height: Rendering height in pixels (default: 800)
        render_width: Rendering width in pixels (default: 800)
        data_path: Path to data directory containing checkpoints and tables
        ckpts_path: Path to checkpoints directory
        tables_path: Path to tables directory
        random_seed: Random seed for reproducibility (default: 42)
        output_path: Path to output directory for results
    """
    render_height: int = 800
    render_width: int = 800
    data_path: Optional[Path] = None
    ckpts_path: Optional[Path] = None
    tables_path: Optional[Path] = None
    random_seed: int = 42
    output_path: Optional[Path] = None
    
    def __post_init__(self):
        """Initialize paths after dataclass creation."""
        if self.data_path is None:
            self.data_path = Path("data")
        else:
            self.data_path = Path(self.data_path)
            
        if self.ckpts_path is None:
            self.ckpts_path = self.data_path / "ckpts"
        else:
            self.ckpts_path = Path(self.ckpts_path)
            
        if self.tables_path is None:
            self.tables_path = self.data_path / "tables"
        else:
            self.tables_path = Path(self.tables_path)
            
        if self.output_path is None:
            self.output_path = Path("outputs")
        else:
            self.output_path = Path(self.output_path)
        
        # Ensure directories exist (create if they don't)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.ckpts_path.mkdir(parents=True, exist_ok=True)
        self.tables_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)

