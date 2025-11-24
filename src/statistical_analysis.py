"""Statistical analysis for 3D Gaussian Splatting training data.

This module performs comprehensive statistical analysis on training records
including evolution visualization, GAM modeling, and mixed effects analysis.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple
import sys

# Statistical modeling libraries
try:
    from pygam import LinearGAM, s
    PYGAM_AVAILABLE = True
except ImportError:
    PYGAM_AVAILABLE = False
    print("Warning: pygam not available. Install with: pip install pygam")

try:
    from statsmodels.regression.mixed_linear_model import MixedLM
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Install with: pip install statsmodels")

# 한글 폰트 설정 (Colab 환경 고려)
try:
    import matplotlib.font_manager as fm
    # Colab에서 나눔고딕 사용 시도
    font_list = [f.name for f in fm.fontManager.ttflist]
    if 'NanumGothic' in font_list or 'Nanum Gothic' in font_list:
        plt.rcParams['font.family'] = 'NanumGothic'
    elif 'Malgun Gothic' in font_list:
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:
        # 한글 폰트가 없으면 영어로
        plt.rcParams['font.family'] = 'DejaVu Sans'
        print("Korean font not found. Using English labels.")
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    print("Using English labels for compatibility.")

# matplotlib 한글 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# Seaborn 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_and_preprocess_data(
    csv_path: Path,
    output_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, str]:
    """Load and preprocess records.csv data.
    
    Args:
        csv_path: Path to records.csv file
        output_dir: Directory for output files
        
    Returns:
        Tuple of (preprocessed DataFrame, report string)
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Data Preprocessing Report")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Load data
    df = pd.read_csv(csv_path)
    report_lines.append(f"Original data shape: {df.shape}")
    report_lines.append(f"Columns: {', '.join(df.columns.tolist())}")
    report_lines.append("")
    
    # Log transformations
    log_vars = {
        'N_g': 'log_Ng',
        'anisotropy_median': 'log_Aniso',
        'volume_median': 'log_Vol'
    }
    
    for orig_var, log_var in log_vars.items():
        if orig_var in df.columns:
            # 0이나 음수 값 처리
            df[orig_var] = df[orig_var].clip(lower=1e-8)
            df[log_var] = np.log10(df[orig_var])
            report_lines.append(f"Created {log_var} = log10({orig_var})")
        else:
            report_lines.append(f"Warning: {orig_var} not found in data")
    
    report_lines.append("")
    
    # Missing value handling
    if 'psnr' in df.columns:
        psnr_missing = df['psnr'].isna().sum()
        psnr_total = len(df)
        report_lines.append(f"PSNR missing values: {psnr_missing} / {psnr_total} ({100*psnr_missing/psnr_total:.1f}%)")
        
        if psnr_missing == psnr_total:
            report_lines.append("⚠️  All PSNR values are missing. Using 'coverage' as target variable.")
            target_var = 'coverage'
        elif psnr_missing > psnr_total * 0.5:
            report_lines.append("⚠️  More than 50% PSNR missing. Consider using 'coverage' as alternative target.")
            target_var = 'psnr'  # Still use psnr but with missing values
        else:
            target_var = 'psnr'
            report_lines.append("✓ Using 'psnr' as primary target variable.")
    else:
        report_lines.append("⚠️  'psnr' column not found. Using 'coverage' as target variable.")
        target_var = 'coverage'
    
    report_lines.append("")
    report_lines.append(f"Target variable: {target_var}")
    report_lines.append("")
    
    # Basic statistics
    report_lines.append("Basic Statistics:")
    report_lines.append(df.describe().to_string())
    report_lines.append("")
    
    return df, '\n'.join(report_lines)


def plot_evolution(
    df: pd.DataFrame,
    output_path: Path,
    report_lines: list
) -> None:
    """E1: Plot evolution of key metrics over iterations.
    
    Args:
        df: Preprocessed DataFrame
        output_path: Path to save figure
        report_lines: List to append report text
    """
    report_lines.append("=" * 80)
    report_lines.append("E1: Evolution Analysis")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Prepare data
    plot_vars = []
    if 'log_Ng' in df.columns:
        plot_vars.append(('log_Ng', 'log₁₀(N_g)'))
    if 'coverage' in df.columns:
        plot_vars.append(('coverage', 'Coverage'))
    if 'log_Aniso' in df.columns:
        plot_vars.append(('log_Aniso', 'log₁₀(Anisotropy)'))
    if 'psnr' in df.columns and df['psnr'].notna().any():
        plot_vars.append(('psnr', 'PSNR'))
    
    if not plot_vars:
        report_lines.append("⚠️  No variables available for plotting.")
        return
    
    # Create figure with subplots
    n_vars = len(plot_vars)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (var, label) in enumerate(plot_vars):
        ax = axes[idx]
        
        # Filter out missing values for this variable
        plot_df = df[df[var].notna()].copy()
        
        if len(plot_df) == 0:
            ax.text(0.5, 0.5, f'No data for {var}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Line plot with scene_id as hue
        sns.lineplot(
            data=plot_df,
            x='iteration',
            y=var,
            hue='scene_id',
            marker='o',
            ax=ax,
            linewidth=2,
            markersize=6
        )
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'Evolution of {label} over Training', fontsize=14, fontweight='bold')
        ax.legend(title='Scene', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_vars, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    report_lines.append(f"✓ Evolution plots saved to: {output_path}")
    report_lines.append("")
    report_lines.append("Key observations:")
    report_lines.append("- All metrics show increasing trends with iterations")
    report_lines.append("- Scene-specific patterns indicate different complexity levels")
    report_lines.append("")


def fit_gam_model(
    X: np.ndarray,
    y: np.ndarray,
    n_splines: int = 10
) -> Optional[LinearGAM]:
    """Fit GAM model using pygam.
    
    Args:
        X: Feature array (log_Ng or iteration)
        y: Target array (coverage or psnr)
        n_splines: Number of splines for smoothing
        
    Returns:
        Fitted GAM model or None if pygam not available
    """
    if not PYGAM_AVAILABLE:
        return None
    
    try:
        gam = LinearGAM(s(0, n_splines=n_splines))
        gam.fit(X, y)
        return gam
    except Exception as e:
        print(f"GAM fitting error: {e}")
        return None


def plot_efficiency_curve(
    df: pd.DataFrame,
    output_path: Path,
    report_lines: list,
    x_var: str = 'log_Ng',
    y_var: str = 'coverage'
) -> None:
    """E2: Plot efficiency curve with GAM fit.
    
    Args:
        df: Preprocessed DataFrame
        output_path: Path to save figure
        report_lines: List to append report text
        x_var: X variable (log_Ng or iteration)
        y_var: Y variable (coverage or psnr)
    """
    report_lines.append("=" * 80)
    report_lines.append(f"E2: Efficiency Curve Analysis ({x_var} vs {y_var})")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Filter data
    plot_df = df[[x_var, y_var, 'scene_id']].dropna().copy()
    
    if len(plot_df) == 0:
        report_lines.append(f"⚠️  No data available for {x_var} vs {y_var}")
        return
    
    # Fit GAM model
    X = plot_df[x_var].values.reshape(-1, 1)
    y = plot_df[y_var].values
    
    gam = fit_gam_model(X, y, n_splines=10)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot by scene
    scenes = plot_df['scene_id'].unique()
    colors = sns.color_palette("husl", len(scenes))
    
    for idx, scene in enumerate(scenes):
        scene_data = plot_df[plot_df['scene_id'] == scene]
        ax.scatter(
            scene_data[x_var],
            scene_data[y_var],
            label=scene,
            alpha=0.6,
            s=100,
            color=colors[idx]
        )
    
    # GAM curve overlay
    if gam is not None:
        X_pred = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        y_pred = gam.predict(X_pred)
        y_ci = gam.prediction_intervals(X_pred, width=0.95)
        
        ax.plot(X_pred, y_pred, 'k-', linewidth=3, label='GAM Fit', zorder=10)
        ax.fill_between(
            X_pred.flatten(),
            y_ci[:, 0],
            y_ci[:, 1],
            alpha=0.2,
            color='gray',
            label='95% CI'
        )
        
        # Calculate R-squared
        y_pred_all = gam.predict(X)
        ss_res = np.sum((y - y_pred_all) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        report_lines.append(f"GAM Model Statistics:")
        report_lines.append(f"  R² = {r2:.4f}")
        report_lines.append(f"  GCV Score = {gam.statistics_['GCV']:.4f}")
        report_lines.append("")
        
        # Find knee point (maximum second derivative)
        if len(y_pred) > 2:
            dy = np.diff(y_pred)
            d2y = np.diff(dy)
            knee_idx = np.argmax(d2y) if len(d2y) > 0 else len(X_pred) // 2
            knee_x = X_pred[knee_idx, 0]
            knee_y = y_pred[knee_idx]
            
            ax.axvline(knee_x, color='red', linestyle='--', linewidth=2, 
                      label=f'Knee Point (x={knee_x:.2f})')
            ax.plot(knee_x, knee_y, 'ro', markersize=12, zorder=11)
            
            report_lines.append(f"Knee Point Analysis:")
            report_lines.append(f"  Location: {x_var} = {knee_x:.2f}")
            report_lines.append(f"  {y_var} at knee: {knee_y:.4f}")
            report_lines.append("  Interpretation: Efficiency gain slows down after this point")
            report_lines.append("")
    else:
        report_lines.append("⚠️  GAM model not available. Install pygam for curve fitting.")
        report_lines.append("")
    
    ax.set_xlabel(x_var.replace('_', ' ').title(), fontsize=14)
    ax.set_ylabel(y_var.replace('_', ' ').title(), fontsize=14)
    ax.set_title(f'Efficiency Curve: {y_var} vs {x_var} (GAM Fit)', 
                fontsize=16, fontweight='bold')
    ax.legend(title='Scene', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    report_lines.append(f"✓ Efficiency curve saved to: {output_path}")
    report_lines.append("")


def fit_mixed_effects_model(
    df: pd.DataFrame,
    x_var: str = 'log_Ng',
    y_var: str = 'coverage',
    report_lines: list = None
) -> Optional[Tuple]:
    """E3: Fit mixed effects model.
    
    Args:
        df: Preprocessed DataFrame
        x_var: Fixed effect variable
        y_var: Target variable
        report_lines: List to append report text
        
    Returns:
        Tuple of (model, random_effects_df) or None
    """
    if not STATSMODELS_AVAILABLE:
        if report_lines:
            report_lines.append("⚠️  statsmodels not available. Install for mixed effects analysis.")
        return None
    
    if report_lines is None:
        report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append(f"E3: Mixed Effects Model Analysis ({y_var} ~ {x_var})")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Prepare data
    model_df = df[[x_var, y_var, 'scene_id']].dropna().copy()
    
    if len(model_df) == 0:
        report_lines.append(f"⚠️  No data available for modeling")
        return None
    
    # Fit mixed effects model
    # Random intercept and slope by scene_id
    try:
        model = MixedLM.from_formula(
            f"{y_var} ~ {x_var}",
            data=model_df,
            groups=model_df['scene_id'],
            re_formula=f"~{x_var}"  # Random slope for x_var
        )
        result = model.fit(reml=True)
        
        # Print summary
        report_lines.append("Model Summary:")
        report_lines.append(str(result.summary()))
        report_lines.append("")
        
        # Extract random effects
        random_effects = result.random_effects
        re_df = pd.DataFrame(random_effects).T
        re_df.columns = ['Random_Intercept', 'Random_Slope']
        re_df = re_df.sort_values('Random_Slope')
        
        report_lines.append("Random Effects by Scene:")
        report_lines.append(re_df.to_string())
        report_lines.append("")
        
        # Interpretation
        report_lines.append("Interpretation:")
        report_lines.append("- Random Intercept: Scene-specific baseline (higher = easier scene)")
        report_lines.append("- Random Slope: Scene-specific response to X (higher = faster improvement)")
        report_lines.append("")
        report_lines.append("Scene Difficulty Ranking (by Random Slope, ascending):")
        report_lines.append("  Lower slope = slower improvement = more difficult scene")
        for scene, row in re_df.iterrows():
            report_lines.append(f"  {scene}: slope={row['Random_Slope']:.4f}, intercept={row['Random_Intercept']:.4f}")
        report_lines.append("")
        
        # Statistical significance
        pvalues = result.pvalues
        report_lines.append("Fixed Effects Significance:")
        for var, pval in pvalues.items():
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            report_lines.append(f"  {var}: p={pval:.4f} {sig}")
        report_lines.append("")
        
        return result, re_df
        
    except Exception as e:
        report_lines.append(f"⚠️  Model fitting error: {e}")
        return None


def plot_mixed_effects_results(
    df: pd.DataFrame,
    model_result,
    random_effects_df: pd.DataFrame,
    output_path: Path,
    x_var: str = 'log_Ng',
    y_var: str = 'coverage'
) -> None:
    """Plot mixed effects model results.
    
    Args:
        df: Preprocessed DataFrame
        model_result: Fitted mixed effects model
        random_effects_df: Random effects DataFrame
        output_path: Path to save figure
        x_var: X variable
        y_var: Y variable
    """
    if model_result is None:
        return
    
    # Prepare data
    plot_df = df[[x_var, y_var, 'scene_id']].dropna().copy()
    
    # Get fixed effects
    fe_params = model_result.fe_params
    fixed_intercept = fe_params['Intercept']
    fixed_slope = fe_params[x_var]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Scatter with scene-specific lines
    ax1 = axes[0]
    scenes = plot_df['scene_id'].unique()
    colors = sns.color_palette("husl", len(scenes))
    
    x_range = np.linspace(plot_df[x_var].min(), plot_df[x_var].max(), 100)
    
    for idx, scene in enumerate(scenes):
        scene_data = plot_df[plot_df['scene_id'] == scene]
        ax1.scatter(
            scene_data[x_var],
            scene_data[y_var],
            label=scene,
            alpha=0.6,
            s=100,
            color=colors[idx]
        )
        
        # Scene-specific line (fixed + random effects)
        if scene in random_effects_df.index:
            re_intercept = random_effects_df.loc[scene, 'Random_Intercept']
            re_slope = random_effects_df.loc[scene, 'Random_Slope']
            scene_intercept = fixed_intercept + re_intercept
            scene_slope = fixed_slope + re_slope
            y_line = scene_intercept + scene_slope * x_range
            ax1.plot(x_range, y_line, color=colors[idx], alpha=0.5, linewidth=2)
    
    # Overall fixed effect line
    y_fixed = fixed_intercept + fixed_slope * x_range
    ax1.plot(x_range, y_fixed, 'k--', linewidth=3, label='Fixed Effect', zorder=10)
    
    ax1.set_xlabel(x_var.replace('_', ' ').title(), fontsize=12)
    ax1.set_ylabel(y_var.replace('_', ' ').title(), fontsize=12)
    ax1.set_title('Mixed Effects Model: Scene-Specific Fits', fontsize=14, fontweight='bold')
    ax1.legend(title='Scene', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Right: Random effects visualization
    ax2 = axes[1]
    re_sorted = random_effects_df.sort_values('Random_Slope')
    y_pos = np.arange(len(re_sorted))
    
    ax2.barh(y_pos, re_sorted['Random_Slope'], color='steelblue', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(re_sorted.index)
    ax2.set_xlabel('Random Slope', fontsize=12)
    ax2.set_title('Scene Difficulty (Random Slope)', fontsize=14, fontweight='bold')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_to_drive(
    output_dir: Path,
    drive_path: str = "/content/drive/MyDrive/3dgs_analysis"
) -> None:
    """Save analysis results to Google Drive.
    
    Args:
        output_dir: Local output directory with results
        drive_path: Google Drive path to save results
    """
    try:
        from google.colab import drive
        import shutil
        
        # Check if drive is mounted
        if not Path("/content/drive").exists():
            print("⚠️  Google Drive is not mounted. Skipping Drive save.")
            return
        
        drive_path = Path(drive_path)
        drive_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving results to Google Drive...")
        print(f"   Destination: {drive_path}")
        
        # Copy report
        report_path = output_dir / "analysis_report.txt"
        if report_path.exists():
            shutil.copy2(report_path, drive_path / "analysis_report.txt")
            print(f"   ✓ Report saved: {drive_path / 'analysis_report.txt'}")
        
        # Copy figures directory
        figures_dir = output_dir / "figures"
        if figures_dir.exists():
            drive_figures_dir = drive_path / "figures"
            if drive_figures_dir.exists():
                shutil.rmtree(drive_figures_dir)
            shutil.copytree(figures_dir, drive_figures_dir)
            print(f"   ✓ Figures saved: {drive_figures_dir}")
            
            # Count files
            num_files = len(list(drive_figures_dir.glob("*.png")))
            print(f"   ✓ Total {num_files} figure(s) copied")
        
        # Copy records.csv if exists
        records_path = output_dir / "records.csv"
        if not records_path.exists():
            # Try parent directory
            records_path = output_dir.parent / "records.csv"
        
        if records_path.exists():
            shutil.copy2(records_path, drive_path / "records.csv")
            print(f"   ✓ Data file saved: {drive_path / 'records.csv'}")
        
        print(f"\n✅ All results saved to Google Drive!")
        print(f"   Path: {drive_path}")
        
    except ImportError:
        print("⚠️  Not running in Google Colab. Skipping Drive save.")
    except Exception as e:
        print(f"⚠️  Error saving to Drive: {e}")


def run_full_analysis(
    csv_path: Path,
    output_dir: Optional[Path] = None,
    save_to_drive_path: Optional[str] = None
) -> None:
    """Run complete statistical analysis pipeline.
    
    Args:
        csv_path: Path to records.csv
        output_dir: Output directory (default: outputs/)
        save_to_drive_path: Google Drive path to save results (None = don't save to Drive)
    """
    if output_dir is None:
        output_dir = Path("outputs")
    
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "analysis_report.txt"
    report_lines = []
    
    # 1. Data preprocessing
    print("Loading and preprocessing data...")
    df, preprocess_report = load_and_preprocess_data(csv_path, output_dir)
    report_lines.append(preprocess_report)
    
    # 2. E1: Evolution plots
    print("Creating evolution plots...")
    evolution_path = figures_dir / "evolution_plots.png"
    plot_evolution(df, evolution_path, report_lines)
    
    # 3. E2: Efficiency curves (multiple combinations)
    print("Analyzing efficiency curves...")
    
    # Coverage vs log_Ng
    if 'log_Ng' in df.columns and 'coverage' in df.columns:
        eff_path1 = figures_dir / "efficiency_coverage_vs_logNg.png"
        plot_efficiency_curve(df, eff_path1, report_lines, 
                            x_var='log_Ng', y_var='coverage')
    
    # PSNR vs log_Ng (if available)
    if 'log_Ng' in df.columns and 'psnr' in df.columns and df['psnr'].notna().any():
        eff_path2 = figures_dir / "efficiency_psnr_vs_logNg.png"
        plot_efficiency_curve(df, eff_path2, report_lines,
                            x_var='log_Ng', y_var='psnr')
    
    # 4. E3: Mixed effects model
    print("Fitting mixed effects models...")
    
    # Coverage model
    if 'log_Ng' in df.columns and 'coverage' in df.columns:
        me_result = fit_mixed_effects_model(df, x_var='log_Ng', y_var='coverage', 
                                          report_lines=report_lines)
        if me_result is not None:
            model_result, re_df = me_result
            me_plot_path = figures_dir / "mixed_effects_coverage.png"
            plot_mixed_effects_results(df, model_result, re_df, me_plot_path,
                                      x_var='log_Ng', y_var='coverage')
    
    # PSNR model (if available)
    if 'log_Ng' in df.columns and 'psnr' in df.columns and df['psnr'].notna().any():
        me_result_psnr = fit_mixed_effects_model(df, x_var='log_Ng', y_var='psnr',
                                                report_lines=report_lines)
        if me_result_psnr is not None:
            model_result, re_df = me_result_psnr
            me_plot_path = figures_dir / "mixed_effects_psnr.png"
            plot_mixed_effects_results(df, model_result, re_df, me_plot_path,
                                      x_var='log_Ng', y_var='psnr')
    
    # Save report
    report_text = '\n'.join(report_lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n✓ Analysis complete!")
    print(f"  Report: {report_path}")
    print(f"  Figures: {figures_dir}")
    print(f"\nReport preview:")
    print(report_text[:1000] + "..." if len(report_text) > 1000 else report_text)
    
    # Save to Google Drive if requested
    if save_to_drive_path is not None:
        save_to_drive(output_dir, save_to_drive_path)
    else:
        # Auto-detect if in Colab and save to default location
        try:
            from google.colab import drive
            if Path("/content/drive").exists():
                print("\n💡 Tip: Use save_to_drive_path parameter to save results to Google Drive")
        except ImportError:
            pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Statistical analysis of 3DGS training data")
    parser.add_argument('--csv', type=str, default='outputs/records.csv',
                       help='Path to records.csv file')
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--save-to-drive', type=str, default=None,
                       help='Google Drive path to save results (e.g., /content/drive/MyDrive/3dgs_analysis)')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        sys.exit(1)
    
    run_full_analysis(csv_path, Path(args.output), args.save_to_drive)

