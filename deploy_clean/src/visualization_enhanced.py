"""
Enhanced Visualization Utilities for Tumor Analysis
- Severity-based colormaps
- Bounding box overlays
- Multi-sequence visualization
- Report generation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict
import seaborn as sns


# Severity color schemes
SEVERITY_COLORS = {
    'T1a': {'rgb': (0, 128, 0), 'hex': '#008000', 'label': 'T1a (Low Risk)', 'priority': 1},
    'T1b': {'rgb': (34, 177, 76), 'hex': '#22b14c', 'label': 'T1b (Low Risk)', 'priority': 2},
    'T1c': {'rgb': (100, 200, 100), 'hex': '#64c864', 'label': 'T1c (Low-Medium Risk)', 'priority': 3},
    'T2': {'rgb': (255, 193, 7), 'hex': '#ffc107', 'label': 'T2 (Medium Risk)', 'priority': 4},
    'T3a': {'rgb': (255, 152, 0), 'hex': '#ff9800', 'label': 'T3a (High Risk)', 'priority': 5},
    'T3b': {'rgb': (255, 87, 34), 'hex': '#ff5722', 'label': 'T3b (High Risk)', 'priority': 6},
    'T4': {'rgb': (244, 67, 54), 'hex': '#f44336', 'label': 'T4 (Critical)', 'priority': 7},
}


def get_severity_color(severity_class: str, format: str = 'rgb') -> Tuple:
    """Get color for severity class"""
    color_info = SEVERITY_COLORS.get(severity_class, SEVERITY_COLORS['T2'])
    
    if format == 'hex':
        return color_info['hex']
    elif format == 'normalized':
        rgb = color_info['rgb']
        return tuple(c / 255.0 for c in rgb)
    else:  # 'rgb'
        return color_info['rgb']


def normalize_to_8bit(data: np.ndarray, percentile_low: float = 2, 
                      percentile_high: float = 98) -> np.ndarray:
    """Normalize array to 8-bit range"""
    data = data.astype(np.float32)
    vmin = np.percentile(data, percentile_low)
    vmax = np.percentile(data, percentile_high)
    
    if vmax > vmin:
        normalized = ((data - vmin) / (vmax - vmin)) * 255
    else:
        normalized = np.zeros_like(data)
    
    return np.clip(normalized, 0, 255).astype(np.uint8)


def draw_bbox_on_image(image: np.ndarray, bbox: Tuple, 
                       severity_class: str = 'T2', 
                       thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box on image
    
    Args:
        image: Input image (H, W, 3) or (H, W)
        bbox: (y_min, x_min, y_max, x_max)
        severity_class: Severity classification for color
        thickness: Line thickness
    
    Returns:
        Image with bbox overlay
    """
    # Ensure 3-channel image
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = image.copy()
    
    # Get color
    color = get_severity_color(severity_class, format='rgb')
    
    # Extract coordinates
    y_min, x_min, y_max, x_max = [int(x) for x in bbox]
    
    # Ensure coordinates are within bounds
    h, w = image.shape[:2]
    y_min = max(0, min(y_min, h-1))
    y_max = max(0, min(y_max, h-1))
    x_min = max(0, min(x_min, w-1))
    x_max = max(0, min(x_max, w-1))
    
    # Draw rectangle
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    
    return image


def visualize_multi_sequence_with_bbox(sequences: Dict[str, np.ndarray],
                                       bbox: Tuple,
                                       severity_class: str = 'T2',
                                       slice_idx: Optional[int] = None,
                                       figsize: Tuple = (15, 5),
                                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize multiple sequences with bounding box
    
    Args:
        sequences: Dict with 'T2', 'ADC', 'DWI' sequences
        bbox: Bounding box (y_min, x_min, y_max, x_max)
        severity_class: Severity for color coding
        slice_idx: Which slice to show (default: middle)
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Figure object
    """
    if slice_idx is None:
        # Use middle slice
        slice_idx = sequences[list(sequences.keys())[0]].shape[2] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    color_tuple = get_severity_color(severity_class, format='normalized')
    
    for idx, (seq_name, seq_data) in enumerate(sequences.items()):
        # Get slice
        if len(seq_data.shape) == 3:
            slice_data = seq_data[:, :, slice_idx]
        else:
            slice_data = seq_data
        
        # Normalize
        slice_norm = normalize_to_8bit(slice_data)
        
        # Draw bbox
        slice_with_bbox = draw_bbox_on_image(slice_norm, bbox, severity_class, thickness=2)
        
        # Plot
        ax = axes[idx]
        ax.imshow(slice_with_bbox, cmap='gray')
        ax.set_title(f'{seq_name} (Slice {slice_idx})', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Add bbox info in corner
        y1, x1, y2, x2 = [int(x) for x in bbox]
        bbox_text = f'Box: {x2-x1:.0f}x{y2-y1:.0f}px'
        ax.text(5, 15, bbox_text, fontsize=9, color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Add severity info to figure
    sev_color = get_severity_color(severity_class, format='hex')
    fig.suptitle(f'Tumor Detection - Severity: {severity_class} {SEVERITY_COLORS[severity_class]["label"]}',
                 fontsize=14, fontweight='bold', color=sev_color)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved: {save_path}")
    
    return fig


def create_severity_legend() -> plt.Figure:
    """Create severity classification legend"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by priority
    sorted_severities = sorted(SEVERITY_COLORS.items(), 
                               key=lambda x: x[1]['priority'])
    
    y_pos = np.arange(len(sorted_severities))
    
    for i, (sev_class, sev_info) in enumerate(sorted_severities):
        color_norm = tuple(c / 255.0 for c in sev_info['rgb'])
        ax.barh(i, 1, color=color_norm, edgecolor='black', linewidth=2)
        
        label = f"{sev_info['label']} ({sev_class})"
        ax.text(1.05, i, label, va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-0.5, len(sorted_severities) - 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.title('TNM Staging - Severity Classification', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig


def create_prediction_report(size_mm: float, bbox: Tuple, severity: Dict,
                             true_size: Optional[float] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Create visual report of predictions
    """
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle('Tumor Size Prediction & Severity Analysis Report',
                 fontsize=16, fontweight='bold')
    
    # 1. Size prediction (with error if available)
    ax1 = fig.add_subplot(gs[0, 0])
    sizes = ['Predicted']
    values = [size_mm]
    colors = ['#2196F3']
    
    if true_size is not None:
        sizes.append('True')
        values.append(true_size)
        colors.append('#4CAF50')
    
    bars = ax1.bar(sizes, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Size (mm)', fontsize=11, fontweight='bold')
    ax1.set_title('Tumor Size', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}mm', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Error (if available)
    if true_size is not None:
        ax2 = fig.add_subplot(gs[0, 1])
        error_mm = abs(size_mm - true_size)
        error_pct = (error_mm / true_size) * 100
        
        error_text = f'Absolute Error\n{error_mm:.2f} mm\n({error_pct:.1f}%)'
        
        error_color = '#4CAF50' if error_pct < 10 else '#FFC107' if error_pct < 20 else '#F44336'
        ax2.text(0.5, 0.5, error_text, ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=error_color, alpha=0.3, linewidth=2))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
    else:
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(0.5, 0.5, f'Size: {size_mm:.1f} mm', ha='center', va='center', 
                fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
    
    # 3. Bounding box info
    ax3 = fig.add_subplot(gs[1, 0])
    y1, x1, y2, x2 = bbox
    bbox_text = f"""Bounding Box
Y: [{int(y1)}, {int(y2)}]
X: [{int(x1)}, {int(x2)}]
Width: {int(x2-x1)} px
Height: {int(y2-y1)} px"""
    
    ax3.text(0.05, 0.95, bbox_text, ha='left', va='top', fontsize=11,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5, linewidth=2),
            transform=ax3.transAxes)
    ax3.axis('off')
    
    # 4. Severity classification
    ax4 = fig.add_subplot(gs[1, 1])
    sev_class = severity.get('class', 'T2')
    sev_color = get_severity_color(sev_class, format='normalized')
    
    severity_text = f"""Severity: {sev_class}
Risk: {severity.get('risk_level', 'N/A')}
Score: {severity.get('score', 0):.3f}
Confidence: {severity.get('confidence', 0):.1%}"""
    
    ax4.text(0.05, 0.95, severity_text, ha='left', va='top', fontsize=11,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor=sev_color, alpha=0.3, linewidth=2),
            transform=ax4.transAxes)
    ax4.axis('off')
    
    # 5. Severity scale
    ax5 = fig.add_subplot(gs[2, :])
    sorted_severities = sorted(SEVERITY_COLORS.items(),
                               key=lambda x: x[1]['priority'])
    
    x_pos = 0
    for sev_name, sev_info in sorted_severities:
        color_norm = tuple(c / 255.0 for c in sev_info['rgb'])
        width = 1 / len(sorted_severities)
        
        rect = patches.Rectangle((x_pos, 0), width, 1, linewidth=2,
                                 edgecolor='black',
                                 facecolor=color_norm)
        ax5.add_patch(rect)
        
        # Highlight current severity
        if sev_name == sev_class:
            ax5.add_patch(patches.Rectangle((x_pos, 0), width, 1, linewidth=4,
                                           edgecolor='white', facecolor='none'))
        
        ax5.text(x_pos + width/2, -0.15, sev_name, ha='center', fontsize=10, fontweight='bold')
        x_pos += width
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(-0.3, 1)
    ax5.axis('off')
    ax5.set_title('TNM Severity Scale', fontsize=12, fontweight='bold', loc='left')
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Report saved: {save_path}")
    
    return fig


def create_size_distribution_chart(predictions: list, true_sizes: Optional[list] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
    """Create distribution chart for multiple predictions"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(predictions, bins=20, color='#2196F3', alpha=0.7, edgecolor='black')
    if true_sizes:
        ax1.hist(true_sizes, bins=20, color='#4CAF50', alpha=0.7, edgecolor='black')
        ax1.legend(['Predicted', 'True'], fontsize=11)
    ax1.set_xlabel('Size (mm)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Size Distribution', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Scatter plot (if comparing)
    if true_sizes:
        ax2 = axes[1]
        ax2.scatter(true_sizes, predictions, s=100, alpha=0.6, edgecolor='black')
        
        # Add perfect prediction line
        min_val = min(min(true_sizes), min(predictions))
        max_val = max(max(true_sizes), max(predictions))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        ax2.set_xlabel('True Size (mm)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Predicted Size (mm)', fontsize=11, fontweight='bold')
        ax2.set_title('Predicted vs True Size', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=11)
        ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == '__main__':
    # Demo
    print("Creating severity legend...")
    fig = create_severity_legend()
    fig.savefig('severity_legend.png', dpi=150, bbox_inches='tight')
    print("✓ Saved to severity_legend.png")
    
    print("\nCreating sample report...")
    fig = create_prediction_report(
        size_mm=25.5,
        bbox=(40, 50, 80, 120),
        severity={'class': 'T2', 'risk_level': 'Medium', 'score': 0.65, 'confidence': 0.92},
        true_size=25.0,
        save_path='sample_report.png'
    )
    print("✓ Saved to sample_report.png")
