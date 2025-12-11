"""
Complete Tumor Size Prediction Pipeline Demo
Shows:
- Tumor size prediction (mm)
- Bounding box detection and rounding
- Severity classification based on T2, ADC, DWI sequences
- Visualization with severity colors
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from size_predictor_model import SizePredictorModel
from bbox_utils import (
    BoundingBoxPostProcessor,
    SeverityClassifier,
    visualize_with_bbox_and_severity,
    round_bbox_corners
)
from utils_dicom import load_dicom_series
from infer_with_bbox import InferenceWithBBox


def load_sample_data(data_dir: str, patient_id: str = "ProstateX-0000"):
    """Load sample DICOM data for demo"""
    patient_path = Path(data_dir) / "PROSTATEx" / patient_id
    
    if not patient_path.exists():
        print(f"❌ Patient directory not found: {patient_path}")
        return None
    
    # Try to load T2, ADC, DWI sequences
    sequences = {}
    for seq_type in ['T2', 'ADC', 'DWI']:
        seq_path = patient_path / seq_type
        if seq_path.exists():
            try:
                series, _ = load_dicom_series(str(seq_path))
                sequences[seq_type] = series
                print(f"✓ Loaded {seq_type}: shape {series.shape}")
            except Exception as e:
                print(f"⚠ Error loading {seq_type}: {e}")
    
    return sequences


def create_synthetic_data():
    """Create synthetic DICOM-like data for demo when real data not available"""
    print("\n📊 Creating synthetic data for demonstration...")
    
    sequences = {
        'T2': np.random.randint(50, 200, (128, 128, 20), dtype=np.uint8),
        'ADC': np.random.randint(30, 150, (128, 128, 20), dtype=np.uint8),
        'DWI': np.random.randint(100, 250, (128, 128, 20), dtype=np.uint8)
    }
    
    print(f"  T2 shape:  {sequences['T2'].shape}")
    print(f"  ADC shape: {sequences['ADC'].shape}")
    print(f"  DWI shape: {sequences['DWI'].shape}")
    
    return sequences


def normalize_sequence(data: np.ndarray, window_min: int = 0, window_max: int = 255) -> np.ndarray:
    """Normalize sequence to 0-255 range"""
    data = data.astype(np.float32)
    data_min = np.percentile(data, 2)
    data_max = np.percentile(data, 98)
    
    if data_max > data_min:
        data = ((data - data_min) / (data_max - data_min)) * (window_max - window_min) + window_min
    
    return np.clip(data, window_min, window_max).astype(np.uint8)


def create_tumor_region(size_mm: float = 25.0, center_slice: int = 10, 
                        sequence_shape: tuple = (128, 128, 20)) -> np.ndarray:
    """Create synthetic tumor region with realistic intensity patterns"""
    mask = np.zeros(sequence_shape, dtype=np.float32)
    
    # Convert mm to pixels (assuming ~0.5mm per pixel)
    size_px = int(size_mm / 0.5)
    
    # Create ellipsoid tumor
    z, y, x = np.ogrid[-size_px:size_px+1, -size_px:size_px+1, -size_px:size_px+1]
    ellipsoid = (x**2 + (y**1.5) + (z**1.5)) <= size_px**2
    
    # Place tumor at center
    center_y, center_x = 64, 64
    y_start = max(0, center_y - size_px)
    y_end = min(sequence_shape[0], center_y + size_px + 1)
    x_start = max(0, center_x - size_px)
    x_end = min(sequence_shape[1], center_x + size_px + 1)
    
    mask[center_slice-2:center_slice+3, y_start:y_end, x_start:x_end] = ellipsoid
    
    return mask


def inject_tumor_into_sequences(sequences: dict, size_mm: float = 25.0) -> dict:
    """Inject synthetic tumor into sequences with realistic intensity patterns"""
    print(f"\n🎯 Injecting synthetic tumor (size: {size_mm:.1f}mm)...")
    
    tumor_mask = create_tumor_region(size_mm, sequence_shape=sequences['T2'].shape)
    
    # Add tumor with realistic intensity modifications
    for seq_type in sequences:
        base = sequences[seq_type].astype(np.float32)
        
        if seq_type == 'T2':
            # T2: hyperintense tumor
            sequences[seq_type] = np.clip(base + tumor_mask * 80, 0, 255).astype(np.uint8)
        elif seq_type == 'ADC':
            # ADC: hyperintense tumor
            sequences[seq_type] = np.clip(base + tumor_mask * 60, 0, 255).astype(np.uint8)
        elif seq_type == 'DWI':
            # DWI: hyperintense tumor
            sequences[seq_type] = np.clip(base + tumor_mask * 100, 0, 255).astype(np.uint8)
    
    return sequences, tumor_mask


def predict_tumor_properties(sequences: dict, model: SizePredictorModel, 
                            device: str = 'cpu') -> dict:
    """Predict tumor size, bounding box, and severity"""
    print("\n🔮 Predicting tumor properties...")
    
    # Normalize sequences
    t2_norm = normalize_sequence(sequences['T2'])
    adc_norm = normalize_sequence(sequences['ADC'])
    dwi_norm = normalize_sequence(sequences['DWI'])
    
    # Create input tensor: (B, C, D, H, W) - batch_size=1, channels=3
    x = np.stack([t2_norm, adc_norm, dwi_norm], axis=0)
    x = torch.FloatTensor(x).unsqueeze(0).to(device)
    
    # Predict size
    with torch.no_grad():
        size_pred = model(x)
    
    predicted_size = size_pred.item() if hasattr(size_pred, 'item') else size_pred
    
    print(f"  📏 Predicted tumor size: {predicted_size:.2f} mm")
    
    # Post-process bounding box
    bbox_processor = BoundingBoxPostProcessor()
    bbox = bbox_processor.estimate_bbox_from_size(predicted_size, image_shape=(128, 128))
    bbox_rounded = round_bbox_corners(bbox, grid_size=8)
    
    print(f"  📦 Bounding box: {bbox}")
    print(f"  📦 Rounded box: {bbox_rounded}")
    
    # Classify severity
    severity_classifier = SeverityClassifier()
    severity = severity_classifier.classify_severity(
        size_mm=predicted_size,
        t2_intensity=np.mean(t2_norm),
        adc_intensity=np.mean(adc_norm),
        dwi_intensity=np.mean(dwi_norm)
    )
    
    print(f"  ⚠️  Severity: {severity['class']} (Score: {severity['score']:.3f})")
    print(f"     Risk: {severity['risk_level']}")
    
    return {
        'size_mm': predicted_size,
        'bbox': bbox,
        'bbox_rounded': bbox_rounded,
        'severity': severity,
        'sequences_normalized': {
            'T2': t2_norm,
            'ADC': adc_norm,
            'DWI': dwi_norm
        }
    }


def create_output_report(results: dict, true_size: float = None) -> str:
    """Create formatted report of predictions"""
    report = []
    report.append("\n" + "="*70)
    report.append("TUMOR SIZE PREDICTION & SEVERITY ANALYSIS REPORT")
    report.append("="*70)
    
    # Size prediction
    report.append(f"\n📏 TUMOR SIZE PREDICTION")
    report.append(f"  Predicted size: {results['size_mm']:.2f} mm")
    if true_size is not None:
        error = abs(results['size_mm'] - true_size)
        percent_error = (error / true_size) * 100
        report.append(f"  True size:      {true_size:.2f} mm")
        report.append(f"  Absolute error: {error:.2f} mm ({percent_error:.1f}%)")
    
    # Bounding box
    report.append(f"\n📦 BOUNDING BOX DETECTION")
    y1, x1, y2, x2 = results['bbox']
    report.append(f"  Original: y:[{y1:.1f}, {y2:.1f}], x:[{x1:.1f}, {x2:.1f}]")
    
    y1r, x1r, y2r, x2r = results['bbox_rounded']
    report.append(f"  Rounded:  y:[{int(y1r)}, {int(y2r)}], x:[{int(x1r)}, {int(x2r)}]")
    report.append(f"  Width:  {x2r - x1r:.1f} px, Height: {y2r - y1r:.1f} px")
    
    # Severity
    report.append(f"\n⚠️  SEVERITY CLASSIFICATION")
    sev = results['severity']
    report.append(f"  Class:      {sev['class']}")
    report.append(f"  Risk Level: {sev['risk_level']}")
    report.append(f"  Score:      {sev['score']:.3f}")
    report.append(f"  Confidence: {sev['confidence']:.1%}")
    
    # Sequence statistics
    report.append(f"\n📊 SEQUENCE STATISTICS")
    report.append(f"  T2  mean intensity:  {np.mean(results['sequences_normalized']['T2']):.1f}")
    report.append(f"  ADC mean intensity: {np.mean(results['sequences_normalized']['ADC']):.1f}")
    report.append(f"  DWI mean intensity: {np.mean(results['sequences_normalized']['DWI']):.1f}")
    
    report.append("\n" + "="*70)
    
    return "\n".join(report)


def main():
    """Run complete pipeline demo"""
    print("\n" + "="*70)
    print("🏥 TUMOR SIZE PREDICTION & SEVERITY ANALYSIS PIPELINE")
    print("="*70)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n💻 Using device: {device}")
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    model_dir = project_root / 'models'
    
    # Try to load real data, fall back to synthetic
    print(f"\n📂 Data directory: {data_dir}")
    sequences = load_sample_data(str(data_dir))
    
    if sequences is None or len(sequences) == 0:
        print("\n⚠️  No DICOM data found, using synthetic data")
        sequences = create_synthetic_data()
    
    # Inject synthetic tumor for demo
    true_size = 25.0
    sequences, tumor_mask = inject_tumor_into_sequences(sequences, size_mm=true_size)
    
    # Load or create model
    print(f"\n🤖 Loading tumor size prediction model...")
    model = SizePredictorModel(in_channels=3, out_channels=1)
    
    # Try to load pre-trained weights
    model_path = model_dir / 'size_predictor_model.pth'
    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"  ✓ Loaded weights from {model_path}")
        except Exception as e:
            print(f"  ⚠️  Could not load weights: {e}")
            print(f"  Using random initialization")
    else:
        print(f"  ⚠️  Model not found at {model_path}")
        print(f"  Using random initialization (train first for accurate predictions)")
    
    model = model.to(device)
    model.eval()
    
    # Run predictions
    results = predict_tumor_properties(sequences, model, device)
    
    # Generate report
    report = create_output_report(results, true_size=true_size)
    print(report)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = project_root / f'demo_report_{timestamp}.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\n💾 Report saved to: {report_file}")
    
    # Save results as JSON
    json_file = project_root / f'demo_results_{timestamp}.json'
    results_json = {
        'timestamp': timestamp,
        'predicted_size_mm': float(results['size_mm']),
        'true_size_mm': true_size,
        'bbox': [float(x) for x in results['bbox']],
        'bbox_rounded': [int(x) for x in results['bbox_rounded']],
        'severity': {
            'class': results['severity']['class'],
            'risk_level': results['severity']['risk_level'],
            'score': float(results['severity']['score']),
            'confidence': float(results['severity']['confidence'])
        }
    }
    
    with open(json_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"💾 Results saved to: {json_file}")
    
    print("\n✅ Demo complete!\n")


if __name__ == '__main__':
    main()
