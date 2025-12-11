#!/usr/bin/env python3
"""
Complete Tumor Size Prediction Pipeline Demo
Demonstrates:
1. Loading multi-sequence DICOM (T2, ADC, DWI)
2. Tumor size prediction
3. Bounding box generation
4. Severity classification (TNM staging)
5. Visualization
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from src.size_predictor_model import SizePredictorModel
from src.bbox_utils import predict_bbox, create_severity_report
from src.utils_dicom import load_dicom_series
from src.visualization_enhanced import visualize_predictions
from src.config import Config


def load_sample_data(sample_id="ProstateX-0000"):
    """Load sample DICOM data"""
    data_dir = Path(__file__).parent.parent / "data" / "PROSTATEx"
    sample_dir = data_dir / sample_id
    
    if not sample_dir.exists():
        print(f"Sample {sample_id} not found at {sample_dir}")
        return None
    
    sequences = {}
    for seq in ['T2', 'ADC', 'DWI']:
        seq_dir = sample_dir / seq
        if seq_dir.exists():
            try:
                dicom_files = list(seq_dir.glob('*.dcm'))
                if dicom_files:
                    sequences[seq] = load_dicom_series(str(seq_dir))
                    print(f"✓ Loaded {seq}: {sequences[seq].shape}")
            except Exception as e:
                print(f"✗ Failed to load {seq}: {e}")
    
    return sequences if sequences else None


def run_complete_pipeline(sample_id="ProstateX-0000"):
    """Run complete tumor size prediction pipeline"""
    
    print("\n" + "="*70)
    print("PROSTATE TUMOR SIZE PREDICTION - COMPLETE PIPELINE")
    print("="*70 + "\n")
    
    # 1. Load data
    print("[1] Loading Multi-Sequence DICOM Data...")
    sequences = load_sample_data(sample_id)
    if not sequences:
        print("Could not load sample data. Using synthetic demo data...")
        sequences = create_synthetic_data()
    
    # 2. Initialize model
    print("\n[2] Initializing Tumor Size Predictor Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    model = SizePredictorModel(
        in_channels=3,  # T2, ADC, DWI
        hidden_dim=64,
        device=device
    )
    
    # Try to load pretrained weights
    model_path = Path(__file__).parent.parent / "models" / "size_predictor.pth"
    if model_path.exists():
        try:
            model.load_weights(str(model_path))
            print(f"   ✓ Loaded pretrained model from {model_path}")
        except:
            print(f"   ! Could not load pretrained model, using random initialization")
    else:
        print(f"   ! Model checkpoint not found at {model_path}")
    
    # 3. Prepare input data
    print("\n[3] Preparing Input Data...")
    if 'T2' in sequences and 'ADC' in sequences and 'DWI' in sequences:
        # Use real sequences
        t2 = sequences['T2']
        adc = sequences['ADC']
        dwi = sequences['DWI']
        
        # Normalize and stack
        t2_norm = (t2 - t2.min()) / (t2.max() - t2.min() + 1e-8)
        adc_norm = (adc - adc.min()) / (adc.max() - adc.min() + 1e-8)
        dwi_norm = (dwi - dwi.min()) / (dwi.max() - dwi.min() + 1e-8)
    else:
        # Use synthetic
        t2_norm = sequences.get('T2', np.random.rand(256, 256))
        adc_norm = sequences.get('ADC', np.random.rand(256, 256))
        dwi_norm = sequences.get('DWI', np.random.rand(256, 256))
    
    print(f"   T2 shape: {t2_norm.shape}")
    print(f"   ADC shape: {adc_norm.shape}")
    print(f"   DWI shape: {dwi_norm.shape}")
    
    # 4. Predict tumor size
    print("\n[4] Predicting Tumor Size...")
    tumor_size_mm = model.predict(t2_norm, adc_norm, dwi_norm)
    print(f"   ✓ Predicted tumor size: {tumor_size_mm:.2f} mm")
    
    # 5. Generate bounding box
    print("\n[5] Generating Bounding Box...")
    bbox, confidence = predict_bbox(t2_norm, tumor_size_mm)
    x1, y1, x2, y2 = bbox
    print(f"   ✓ Bounding box: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
    print(f"   Confidence: {confidence:.2%}")
    
    # 6. Classify severity (TNM staging)
    print("\n[6] Classifying Tumor Severity (TNM Staging)...")
    severity_report = create_severity_report(tumor_size_mm, bbox)
    print(f"   T-Stage: {severity_report['t_stage']}")
    print(f"   Severity: {severity_report['severity']}")
    print(f"   Clinical Notes: {severity_report['clinical_notes']}")
    
    # 7. Create visualization
    print("\n[7] Creating Visualizations...")
    try:
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        # Stack sequences for visualization
        stacked = np.stack([t2_norm, adc_norm, dwi_norm], axis=0)
        
        viz_path = visualize_predictions(
            stacked,
            bbox,
            tumor_size_mm,
            severity_report,
            output_path=str(output_dir / f"{sample_id}_prediction.png")
        )
        print(f"   ✓ Saved visualization to: {viz_path}")
    except Exception as e:
        print(f"   ! Visualization failed: {e}")
    
    # 8. Generate report
    print("\n[8] Generating Report...")
    report = {
        "sample_id": sample_id,
        "tumor_size_mm": float(tumor_size_mm),
        "bounding_box": {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": float(confidence)
        },
        "severity": severity_report,
        "device": str(device),
        "timestamp": str(Path(__file__).parent.parent / "outputs" / f"{sample_id}_report.json")
    }
    
    # Save report
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / f"{sample_id}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"   ✓ Saved report to: {report_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    print(f"Sample ID: {sample_id}")
    print(f"Tumor Size: {tumor_size_mm:.2f} mm")
    print(f"T-Stage: {severity_report['t_stage']}")
    print(f"Severity Level: {severity_report['severity']}")
    print(f"Bounding Box: ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")
    print(f"Confidence: {confidence:.2%}")
    print("="*70 + "\n")
    
    return report


def create_synthetic_data():
    """Create synthetic DICOM data for demo"""
    print("   Creating synthetic data...")
    h, w = 256, 256
    
    # Create tumor-like structures
    y, x = np.ogrid[:h, :w]
    cx, cy = h//2, w//2
    
    # Gaussian blob for tumor
    tumor_mask = np.exp(-((x - cx)**2 + (y - cy)**2) / (30**2))
    
    sequences = {
        'T2': (np.random.rand(h, w) * 0.3 + tumor_mask * 0.7) * 4095,
        'ADC': (np.random.rand(h, w) * 0.4 + tumor_mask * 0.6) * 4095,
        'DWI': (np.random.rand(h, w) * 0.2 + tumor_mask * 0.8) * 4095,
    }
    
    return sequences


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tumor Size Prediction Demo")
    parser.add_argument('--sample-id', default='ProstateX-0000',
                       help='Sample ID to process')
    parser.add_argument('--output-dir', default='./outputs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run pipeline
    report = run_complete_pipeline(args.sample_id)
    
    print("\n✓ Pipeline completed successfully!")
    print(f"Check outputs directory for detailed results.")


if __name__ == '__main__':
    main()
