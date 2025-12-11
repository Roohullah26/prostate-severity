#!/usr/bin/env python3
"""
Complete Tumor Size Prediction Pipeline
- Predicts tumor size from multi-sequence MRI (T2, ADC, DWI)
- Generates bounding boxes
- Classifies severity based on size
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from size_predictor_model import TumorSizePredictorMultiSeq
from bbox_utils import generate_bounding_box, apply_bbox_to_image
from utils_image import load_dicom_series
from infer_with_bbox import InferenceWithBBox


def classify_severity(tumor_size_mm):
    """
    Classify tumor severity based on size
    T2 TNM classification
    """
    if tumor_size_mm < 10:
        return "T1a", "Small (<10mm) - Early Stage"
    elif tumor_size_mm < 20:
        return "T1b", "Medium (10-20mm) - Stage I"
    elif tumor_size_mm < 30:
        return "T1c", "Medium-Large (20-30mm) - Stage II"
    elif tumor_size_mm < 50:
        return "T2", "Large (30-50mm) - Stage III"
    else:
        return "T3+", "Very Large (>50mm) - Advanced Stage"


def run_complete_pipeline(t2_path, adc_path, dwi_path, output_dir="output"):
    """
    Run complete tumor size prediction pipeline
    """
    print("\n" + "="*60)
    print("TUMOR SIZE PREDICTION PIPELINE")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load images
        print("\n[1/6] Loading DICOM images...")
        if os.path.isdir(t2_path):
            t2_img = load_dicom_series(t2_path)
        else:
            t2_img = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)
        
        if os.path.isdir(adc_path):
            adc_img = load_dicom_series(adc_path)
        else:
            adc_img = cv2.imread(adc_path, cv2.IMREAD_GRAYSCALE)
        
        if os.path.isdir(dwi_path):
            dwi_img = load_dicom_series(dwi_path)
        else:
            dwi_img = cv2.imread(dwi_path, cv2.IMREAD_GRAYSCALE)
        
        print(f"   ✓ T2 shape: {t2_img.shape}")
        print(f"   ✓ ADC shape: {adc_img.shape}")
        print(f"   ✓ DWI shape: {dwi_img.shape}")
        
        # Initialize model
        print("\n[2/6] Loading tumor size prediction model...")
        model = TumorSizePredictorMultiSeq()
        print("   ✓ Model loaded successfully")
        
        # Prepare input
        print("\n[3/6] Preparing multi-sequence input...")
        input_data = np.stack([t2_img, adc_img, dwi_img], axis=0)
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
        print(f"   ✓ Input shape: {input_data.shape}")
        
        # Predict tumor size
        print("\n[4/6] Predicting tumor size...")
        tumor_size_mm = model.predict(input_data)
        print(f"   ✓ Predicted tumor size: {tumor_size_mm:.2f} mm")
        
        # Generate bounding box
        print("\n[5/6] Generating bounding box...")
        bbox = generate_bounding_box(dwi_img, tumor_size_mm)
        print(f"   ✓ Bounding box: {bbox}")
        
        # Classify severity
        print("\n[6/6] Classifying severity...")
        tnm_stage, severity_desc = classify_severity(tumor_size_mm)
        print(f"   ✓ TNM Stage: {tnm_stage}")
        print(f"   ✓ Severity: {severity_desc}")
        
        # Visualize results
        print("\n" + "-"*60)
        print("VISUALIZATION")
        print("-"*60)
        
        # Draw bbox on DWI image
        dwi_with_bbox = apply_bbox_to_image(dwi_img, bbox)
        output_path = os.path.join(output_dir, "tumor_with_bbox.png")
        cv2.imwrite(output_path, dwi_with_bbox)
        print(f"   ✓ Saved: {output_path}")
        
        # Create composite visualization
        h, w = dwi_img.shape
        composite = np.zeros((h, w*3), dtype=np.uint8)
        composite[:, :w] = t2_img
        composite[:, w:w*2] = adc_img
        composite[:, w*2:] = dwi_with_bbox
        
        composite_path = os.path.join(output_dir, "multi_sequence_with_bbox.png")
        cv2.imwrite(composite_path, composite)
        print(f"   ✓ Saved: {composite_path}")
        
        # Generate report
        print("\n" + "="*60)
        print("FINAL REPORT")
        print("="*60)
        print(f"Tumor Size:       {tumor_size_mm:.2f} mm")
        print(f"TNM Stage:        {tnm_stage}")
        print(f"Severity:         {severity_desc}")
        print(f"Bounding Box:     {bbox}")
        print("="*60 + "\n")
        
        # Save report
        report_path = os.path.join(output_dir, "report.txt")
        with open(report_path, 'w') as f:
            f.write("TUMOR SIZE PREDICTION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Tumor Size:       {tumor_size_mm:.2f} mm\n")
            f.write(f"TNM Stage:        {tnm_stage}\n")
            f.write(f"Severity:         {severity_desc}\n")
            f.write(f"Bounding Box:     {bbox}\n")
            f.write("\nIMAGE INFORMATION:\n")
            f.write(f"T2 Shape:         {t2_img.shape}\n")
            f.write(f"ADC Shape:        {adc_img.shape}\n")
            f.write(f"DWI Shape:        {dwi_img.shape}\n")
        print(f"✓ Report saved: {report_path}")
        
        return {
            'tumor_size_mm': tumor_size_mm,
            'tnm_stage': tnm_stage,
            'severity': severity_desc,
            'bbox': bbox,
            'output_dir': output_dir
        }
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point"""
    if len(sys.argv) < 4:
        print("Usage: python run_tumor_size_pipeline.py <t2_path> <adc_path> <dwi_path> [output_dir]")
        print("\nExample:")
        print("  python run_tumor_size_pipeline.py data/t2.png data/adc.png data/dwi.png output/")
        sys.exit(1)
    
    t2_path = sys.argv[1]
    adc_path = sys.argv[2]
    dwi_path = sys.argv[3]
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "output"
    
    result = run_complete_pipeline(t2_path, adc_path, dwi_path, output_dir)
    
    if result:
        print("✓ Pipeline completed successfully!")
    else:
        print("✗ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
