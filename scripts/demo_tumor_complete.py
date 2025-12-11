"""
Complete Tumor Size Prediction Demo with Bounding Box and Severity Classification
Demonstrates end-to-end workflow: Load multi-sequence DICOM -> Predict size -> Generate bbox -> Classify severity
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from size_predictor_model import TumorSizePredictor
from bbox_utils import BoundingBoxGenerator
from utils_dicom import load_dicom_image, get_dicom_metadata
from utils_image import normalize_image

# Add webapp to path for inference utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../webapp'))


class TumorSeverityClassifier:
    """
    Classify tumor severity based on size using T2 weighted guidelines.
    T2, ADC, and DWI sequences are used for size prediction.
    """
    
    SEVERITY_THRESHOLDS = {
        'T1': (0, 10),      # Clinically insignificant
        'T2': (10, 20),     # Localized to prostate
        'T3': (20, 50),     # Extends beyond prostate
        'T4': (50, 1000),   # Invades adjacent structures
    }
    
    @staticmethod
    def classify(size_mm: float) -> dict:
        """
        Classify severity based on tumor size in mm.
        
        Args:
            size_mm: Tumor size in millimeters
            
        Returns:
            dict with severity classification and details
        """
        for severity, (lower, upper) in TumorSeverityClassifier.SEVERITY_THRESHOLDS.items():
            if lower <= size_mm < upper:
                return {
                    'severity': severity,
                    'size_mm': size_mm,
                    'range': f"{lower}-{upper}mm",
                    'clinical_stage': severity,
                    'description': TumorSeverityClassifier._get_description(severity)
                }
        
        # Default to T4 if size exceeds all thresholds
        return {
            'severity': 'T4',
            'size_mm': size_mm,
            'range': '50+mm',
            'clinical_stage': 'T4',
            'description': 'Invades adjacent structures'
        }
    
    @staticmethod
    def _get_description(severity: str) -> str:
        descriptions = {
            'T1': 'Clinically insignificant - confined to prostate',
            'T2': 'Localized to prostate - confined within capsule',
            'T3': 'Extends beyond prostate - through capsule',
            'T4': 'Invades adjacent structures - bladder/rectum',
        }
        return descriptions.get(severity, 'Unknown')


def load_multi_sequence_dicom(patient_dir: str, sequences: list = ['T2', 'ADC', 'DWI']) -> dict:
    """
    Load multiple DICOM sequences for a patient.
    
    Args:
        patient_dir: Path to patient directory
        sequences: List of sequence types to load
        
    Returns:
        dict with loaded sequences and metadata
    """
    patient_path = Path(patient_dir)
    loaded_sequences = {}
    
    print(f"\n📁 Loading sequences from: {patient_dir}")
    
    for sequence in sequences:
        # Look for DICOM files matching sequence
        dicom_files = list(patient_path.glob(f"*{sequence}*.dcm"))
        
        if not dicom_files:
            print(f"  ⚠️  {sequence}: No DICOM files found")
            continue
        
        try:
            # Load first matching DICOM
            img = load_dicom_image(str(dicom_files[0]))
            loaded_sequences[sequence] = img
            print(f"  ✅ {sequence}: Loaded {img.shape}")
        except Exception as e:
            print(f"  ❌ {sequence}: Error loading - {e}")
    
    return loaded_sequences


def predict_tumor_properties(t2_img: np.ndarray, adc_img: np.ndarray = None, 
                            dwi_img: np.ndarray = None, 
                            size_predictor: TumorSizePredictor = None) -> dict:
    """
    Predict tumor size using multi-sequence images.
    
    Args:
        t2_img: T2-weighted image
        adc_img: ADC (Apparent Diffusion Coefficient) image
        dwi_img: DWI (Diffusion Weighted Imaging) image
        size_predictor: Pre-loaded model or None to create new
        
    Returns:
        dict with predictions
    """
    
    if size_predictor is None:
        print("\n🤖 Loading Size Predictor Model...")
        size_predictor = TumorSizePredictor()
    
    # Normalize images
    t2_norm = normalize_image(t2_img)
    adc_norm = normalize_image(adc_img) if adc_img is not None else None
    dwi_norm = normalize_image(dwi_img) if dwi_img is not None else None
    
    print("🔮 Predicting tumor size...")
    
    # Predict size (model handles variable inputs)
    size_prediction = size_predictor.predict(
        t2_image=t2_norm,
        adc_image=adc_norm,
        dwi_image=dwi_norm
    )
    
    return size_prediction


def generate_bbox_with_visualization(t2_img: np.ndarray, predicted_size: float,
                                     bbox_generator: BoundingBoxGenerator = None) -> dict:
    """
    Generate bounding box based on predicted tumor size.
    
    Args:
        t2_img: T2-weighted image to draw bbox on
        predicted_size: Predicted tumor size in mm
        bbox_generator: Pre-loaded generator or None to create new
        
    Returns:
        dict with bbox coordinates and visualization
    """
    
    if bbox_generator is None:
        print("📦 Loading Bounding Box Generator...")
        bbox_generator = BoundingBoxGenerator()
    
    print(f"📐 Generating bounding box for size: {predicted_size:.1f}mm...")
    
    # Generate bbox
    bbox_result = bbox_generator.generate_bbox(
        image=t2_img,
        tumor_size_mm=predicted_size
    )
    
    # Create visualization
    img_viz = t2_img.copy()
    if len(img_viz.shape) == 2:
        img_viz = cv2.cvtColor((img_viz * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    x1, y1, x2, y2 = bbox_result['bbox_coords']
    
    # Draw bounding box
    cv2.rectangle(img_viz, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Add text annotations
    cv2.putText(img_viz, f"Size: {predicted_size:.1f}mm", (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    bbox_result['visualization'] = img_viz
    
    return bbox_result


def run_complete_analysis(patient_dir: str) -> dict:
    """
    Run complete tumor analysis pipeline.
    
    Args:
        patient_dir: Path to patient DICOM directory
        
    Returns:
        dict with complete analysis results
    """
    
    print("\n" + "="*70)
    print("🏥 TUMOR SIZE PREDICTION & SEVERITY CLASSIFICATION")
    print("="*70)
    
    # Step 1: Load sequences
    sequences = load_multi_sequence_dicom(patient_dir)
    
    if 'T2' not in sequences:
        print("❌ T2 sequence required for analysis")
        return None
    
    # Step 2: Initialize models
    print("\n⚙️  Initializing models...")
    size_predictor = TumorSizePredictor()
    bbox_generator = BoundingBoxGenerator()
    
    # Step 3: Predict tumor size
    print("\n🔬 STEP 1: Tumor Size Prediction")
    print("-" * 70)
    size_pred = predict_tumor_properties(
        t2_img=sequences['T2'],
        adc_img=sequences.get('ADC'),
        dwi_img=sequences.get('DWI'),
        size_predictor=size_predictor
    )
    
    predicted_size = size_pred['predicted_size']
    print(f"✅ Predicted Size: {predicted_size:.2f}mm")
    print(f"   Confidence: {size_pred.get('confidence', 'N/A')}")
    
    # Step 4: Generate bounding box
    print("\n📦 STEP 2: Bounding Box Detection")
    print("-" * 70)
    bbox_result = generate_bbox_with_visualization(
        t2_img=sequences['T2'],
        predicted_size=predicted_size,
        bbox_generator=bbox_generator
    )
    
    print(f"✅ Bounding Box: {bbox_result['bbox_coords']}")
    print(f"   Area: {bbox_result['area_pixels']} pixels")
    
    # Step 5: Classify severity
    print("\n⚕️  STEP 3: Severity Classification")
    print("-" * 70)
    severity = TumorSeverityClassifier.classify(predicted_size)
    
    print(f"✅ Clinical Stage: {severity['severity']}")
    print(f"   Size Range: {severity['range']}")
    print(f"   Description: {severity['description']}")
    
    # Compile results
    results = {
        'patient_dir': patient_dir,
        'sequences_loaded': list(sequences.keys()),
        'tumor_size_mm': predicted_size,
        'tumor_size_confidence': size_pred.get('confidence'),
        'bounding_box': bbox_result['bbox_coords'],
        'bbox_area_pixels': bbox_result['area_pixels'],
        'severity_stage': severity['severity'],
        'severity_range_mm': severity['range'],
        'clinical_description': severity['description'],
        'visualization': bbox_result['visualization']
    }
    
    print("\n" + "="*70)
    print("📊 ANALYSIS COMPLETE")
    print("="*70)
    
    return results


def save_visualization(visualization: np.ndarray, output_path: str):
    """Save visualization to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if len(visualization.shape) == 2:
        cv2.imwrite(output_path, (visualization * 255).astype(np.uint8))
    else:
        cv2.imwrite(output_path, visualization)
    
    print(f"💾 Visualization saved to: {output_path}")


def main():
    """Main execution."""
    
    # Look for patient directories in data/PROSTATEx
    base_dir = Path(__file__).parent.parent / 'data' / 'PROSTATEx'
    
    if not base_dir.exists():
        print(f"❌ Patient data directory not found: {base_dir}")
        print("\nUsage: python demo_tumor_complete.py [patient_dir]")
        return
    
    # Get list of patient directories
    patient_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    
    if not patient_dirs:
        print(f"❌ No patient directories found in: {base_dir}")
        return
    
    # Process first patient as demo
    patient_dir = patient_dirs[0]
    print(f"\n🔍 Using sample patient: {patient_dir.name}")
    
    # Run analysis
    results = run_complete_analysis(str(patient_dir))
    
    if results:
        # Save visualization
        output_dir = Path(__file__).parent.parent / 'outputs'
        output_path = output_dir / f"{patient_dir.name}_tumor_analysis.png"
        
        if results['visualization'] is not None:
            save_visualization(results['visualization'], str(output_path))
        
        # Print summary
        print("\n📋 ANALYSIS SUMMARY")
        print(f"  Tumor Size: {results['tumor_size_mm']:.2f}mm")
        print(f"  Clinical Stage: {results['severity_stage']}")
        print(f"  Bounding Box: {results['bounding_box']}")


if __name__ == '__main__':
    main()
