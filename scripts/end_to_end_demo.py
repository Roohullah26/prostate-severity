"""
End-to-End Tumor Size Prediction Demo
Shows complete workflow from T2/ADC/DWI to predictions with severity
"""
import numpy as np
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from size_predictor_model import SizePredictorModel
from bbox_utils import BoundingBoxPredictor
from infer_with_bbox import TumorSizeInferenceEngine


def create_synthetic_mri_data():
    """Create synthetic MRI data for demo"""
    print("\n" + "=" * 70)
    print("CREATING SYNTHETIC MRI DATA")
    print("=" * 70)
    
    # Create realistic synthetic MRI images
    size = 256
    
    # T2-weighted image (high signal in tumor, medium in normal prostate)
    t2_image = np.ones((size, size)) * 100
    # Add tumor region (bright in T2)
    cy, cx = 128, 128
    r = 40
    y, x = np.ogrid[:size, :size]
    tumor_mask = (x - cx)**2 + (y - cy)**2 <= r**2
    t2_image[tumor_mask] = 200
    # Add noise
    t2_image += np.random.normal(0, 10, t2_image.shape)
    t2_image = np.clip(t2_image, 0, 255)
    
    # ADC map (low ADC in tumor, high in normal)
    adc_image = np.ones((size, size)) * 1200
    adc_image[tumor_mask] = 600
    adc_image += np.random.normal(0, 50, adc_image.shape)
    adc_image = np.clip(adc_image, 0, 2000)
    
    # DWI image (high signal in tumor, low in normal)
    dwi_image = np.ones((size, size)) * 80
    dwi_image[tumor_mask] = 180
    dwi_image += np.random.normal(0, 10, dwi_image.shape)
    dwi_image = np.clip(dwi_image, 0, 255)
    
    print(f"✓ T2 image shape: {t2_image.shape}, range: [{t2_image.min():.1f}, {t2_image.max():.1f}]")
    print(f"✓ ADC image shape: {adc_image.shape}, range: [{adc_image.min():.1f}, {adc_image.max():.1f}]")
    print(f"✓ DWI image shape: {dwi_image.shape}, range: [{dwi_image.min():.1f}, {dwi_image.max():.1f}]")
    print(f"✓ Synthetic tumor radius: ~{r} pixels")
    
    return t2_image.astype(np.float32), adc_image.astype(np.float32), dwi_image.astype(np.float32), tumor_mask


def normalize_images(t2, adc, dwi):
    """Normalize images to 0-1 range"""
    print("\n" + "=" * 70)
    print("NORMALIZING IMAGES")
    print("=" * 70)
    
    t2_norm = (t2 - t2.min()) / (t2.max() - t2.min() + 1e-8)
    adc_norm = (adc - adc.min()) / (adc.max() - adc.min() + 1e-8)
    dwi_norm = (dwi - dwi.min()) / (dwi.max() - dwi.min() + 1e-8)
    
    print(f"✓ T2 normalized to [{t2_norm.min():.3f}, {t2_norm.max():.3f}]")
    print(f"✓ ADC normalized to [{adc_norm.min():.3f}, {adc_norm.max():.3f}]")
    print(f"✓ DWI normalized to [{dwi_norm.min():.3f}, {dwi_norm.max():.3f}]")
    
    return t2_norm, adc_norm, dwi_norm


def extract_tumor_features(t2, adc, dwi, tumor_mask):
    """Extract tumor characteristics"""
    print("\n" + "=" * 70)
    print("EXTRACTING TUMOR FEATURES")
    print("=" * 70)
    
    # Calculate statistics
    features = {
        "t2_mean": float(t2[tumor_mask].mean()),
        "t2_std": float(t2[tumor_mask].std()),
        "adc_mean": float(adc[tumor_mask].mean()),
        "adc_std": float(adc[tumor_mask].std()),
        "dwi_mean": float(dwi[tumor_mask].mean()),
        "dwi_std": float(dwi[tumor_mask].std()),
        "tumor_pixels": int(tumor_mask.sum()),
        "tumor_area_mm2": float(tumor_mask.sum() * 0.64)  # Assuming 0.8mm resolution
    }
    
    print(f"✓ T2 signal: mean={features['t2_mean']:.1f}±{features['t2_std']:.1f}")
    print(f"✓ ADC signal: mean={features['adc_mean']:.1f}±{features['adc_std']:.1f}")
    print(f"✓ DWI signal: mean={features['dwi_mean']:.1f}±{features['dwi_std']:.1f}")
    print(f"✓ Tumor area: {features['tumor_area_mm2']:.1f} mm²")
    
    return features


def estimate_tumor_size(features):
    """Estimate tumor size from features"""
    print("\n" + "=" * 70)
    print("ESTIMATING TUMOR SIZE")
    print("=" * 70)
    
    # Simple size estimation (in real model, this is learned)
    area_mm2 = features['tumor_area_mm2']
    diameter_mm = 2 * np.sqrt(area_mm2 / np.pi)
    
    print(f"✓ Estimated diameter: {diameter_mm:.2f} mm")
    print(f"✓ Estimated volume: {(4/3 * np.pi * (diameter_mm/2)**3):.2f} mm³")
    
    return diameter_mm


def classify_severity(size_mm):
    """Classify TNM severity based on tumor size"""
    print("\n" + "=" * 70)
    print("CLASSIFYING SEVERITY (TNM)")
    print("=" * 70)
    
    # TNM Classification for prostate cancer
    if size_mm <= 5:
        severity = "T1a"
        description = "Microscopic, found in <5% of tissue"
    elif size_mm <= 10:
        severity = "T1b"
        description = "Microscopic, found in >5% of tissue"
    elif size_mm <= 20:
        severity = "T2a"
        description = "≤1/2 of prostate involved"
    elif size_mm <= 35:
        severity = "T2b"
        description = ">1/2 of prostate involved"
    elif size_mm <= 50:
        severity = "T2c"
        description = "Bilateral involvement"
    elif size_mm <= 70:
        severity = "T3a"
        description = "Extraprostatic extension"
    elif size_mm <= 100:
        severity = "T3b"
        description = "Seminal vesicle invasion"
    else:
        severity = "T4"
        description = "Invasion of adjacent structures"
    
    confidence = min(0.95, 0.5 + (1 - abs(size_mm - 25) / 100))  # Higher confidence for mid-range sizes
    
    print(f"✓ Severity Classification: {severity}")
    print(f"✓ Description: {description}")
    print(f"✓ Confidence: {confidence:.1%}")
    
    return severity, description, confidence


def detect_bounding_box(tumor_mask):
    """Detect bounding box from tumor mask"""
    print("\n" + "=" * 70)
    print("DETECTING BOUNDING BOX")
    print("=" * 70)
    
    rows = np.any(tumor_mask, axis=1)
    cols = np.any(tumor_mask, axis=0)
    
    if rows.any() and cols.any():
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        x = x_min
        y = y_min
        w = x_max - x_min
        h = y_max - y_min
        
        bbox = [x, y, w, h]
        print(f"✓ Bounding box detected:")
        print(f"  Position: x={x}, y={y}")
        print(f"  Size: width={w}, height={h}")
        
        return bbox
    else:
        print("✗ Could not detect bounding box")
        return None


def generate_prediction_report(
    patient_id,
    t2, adc, dwi,
    features,
    size_mm,
    severity,
    description,
    confidence,
    bbox
):
    """Generate comprehensive prediction report"""
    print("\n" + "=" * 70)
    print("FINAL PREDICTION REPORT")
    print("=" * 70)
    
    report = {
        "patient_id": patient_id,
        "timestamp": str(np.datetime64('now')),
        "tumor_size_mm": round(size_mm, 2),
        "severity_tnm": severity,
        "severity_description": description,
        "confidence": round(confidence, 3),
        "bounding_box": {
            "x": int(bbox[0]) if bbox else None,
            "y": int(bbox[1]) if bbox else None,
            "width": int(bbox[2]) if bbox else None,
            "height": int(bbox[3]) if bbox else None
        },
        "mri_features": {
            "t2_signal": {
                "mean": round(features['t2_mean'], 1),
                "std": round(features['t2_std'], 1)
            },
            "adc_signal": {
                "mean": round(features['adc_mean'], 1),
                "std": round(features['adc_std'], 1)
            },
            "dwi_signal": {
                "mean": round(features['dwi_mean'], 1),
                "std": round(features['dwi_std'], 1)
            }
        },
        "recommendations": generate_recommendations(severity)
    }
    
    print("\nPREDICTION RESULTS:")
    print(f"  Patient ID: {report['patient_id']}")
    print(f"  Tumor Size: {report['tumor_size_mm']} mm")
    print(f"  TNM Stage: {report['severity_tnm']}")
    print(f"  Confidence: {report['confidence']:.1%}")
    print(f"  Bounding Box: {report['bounding_box']}")
    
    print("\nRECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    return report


def generate_recommendations(severity):
    """Generate clinical recommendations based on severity"""
    recommendations = {
        "T1a": [
            "Active surveillance recommended",
            "PSA monitoring every 6-12 months",
            "Consider repeat biopsy if PSA rises"
        ],
        "T1b": [
            "Active surveillance or radiation therapy",
            "Hormone therapy may be considered",
            "Regular follow-up imaging"
        ],
        "T2a": [
            "Radical prostatectomy or radiation therapy",
            "Consider hormone therapy",
            "MRI follow-up at 3-6 months"
        ],
        "T2b": [
            "Multimodal therapy recommended",
            "External beam radiation + hormone therapy",
            "Consider MRI-guided biopsy"
        ],
        "T2c": [
            "Aggressive treatment recommended",
            "Combined radiation and hormone therapy",
            "Regular imaging and PSA monitoring"
        ],
        "T3a": [
            "Multimodal therapy with external beam radiation",
            "Long-term hormone therapy",
            "Close imaging follow-up"
        ],
        "T3b": [
            "Intensive hormone therapy with radiation",
            "Consider additional chemotherapy",
            "Monthly monitoring"
        ],
        "T4": [
            "Palliative care and hormone therapy",
            "Chemotherapy consideration",
            "Multidisciplinary team involvement"
        ]
    }
    
    return recommendations.get(severity, ["Consult with oncologist for personalized plan"])


def main():
    """Run complete end-to-end demo"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "TUMOR SIZE PREDICTION - END-TO-END DEMO" + " " * 15 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Step 1: Create synthetic data
    t2, adc, dwi, tumor_mask = create_synthetic_mri_data()
    
    # Step 2: Normalize
    t2_norm, adc_norm, dwi_norm = normalize_images(t2, adc, dwi)
    
    # Step 3: Extract features
    features = extract_tumor_features(t2, adc, dwi, tumor_mask)
    
    # Step 4: Estimate size
    size_mm = estimate_tumor_size(features)
    
    # Step 5: Classify severity
    severity, description, confidence = classify_severity(size_mm)
    
    # Step 6: Detect bounding box
    bbox = detect_bounding_box(tumor_mask)
    
    # Step 7: Generate report
    report = generate_prediction_report(
        patient_id="DEMO_001",
        t2=t2, adc=adc, dwi=dwi,
        features=features,
        size_mm=size_mm,
        severity=severity,
        description=description,
        confidence=confidence,
        bbox=bbox
    )
    
    # Step 8: Display final result
    print("\n" + "=" * 70)
    print("COMPLETE PREDICTION SUMMARY")
    print("=" * 70)
    print(json.dumps(report, indent=2))
    
    print("\n" + "=" * 70)
    print("✓ END-TO-END DEMO COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    return report


if __name__ == "__main__":
    report = main()
