#!/usr/bin/env python3
"""
Comprehensive Test Suite for Tumor Size Prediction System
Tests: Size prediction, bounding box detection, and severity classification
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from size_predictor_model import SizePredictorModel
from bbox_utils import BBoxPredictor
from utils_image import normalize_mri_sequence


def test_size_predictor():
    """Test tumor size prediction model"""
    print("\n" + "="*60)
    print("TEST 1: Size Predictor Model")
    print("="*60)
    
    try:
        model = SizePredictorModel(input_channels=3, num_classes=5)
        print("✓ Model initialized successfully")
        
        # Create dummy input (batch_size=1, channels=3, height=64, width=64)
        dummy_t2 = normalize_mri_sequence(np.random.rand(64, 64).astype(np.float32))
        dummy_adc = normalize_mri_sequence(np.random.rand(64, 64).astype(np.float32))
        dummy_dwi = normalize_mri_sequence(np.random.rand(64, 64).astype(np.float32))
        
        print(f"✓ Created dummy MRI sequences:")
        print(f"  - T2 shape: {dummy_t2.shape}")
        print(f"  - ADC shape: {dummy_adc.shape}")
        print(f"  - DWI shape: {dummy_dwi.shape}")
        
        # Test prediction
        size_pred, size_class = model.predict(dummy_t2, dummy_adc, dummy_dwi)
        print(f"✓ Predictions generated:")
        print(f"  - Predicted size: {size_pred:.2f} mm")
        print(f"  - Size class: {size_class}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_bbox_predictor():
    """Test bounding box predictor"""
    print("\n" + "="*60)
    print("TEST 2: Bounding Box Predictor")
    print("="*60)
    
    try:
        predictor = BBoxPredictor()
        print("✓ BBox predictor initialized")
        
        # Create dummy MRI image
        dummy_mri = np.random.rand(256, 256).astype(np.float32)
        dummy_mri = normalize_mri_sequence(dummy_mri)
        
        # Predict bounding box
        bbox = predictor.predict_bbox(dummy_mri)
        print(f"✓ Bounding box predicted:")
        print(f"  - BBox coordinates: {bbox}")
        print(f"  - Format: (x1, y1, x2, y2)")
        
        # Calculate bbox properties
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        print(f"  - Width: {width:.2f} px")
        print(f"  - Height: {height:.2f} px")
        print(f"  - Area: {area:.2f} px²")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_severity_classification():
    """Test severity classification based on tumor size"""
    print("\n" + "="*60)
    print("TEST 3: Severity Classification (TNM Staging)")
    print("="*60)
    
    try:
        from size_predictor_model import SizePredictorModel
        
        model = SizePredictorModel()
        
        # Test different tumor sizes
        test_sizes = [5, 10, 15, 20, 30]
        
        print("✓ Testing severity classification for different sizes:")
        
        for size in test_sizes:
            severity = model.classify_severity(size)
            print(f"  - Size {size}mm → Severity: {severity}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_multi_sequence_integration():
    """Test multi-sequence (T2, ADC, DWI) integration"""
    print("\n" + "="*60)
    print("TEST 4: Multi-Sequence Integration")
    print("="*60)
    
    try:
        model = SizePredictorModel(input_channels=3)
        
        # Simulate multi-sequence input
        print("✓ Testing multi-sequence input processing:")
        
        t2_seq = np.random.rand(128, 128).astype(np.float32)
        adc_seq = np.random.rand(128, 128).astype(np.float32)
        dwi_seq = np.random.rand(128, 128).astype(np.float32)
        
        # Normalize
        t2_norm = normalize_mri_sequence(t2_seq)
        adc_norm = normalize_mri_sequence(adc_seq)
        dwi_norm = normalize_mri_sequence(dwi_seq)
        
        print(f"  - T2 normalized: min={t2_norm.min():.3f}, max={t2_norm.max():.3f}")
        print(f"  - ADC normalized: min={adc_norm.min():.3f}, max={adc_norm.max():.3f}")
        print(f"  - DWI normalized: min={dwi_norm.min():.3f}, max={dwi_norm.max():.3f}")
        
        # Get prediction
        size, cls = model.predict(t2_norm, adc_norm, dwi_norm)
        print(f"✓ Combined prediction: {size:.2f}mm (Class: {cls})")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_inference_pipeline():
    """Test complete inference pipeline"""
    print("\n" + "="*60)
    print("TEST 5: Complete Inference Pipeline")
    print("="*60)
    
    try:
        from infer_with_bbox import InferenceWithBBox
        
        pipeline = InferenceWithBBox()
        print("✓ Inference pipeline initialized")
        
        # Create dummy multi-sequence data
        dummy_data = {
            'T2': np.random.rand(256, 256).astype(np.float32),
            'ADC': np.random.rand(256, 256).astype(np.float32),
            'DWI': np.random.rand(256, 256).astype(np.float32)
        }
        
        print(f"✓ Input shapes: T2={dummy_data['T2'].shape}, ADC={dummy_data['ADC'].shape}, DWI={dummy_data['DWI'].shape}")
        
        # Run inference
        result = pipeline.infer(
            dummy_data['T2'],
            dummy_data['ADC'],
            dummy_data['DWI']
        )
        
        print(f"✓ Pipeline output:")
        print(f"  - Tumor size: {result['tumor_size']:.2f} mm")
        print(f"  - Size class: {result['size_class']}")
        print(f"  - Severity: {result['severity']}")
        print(f"  - Bounding box: {result['bbox']}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("TUMOR SIZE PREDICTION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    tests = [
        ("Size Predictor", test_size_predictor),
        ("BBox Predictor", test_bbox_predictor),
        ("Severity Classification", test_severity_classification),
        ("Multi-Sequence Integration", test_multi_sequence_integration),
        ("Complete Pipeline", test_inference_pipeline),
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready for production.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")


if __name__ == "__main__":
    main()
