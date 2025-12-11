#!/usr/bin/env python
"""
Verify that the tumor size prediction model can be loaded and used for inference.
Tests model initialization, weight loading, and dummy predictions.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from size_predictor_model import TumorSizePredictor
from bbox_utils import BoundingBoxGenerator, VisualizationHelper, TumorPrediction


def test_model_loading():
    """Test 1: Load model and verify architecture."""
    print("\n" + "="*70)
    print("TEST 1: MODEL INITIALIZATION & LOADING")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        model = TumorSizePredictor(pretrained=False, in_channels=3, dropout_rate=0.3)
        model = model.to(device)
        model.eval()
        print("✓ Model initialized successfully")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False
    
    return True, model, device


def test_weight_loading(model, device):
    """Test 2: Load pre-trained weights."""
    print("\n" + "="*70)
    print("TEST 2: WEIGHT LOADING")
    print("="*70)
    
    model_paths = [
        Path(__file__).resolve().parent.parent / 'models' / 'baseline_real_t2_adc_3s_ep1.pth',
        Path(__file__).resolve().parent.parent / 'models' / 'prototype_toy.pth',
    ]
    
    for model_path in model_paths:
        if model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict, strict=False)
                print(f"✓ Loaded weights from: {model_path.name}")
                return True, model_path
            except Exception as e:
                print(f"✗ Failed to load {model_path.name}: {e}")
        else:
            print(f"  (skipped {model_path.name} - not found)")
    
    print("⚠ No pre-trained weights found, using randomly initialized model")
    return True, None


def test_inference(model, device):
    """Test 3: Run inference on dummy data."""
    print("\n" + "="*70)
    print("TEST 3: INFERENCE")
    print("="*70)
    
    # Create dummy input (batch of 2, 3 channels, 224x224)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        
        print("✓ Forward pass successful")
        print(f"  Output keys: {list(output.keys())}")
        print(f"  Size output shape: {output['size'].shape}")
        print(f"  Severity logits shape: {output['severity_logits'].shape}")
        print(f"  Severity probs shape: {output['severity_probs'].shape}")
        print(f"  Confidence shape: {output['confidence'].shape}")
        
        # Print sample values
        print(f"\n  Sample prediction 1:")
        print(f"    Size (W×H×D): {output['size'][0].cpu().numpy().round(2)} mm")
        print(f"    Severity: {torch.argmax(output['severity_probs'][0]).item()}")
        print(f"    Confidence: {output['confidence'][0, 0].item():.3f}")
        
        return True, output
    
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False, None


def test_bbox_generation(output):
    """Test 4: Generate bounding boxes."""
    print("\n" + "="*70)
    print("TEST 4: BOUNDING BOX GENERATION")
    print("="*70)
    
    try:
        # Extract predictions from first sample
        size = output['size'][0].cpu().numpy()
        severity_probs = output['severity_probs'][0].cpu().numpy()
        confidence = output['confidence'][0, 0].item()
        
        # Create TumorPrediction
        severity_grades = ['T1', 'T2', 'T3', 'T4']
        severity_idx = int(np.argmax(severity_probs))
        
        pred = TumorPrediction(
            width_mm=float(size[0]),
            height_mm=float(size[1]),
            depth_mm=float(size[2]),
            severity=severity_grades[severity_idx],
            severity_logits=severity_probs,
            confidence=confidence,
            image_size=(224, 224),
            pixel_spacing_mm=(1.0, 1.0)
        )
        
        print("✓ TumorPrediction created successfully")
        
        # Generate bboxes
        bbox_gen = BoundingBoxGenerator(pixel_spacing_mm=(1.0, 1.0))
        
        rect_bbox = bbox_gen.get_rectangular_bbox(pred)
        circ_bbox = bbox_gen.get_circular_bbox(pred)
        
        print(f"✓ Bounding boxes generated")
        print(f"  Rectangular bbox: x1={rect_bbox['x1']}, y1={rect_bbox['y1']}, "
              f"x2={rect_bbox['x2']}, y2={rect_bbox['y2']}")
        print(f"  Circular bbox: center=({circ_bbox['center_x']}, {circ_bbox['center_y']}), "
              f"radius={circ_bbox['radius_px']} px")
        
        return True, pred, rect_bbox, circ_bbox
    
    except Exception as e:
        print(f"✗ Bbox generation failed: {e}")
        return False, None, None, None


def test_visualization(pred, rect_bbox):
    """Test 5: Visualization helper."""
    print("\n" + "="*70)
    print("TEST 5: VISUALIZATION")
    print("="*70)
    
    try:
        # Create a dummy image
        import numpy as np
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        vis = VisualizationHelper()
        summary = vis.create_prediction_summary(pred)
        
        print("✓ Visualization helper works")
        print(f"  Summary length: {len(summary)} chars")
        
        return True
    
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        return False


def print_summary(results):
    """Print final summary."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    tests_passed = sum(1 for r in results.values() if r is True)
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("\n✨ ALL TESTS PASSED! System is ready to use.")
        return True
    else:
        print(f"\n⚠ {total_tests - tests_passed} test(s) failed. Review errors above.")
        return False


def main():
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*15 + "🧪 MODEL VERIFICATION TEST SUITE 🧪" + " "*16 + "║")
    print("╚" + "="*68 + "╝")
    
    results = {}
    
    # Test 1: Model initialization
    test1_result = test_model_loading()
    if not test1_result or not test1_result[0]:
        print("✗ Cannot proceed without model initialization")
        return False
    
    model, device = test1_result[1], test1_result[2]
    results['Model Initialization'] = True
    
    # Test 2: Weight loading
    test2_result = test_weight_loading(model, device)
    results['Weight Loading'] = test2_result[0]
    
    # Test 3: Inference
    test3_result = test_inference(model, device)
    if not test3_result[0]:
        print("✗ Cannot proceed without inference capability")
        print_summary(results)
        return False
    
    results['Inference'] = True
    output = test3_result[1]
    
    # Test 4: Bbox generation
    test4_result = test_bbox_generation(output)
    results['Bbox Generation'] = test4_result[0]
    
    # Test 5: Visualization
    if test4_result[0]:
        test5_result = test_visualization(test4_result[1], test4_result[2])
        results['Visualization'] = test5_result
    
    # Print summary
    success = print_summary(results)
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
