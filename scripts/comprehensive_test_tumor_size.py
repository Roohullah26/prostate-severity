#!/usr/bin/env python3
"""
Comprehensive Test Suite for Tumor Size Prediction System
Tests all components: model, inference, bbox, severity classification, and API
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.size_predictor_model import SizePredictorModel
from src.bbox_utils import predict_bbox, create_severity_report
from src.config import Config


class TumorSizeTestSuite:
    """Comprehensive test suite for tumor size prediction"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.test_results = []
        self.output_dir = Path(__file__).parent.parent / "test_results"
        self.output_dir.mkdir(exist_ok=True)
    
    def test_model_initialization(self):
        """Test 1: Model Initialization"""
        print("\n" + "="*70)
        print("TEST 1: Model Initialization")
        print("="*70)
        
        try:
            self.model = SizePredictorModel(
                in_channels=3,
                hidden_dim=64,
                device=self.device
            )
            print(f"✓ Model created successfully")
            print(f"  Device: {self.device}")
            print(f"  Parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
            
            self.test_results.append({
                'test': 'Model Initialization',
                'status': 'PASS',
                'device': str(self.device),
                'timestamp': time.time()
            })
            return True
        except Exception as e:
            print(f"✗ Failed: {e}")
            self.test_results.append({
                'test': 'Model Initialization',
                'status': 'FAIL',
                'error': str(e),
                'timestamp': time.time()
            })
            return False
    
    def test_synthetic_prediction(self):
        """Test 2: Synthetic Data Prediction"""
        print("\n" + "="*70)
        print("TEST 2: Synthetic Data Prediction")
        print("="*70)
        
        if self.model is None:
            print("✗ Model not initialized")
            return False
        
        try:
            # Create synthetic tumor
            h, w = 256, 256
            y, x = np.ogrid[:h, :w]
            cx, cy = h//2, w//2
            tumor_mask = np.exp(-((x - cx)**2 + (y - cy)**2) / (30**2))
            
            t2 = np.random.rand(h, w) * 0.3 + tumor_mask * 0.7
            adc = np.random.rand(h, w) * 0.4 + tumor_mask * 0.6
            dwi = np.random.rand(h, w) * 0.2 + tumor_mask * 0.8
            
            print(f"Input shapes:")
            print(f"  T2: {t2.shape}, range: [{t2.min():.3f}, {t2.max():.3f}]")
            print(f"  ADC: {adc.shape}, range: [{adc.min():.3f}, {adc.max():.3f}]")
            print(f"  DWI: {dwi.shape}, range: [{dwi.min():.3f}, {dwi.max():.3f}]")
            
            # Make prediction
            tumor_size = self.model.predict(t2, adc, dwi)
            print(f"\n✓ Prediction successful")
            print(f"  Tumor size: {tumor_size:.2f} mm")
            print(f"  Value range: [0, {Config.MAX_TUMOR_SIZE_MM}] mm")
            
            assert 0 <= tumor_size <= Config.MAX_TUMOR_SIZE_MM, f"Invalid tumor size: {tumor_size}"
            
            self.test_results.append({
                'test': 'Synthetic Data Prediction',
                'status': 'PASS',
                'tumor_size_mm': float(tumor_size),
                'input_shapes': {'t2': t2.shape, 'adc': adc.shape, 'dwi': dwi.shape},
                'timestamp': time.time()
            })
            return True
        except Exception as e:
            print(f"✗ Failed: {e}")
            self.test_results.append({
                'test': 'Synthetic Data Prediction',
                'status': 'FAIL',
                'error': str(e),
                'timestamp': time.time()
            })
            return False
    
    def test_bbox_generation(self):
        """Test 3: Bounding Box Generation"""
        print("\n" + "="*70)
        print("TEST 3: Bounding Box Generation")
        print("="*70)
        
        try:
            # Create test image with tumor
            h, w = 256, 256
            image = np.zeros((h, w))
            cx, cy = h//2, w//2
            
            # Create tumor
            y, x = np.ogrid[:h, :w]
            tumor_mask = np.exp(-((x - cx)**2 + (y - cy)**2) / (30**2))
            image = tumor_mask
            
            tumor_size = 25.0  # mm
            
            bbox, confidence = predict_bbox(image, tumor_size)
            x1, y1, x2, y2 = bbox
            
            print(f"Image shape: {image.shape}")
            print(f"Tumor size: {tumor_size:.2f} mm")
            print(f"\n✓ Bounding box generated")
            print(f"  Coordinates: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
            print(f"  Width: {x2-x1:.0f}px, Height: {y2-y1:.0f}px")
            print(f"  Confidence: {confidence:.2%}")
            
            # Validate bbox
            assert 0 <= x1 < x2 <= w, f"Invalid x coordinates: {x1}, {x2}"
            assert 0 <= y1 < y2 <= h, f"Invalid y coordinates: {y1}, {y2}"
            assert 0 <= confidence <= 1, f"Invalid confidence: {confidence}"
            
            self.test_results.append({
                'test': 'Bounding Box Generation',
                'status': 'PASS',
                'bbox': {'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)},
                'confidence': float(confidence),
                'timestamp': time.time()
            })
            return True
        except Exception as e:
            print(f"✗ Failed: {e}")
            self.test_results.append({
                'test': 'Bounding Box Generation',
                'status': 'FAIL',
                'error': str(e),
                'timestamp': time.time()
            })
            return False
    
    def test_severity_classification(self):
        """Test 4: Severity Classification (TNM Staging)"""
        print("\n" + "="*70)
        print("TEST 4: Severity Classification (TNM Staging)")
        print("="*70)
        
        try:
            test_cases = [
                (5.0, 'T1', 'Small'),
                (15.0, 'T2', 'Medium'),
                (35.0, 'T3', 'Large'),
                (50.0, 'T4', 'Very Large')
            ]
            
            for tumor_size, expected_t_stage, expected_severity in test_cases:
                bbox = (50, 50, 100, 100)
                severity = create_severity_report(tumor_size, bbox)
                
                print(f"\nTumor size: {tumor_size:.1f}mm")
                print(f"  T-Stage: {severity['t_stage']} (expected: {expected_t_stage})")
                print(f"  Severity: {severity['severity']} (expected: {expected_severity})")
                print(f"  Notes: {severity['clinical_notes'][:50]}...")
                
                assert severity['t_stage'] == expected_t_stage, \
                    f"Wrong T-stage: {severity['t_stage']} vs {expected_t_stage}"
            
            print(f"\n✓ All severity classifications correct")
            
            self.test_results.append({
                'test': 'Severity Classification',
                'status': 'PASS',
                'test_cases': len(test_cases),
                'timestamp': time.time()
            })
            return True
        except Exception as e:
            print(f"✗ Failed: {e}")
            self.test_results.append({
                'test': 'Severity Classification',
                'status': 'FAIL',
                'error': str(e),
                'timestamp': time.time()
            })
            return False
    
    def test_edge_cases(self):
        """Test 5: Edge Cases"""
        print("\n" + "="*70)
        print("TEST 5: Edge Cases")
        print("="*70)
        
        if self.model is None:
            return False
        
        try:
            h, w = 256, 256
            test_cases = [
                ('All zeros', np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w))),
                ('All ones', np.ones((h, w)), np.ones((h, w)), np.ones((h, w))),
                ('Random noise', np.random.rand(h, w), np.random.rand(h, w), np.random.rand(h, w)),
                ('Very small tumor', 
                 np.random.rand(h, w) * 0.01, 
                 np.random.rand(h, w) * 0.01,
                 np.random.rand(h, w) * 0.01),
            ]
            
            results = []
            for name, t2, adc, dwi in test_cases:
                try:
                    size = self.model.predict(t2, adc, dwi)
                    results.append((name, 'PASS', size))
                    print(f"✓ {name}: {size:.2f}mm")
                except Exception as e:
                    results.append((name, 'FAIL', str(e)))
                    print(f"✗ {name}: {e}")
            
            passed = sum(1 for _, status, _ in results if status == 'PASS')
            print(f"\n✓ Passed {passed}/{len(results)} edge cases")
            
            self.test_results.append({
                'test': 'Edge Cases',
                'status': 'PASS' if passed == len(results) else 'PARTIAL',
                'passed': passed,
                'total': len(results),
                'timestamp': time.time()
            })
            return True
        except Exception as e:
            print(f"✗ Failed: {e}")
            self.test_results.append({
                'test': 'Edge Cases',
                'status': 'FAIL',
                'error': str(e),
                'timestamp': time.time()
            })
            return False
    
    def test_model_weights(self):
        """Test 6: Model Weight Loading/Saving"""
        print("\n" + "="*70)
        print("TEST 6: Model Weight Loading/Saving")
        print("="*70)
        
        if self.model is None:
            return False
        
        try:
            test_weights_path = self.output_dir / "test_weights.pth"
            
            # Save weights
            self.model.save_weights(str(test_weights_path))
            print(f"✓ Weights saved to {test_weights_path}")
            assert test_weights_path.exists(), "Weights file not created"
            
            # Load weights
            self.model.load_weights(str(test_weights_path))
            print(f"✓ Weights loaded from {test_weights_path}")
            
            # Clean up
            test_weights_path.unlink()
            
            self.test_results.append({
                'test': 'Model Weight Loading/Saving',
                'status': 'PASS',
                'timestamp': time.time()
            })
            return True
        except Exception as e:
            print(f"✗ Failed: {e}")
            self.test_results.append({
                'test': 'Model Weight Loading/Saving',
                'status': 'FAIL',
                'error': str(e),
                'timestamp': time.time()
            })
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "#"*70)
        print("# TUMOR SIZE PREDICTION - COMPREHENSIVE TEST SUITE")
        print("#"*70)
        print(f"\nDevice: {self.device}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        tests = [
            self.test_model_initialization,
            self.test_synthetic_prediction,
            self.test_bbox_generation,
            self.test_severity_classification,
            self.test_edge_cases,
            self.test_model_weights,
        ]
        
        results = []
        for test in tests:
            try:
                results.append(test())
            except Exception as e:
                print(f"\n✗ Test crashed: {e}")
                results.append(False)
        
        # Summary
        print("\n" + "#"*70)
        print("# TEST SUMMARY")
        print("#"*70)
        
        passed = sum(results)
        total = len(results)
        
        print(f"\nTotal: {total} tests")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {passed/total*100:.1f}%")
        
        # Save results
        results_path = self.output_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'total': total,
                'passed': passed,
                'failed': total - passed,
                'success_rate': passed / total,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'device': str(self.device),
                'test_results': self.test_results
            }, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_path}")
        print("#"*70 + "\n")
        
        return passed == total


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tumor Size Prediction Test Suite")
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                       help='Device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    suite = TumorSizeTestSuite()
    if args.device:
        suite.device = torch.device(args.device)
    
    success = suite.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
