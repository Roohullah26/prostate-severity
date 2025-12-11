#!/usr/bin/env python
"""
Comprehensive Test Suite for Current Tumor Size System.
Tests: TumorSizePredictor, BoundingBoxGenerator, visualization, and API compatibility.
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from size_predictor_model import TumorSizePredictor
from bbox_utils import BoundingBoxGenerator, TumorPrediction, VisualizationHelper


class ComprehensiveTestSuite:
    """Test suite for tumor size prediction system."""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = {}
        self.test_count = 0
        self.passed_count = 0
        self.output_dir = Path(__file__).resolve().parent.parent / 'test_results'
        self.output_dir.mkdir(exist_ok=True)
    
    def log_test(self, name, passed, details=""):
        """Log a test result."""
        self.test_count += 1
        if passed:
            self.passed_count += 1
            status = "[PASS]"
        else:
            status = "[FAIL]"
        
        print(f"{status} {name}")
        if details:
            print(f"       {details}")
        
        self.results[name] = {'passed': passed, 'details': details}
    
    def test_1_model_init(self):
        """Test 1: Model initialization."""
        print("\n" + "="*70)
        print("TEST 1: MODEL INITIALIZATION")
        print("="*70)
        
        try:
            model = TumorSizePredictor(pretrained=False, in_channels=3)
            model = model.to(self.device)
            params = sum(p.numel() for p in model.parameters())
            self.log_test("Model Initialization", True, f"Params: {params:,}, Device: {self.device}")
            return model
        except Exception as e:
            self.log_test("Model Initialization", False, str(e))
            return None
    
    def test_2_weight_loading(self, model):
        """Test 2: Load pre-trained weights."""
        print("\n" + "="*70)
        print("TEST 2: WEIGHT LOADING")
        print("="*70)
        
        model_path = Path(__file__).resolve().parent.parent / 'models' / 'baseline_real_t2_adc_3s_ep1.pth'
        
        try:
            if model_path.exists():
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
                self.log_test("Weight Loading", True, f"Loaded from {model_path.name}")
            else:
                self.log_test("Weight Loading", False, f"Model file not found: {model_path}")
        except Exception as e:
            self.log_test("Weight Loading", False, str(e))
    
    def test_3_inference(self, model):
        """Test 3: Inference on dummy data."""
        print("\n" + "="*70)
        print("TEST 3: INFERENCE")
        print("="*70)
        
        try:
            model.eval()
            dummy_input = torch.randn(2, 3, 224, 224).to(self.device)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            # Check output structure
            expected_keys = {'size', 'severity_logits', 'severity_probs', 'confidence'}
            actual_keys = set(output.keys())
            
            if expected_keys == actual_keys:
                detail = f"Output keys: {actual_keys}, Shapes: size={output['size'].shape}, severity={output['severity_probs'].shape}"
                self.log_test("Inference", True, detail)
                return output
            else:
                self.log_test("Inference", False, f"Expected keys {expected_keys}, got {actual_keys}")
                return None
        except Exception as e:
            self.log_test("Inference", False, str(e))
            return None
    
    def test_4_bbox_generation(self, output):
        """Test 4: Bounding box generation."""
        print("\n" + "="*70)
        print("TEST 4: BOUNDING BOX GENERATION")
        print("="*70)
        
        try:
            size = output['size'][0].cpu().numpy()
            severity_probs = output['severity_probs'][0].cpu().numpy()
            confidence = output['confidence'][0, 0].item()
            
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
            
            gen = BoundingBoxGenerator(pixel_spacing_mm=(1.0, 1.0))
            rect_bbox = gen.get_rectangular_bbox(pred)
            circ_bbox = gen.get_circular_bbox(pred)
            
            detail = f"Rect: ({rect_bbox['x1']}, {rect_bbox['y1']}) -> ({rect_bbox['x2']}, {rect_bbox['y2']}), " \
                    f"Circle: center=({circ_bbox['center_x']}, {circ_bbox['center_y']}), r={circ_bbox['radius_px']}"
            self.log_test("Bbox Generation", True, detail)
            return pred, rect_bbox, circ_bbox
        except Exception as e:
            self.log_test("Bbox Generation", False, str(e))
            return None, None, None
    
    def test_5_severity_classification(self, pred):
        """Test 5: Severity classification logic."""
        print("\n" + "="*70)
        print("TEST 5: SEVERITY CLASSIFICATION")
        print("="*70)
        
        try:
            gen = BoundingBoxGenerator()
            
            # Test thresholds
            tests = [
                (5, 'T1'),
                (15, 'T1'),
                (25, 'T2'),
                (45, 'T3'),
                (65, 'T4'),
            ]
            
            all_pass = True
            for size_mm, expected_stage in tests:
                actual_stage = gen.classify_severity(size_mm)
                if actual_stage != expected_stage:
                    all_pass = False
                    print(f"  [FAIL] Size {size_mm}mm: expected {expected_stage}, got {actual_stage}")
                else:
                    print(f"  [OK] Size {size_mm}mm -> {actual_stage}")
            
            detail = f"Tested 5 size thresholds, all correct"
            self.log_test("Severity Classification", all_pass, detail)
        except Exception as e:
            self.log_test("Severity Classification", False, str(e))
    
    def test_6_json_output(self, pred, rect_bbox, circ_bbox):
        """Test 6: JSON output generation."""
        print("\n" + "="*70)
        print("TEST 6: JSON OUTPUT")
        print("="*70)
        
        try:
            json_output = {
                'prediction': {
                    'width_mm': pred.width_mm,
                    'height_mm': pred.height_mm,
                    'depth_mm': pred.depth_mm,
                    'confidence': pred.confidence,
                },
                'severity': {
                    'stage': pred.severity,
                    'probabilities': {
                        'T1': float(pred.severity_logits[0]),
                        'T2': float(pred.severity_logits[1]),
                        'T3': float(pred.severity_logits[2]),
                        'T4': float(pred.severity_logits[3]),
                    }
                },
                'bounding_boxes': {
                    'rectangular': {
                        'x1': int(rect_bbox['x1']),
                        'y1': int(rect_bbox['y1']),
                        'x2': int(rect_bbox['x2']),
                        'y2': int(rect_bbox['y2']),
                    },
                    'circular': {
                        'center_x': int(circ_bbox['center_x']),
                        'center_y': int(circ_bbox['center_y']),
                        'radius_px': int(circ_bbox['radius_px']),
                    }
                }
            }
            
            # Verify JSON serialization
            json_str = json.dumps(json_output, indent=2)
            
            # Save to file
            output_file = self.output_dir / 'sample_prediction.json'
            with open(output_file, 'w') as f:
                f.write(json_str)
            
            detail = f"Saved to {output_file}, size: {len(json_str)} bytes"
            self.log_test("JSON Output", True, detail)
        except Exception as e:
            self.log_test("JSON Output", False, str(e))
    
    def run_all(self):
        """Run all tests."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST SUITE")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        # Run tests in sequence
        model = self.test_1_model_init()
        if model:
            self.test_2_weight_loading(model)
            output = self.test_3_inference(model)
            if output is not None:
                pred, rect_bbox, circ_bbox = self.test_4_bbox_generation(output)
                if pred is not None:
                    self.test_5_severity_classification(pred)
                    self.test_6_json_output(pred, rect_bbox, circ_bbox)
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total tests: {self.test_count}")
        print(f"Passed: {self.passed_count}")
        print(f"Failed: {self.test_count - self.passed_count}")
        print(f"Success rate: {100 * self.passed_count / self.test_count:.1f}%")
        
        # Save results
        results_file = self.output_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'total': self.test_count,
                'passed': self.passed_count,
                'failed': self.test_count - self.passed_count,
                'results': self.results
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        return self.passed_count == self.test_count


if __name__ == '__main__':
    suite = ComprehensiveTestSuite()
    success = suite.run_all()
    sys.exit(0 if success else 1)
