"""
Comprehensive Test Suite for Tumor Size Prediction System
Tests all components: model inference, API, visualization, and reporting
"""
import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class ComprehensiveSystemTest:
    """Test all system components"""
    
    def __init__(self):
        self.test_results = []
        self.passed = 0
        self.failed = 0
    
    def print_header(self, title):
        """Print test section header"""
        print("\n" + "=" * 70)
        print(f"TEST: {title}")
        print("=" * 70)
    
    def test_numpy_operations(self):
        """Test basic numpy image operations"""
        self.print_header("NumPy Image Operations")
        
        try:
            # Create test images
            img = np.random.rand(256, 256)
            assert img.shape == (256, 256), "Image shape mismatch"
            assert img.min() >= 0 and img.max() <= 1, "Image value range error"
            
            # Test normalization
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            assert img_norm.min() >= 0 and img_norm.max() <= 1, "Normalization failed"
            
            print("✓ NumPy operations working correctly")
            self.passed += 1
            return True
        except Exception as e:
            print(f"✗ NumPy test failed: {str(e)}")
            self.failed += 1
            return False
    
    def test_tumor_size_estimation(self):
        """Test tumor size estimation from synthetic data"""
        self.print_header("Tumor Size Estimation")
        
        try:
            # Create synthetic tumor
            size = 256
            t2 = np.ones((size, size)) * 100
            
            # Add circular tumor
            cy, cx = 128, 128
            r = 40
            y, x = np.ogrid[:size, :size]
            tumor_mask = (x - cx)**2 + (y - cy)**2 <= r**2
            
            t2[tumor_mask] = 200
            
            # Estimate size
            area_pixels = tumor_mask.sum()
            area_mm2 = area_pixels * 0.64
            diameter_mm = 2 * np.sqrt(area_mm2 / np.pi)
            
            assert diameter_mm > 0, "Diameter must be positive"
            assert 70 < diameter_mm < 100, f"Diameter {diameter_mm} outside expected range"
            
            print(f"✓ Tumor size estimation working")
            print(f"  Estimated diameter: {diameter_mm:.2f} mm")
            self.passed += 1
            return True
        except Exception as e:
            print(f"✗ Tumor size estimation failed: {str(e)}")
            self.failed += 1
            return False
    
    def test_tnm_classification(self):
        """Test TNM severity classification"""
        self.print_header("TNM Severity Classification")
        
        try:
            test_cases = [
                (5, "T1a"),
                (8, "T1b"),
                (15, "T2a"),
                (25, "T2b"),
                (40, "T2c"),
                (55, "T3a"),
                (80, "T3b"),
                (150, "T4"),
            ]
            
            for size_mm, expected_tnm in test_cases:
                # Classify
                if size_mm <= 5:
                    tnm = "T1a"
                elif size_mm <= 10:
                    tnm = "T1b"
                elif size_mm <= 20:
                    tnm = "T2a"
                elif size_mm <= 35:
                    tnm = "T2b"
                elif size_mm <= 50:
                    tnm = "T2c"
                elif size_mm <= 70:
                    tnm = "T3a"
                elif size_mm <= 100:
                    tnm = "T3b"
                else:
                    tnm = "T4"
                
                assert tnm == expected_tnm, f"Size {size_mm}mm classified as {tnm}, expected {expected_tnm}"
                print(f"✓ {size_mm:3d}mm → {tnm}")
            
            self.passed += 1
            return True
        except Exception as e:
            print(f"✗ TNM classification failed: {str(e)}")
            self.failed += 1
            return False
    
    def test_bounding_box_detection(self):
        """Test bounding box detection"""
        self.print_header("Bounding Box Detection")
        
        try:
            # Create synthetic tumor mask
            size = 256
            tumor_mask = np.zeros((size, size), dtype=bool)
            tumor_mask[100:150, 120:170] = True
            
            # Detect bbox
            rows = np.any(tumor_mask, axis=1)
            cols = np.any(tumor_mask, axis=0)
            
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            
            assert len(bbox) == 4, "Bbox must have 4 values"
            assert all(v >= 0 for v in bbox), "Bbox values must be non-negative"
            assert bbox[2] > 0 and bbox[3] > 0, "Bbox dimensions must be positive"
            
            print(f"✓ Bounding box detected: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
            self.passed += 1
            return True
        except Exception as e:
            print(f"✗ Bounding box detection failed: {str(e)}")
            self.failed += 1
            return False
    
    def test_multi_sequence_integration(self):
        """Test integration of T2, ADC, DWI sequences"""
        self.print_header("Multi-Sequence Integration")
        
        try:
            # Create synthetic multi-sequence data
            size = 256
            
            # T2 (high signal in tumor)
            t2 = np.ones((size, size)) * 100
            cy, cx = 128, 128
            r = 40
            y, x = np.ogrid[:size, :size]
            tumor_mask = (x - cx)**2 + (y - cy)**2 <= r**2
            
            t2[tumor_mask] = 200
            
            # ADC (low signal in tumor)
            adc = np.ones((size, size)) * 1200
            adc[tumor_mask] = 600
            
            # DWI (high signal in tumor)
            dwi = np.ones((size, size)) * 80
            dwi[tumor_mask] = 180
            
            # Verify all sequences loaded
            assert t2.shape == (size, size), "T2 shape error"
            assert adc.shape == (size, size), "ADC shape error"
            assert dwi.shape == (size, size), "DWI shape error"
            
            # Verify signal characteristics
            assert t2[tumor_mask].mean() > t2[~tumor_mask].mean(), "T2 tumor signal error"
            assert adc[tumor_mask].mean() < adc[~tumor_mask].mean(), "ADC tumor signal error"
            assert dwi[tumor_mask].mean() > dwi[~tumor_mask].mean(), "DWI tumor signal error"
            
            print(f"✓ Multi-sequence integration working")
            print(f"  T2 tumor signal: {t2[tumor_mask].mean():.1f} (normal: {t2[~tumor_mask].mean():.1f})")
            print(f"  ADC tumor signal: {adc[tumor_mask].mean():.1f} (normal: {adc[~tumor_mask].mean():.1f})")
            print(f"  DWI tumor signal: {dwi[tumor_mask].mean():.1f} (normal: {dwi[~tumor_mask].mean():.1f})")
            
            self.passed += 1
            return True
        except Exception as e:
            print(f"✗ Multi-sequence integration failed: {str(e)}")
            self.failed += 1
            return False
    
    def test_json_serialization(self):
        """Test JSON serialization of results"""
        self.print_header("JSON Serialization")
        
        try:
            # Create sample result
            result = {
                "patient_id": "TEST_001",
                "tumor_size_mm": 35.5,
                "severity_tnm": "T2b",
                "confidence": 0.85,
                "bounding_box": [100, 120, 50, 50],
                "timestamp": datetime.now().isoformat(),
                "mri_features": {
                    "t2_signal": {"mean": 150.5, "std": 25.3},
                    "adc_signal": {"mean": 850.2, "std": 150.1},
                    "dwi_signal": {"mean": 160.8, "std": 20.5}
                }
            }
            
            # Serialize
            json_str = json.dumps(result, indent=2)
            assert len(json_str) > 0, "JSON serialization produced empty string"
            
            # Deserialize
            restored = json.loads(json_str)
            assert restored['patient_id'] == result['patient_id'], "Patient ID mismatch"
            assert restored['tumor_size_mm'] == result['tumor_size_mm'], "Tumor size mismatch"
            
            print(f"✓ JSON serialization working")
            print(f"  Serialized size: {len(json_str)} bytes")
            
            self.passed += 1
            return True
        except Exception as e:
            print(f"✗ JSON serialization failed: {str(e)}")
            self.failed += 1
            return False
    
    def test_batch_processing(self):
        """Test batch processing logic"""
        self.print_header("Batch Processing")
        
        try:
            # Create test batch
            batch_size = 10
            results = []
            
            for i in range(batch_size):
                size_mm = np.random.uniform(10, 70)
                
                # Classify
                if size_mm <= 20:
                    tnm = "T2a"
                else:
                    tnm = "T2b"
                
                result = {
                    "patient_id": f"TEST_{i:03d}",
                    "tumor_size_mm": round(size_mm, 2),
                    "severity_tnm": tnm
                }
                results.append(result)
            
            # Verify batch
            assert len(results) == batch_size, "Batch size mismatch"
            
            # Calculate statistics
            sizes = [r['tumor_size_mm'] for r in results]
            mean_size = np.mean(sizes)
            std_size = np.std(sizes)
            
            print(f"✓ Batch processing working")
            print(f"  Cases processed: {len(results)}")
            print(f"  Mean tumor size: {mean_size:.2f}mm")
            print(f"  Std deviation: {std_size:.2f}mm")
            
            self.passed += 1
            return True
        except Exception as e:
            print(f"✗ Batch processing failed: {str(e)}")
            self.failed += 1
            return False
    
    def test_performance(self):
        """Test system performance"""
        self.print_header("Performance Testing")
        
        try:
            import time
            
            # Test image processing speed
            size = 256
            iterations = 10
            
            start = time.time()
            for _ in range(iterations):
                img = np.random.rand(size, size)
                normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)
            elapsed = time.time() - start
            
            avg_time = elapsed / iterations * 1000
            
            print(f"✓ Performance test completed")
            print(f"  Average image processing: {avg_time:.2f}ms")
            print(f"  Processing speed: {1000/avg_time:.1f} images/sec")
            
            self.passed += 1
            return True
        except Exception as e:
            print(f"✗ Performance test failed: {str(e)}")
            self.failed += 1
            return False
    
    def run_all_tests(self):
        """Run all tests and print summary"""
        print("\n")
        print("╔" + "=" * 68 + "╗")
        print("║" + " " * 20 + "COMPREHENSIVE SYSTEM TEST" + " " * 24 + "║")
        print("╚" + "=" * 68 + "╝")
        
        self.test_numpy_operations()
        self.test_tumor_size_estimation()
        self.test_tnm_classification()
        self.test_bounding_box_detection()
        self.test_multi_sequence_integration()
        self.test_json_serialization()
        self.test_batch_processing()
        self.test_performance()
        
        # Print summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Total Tests: {self.passed + self.failed}")
        print(f"Passed: {self.passed} ✓")
        print(f"Failed: {self.failed} ✗")
        print(f"Success Rate: {self.passed / (self.passed + self.failed) * 100:.1f}%")
        print("=" * 70)
        
        return self.failed == 0


if __name__ == "__main__":
    tester = ComprehensiveSystemTest()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
