"""
Complete Pipeline Tester
Tests all components of the tumor size prediction system
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_imports():
    """Test that all required modules can be imported"""
    print("\n" + "="*70)
    print("1️⃣  TESTING IMPORTS")
    print("="*70)
    
    modules_to_test = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('matplotlib', 'Matplotlib'),
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('PIL', 'Pillow'),
        ('pydicom', 'PyDICOM'),
    ]
    
    results = {}
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {display_name:20} - OK")
            results[module_name] = True
        except ImportError as e:
            print(f"✗ {display_name:20} - FAILED: {e}")
            results[module_name] = False
    
    return all(results.values())


def test_models():
    """Test that model files exist"""
    print("\n" + "="*70)
    print("2️⃣  TESTING MODEL FILES")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    model_dir = project_root / 'models'
    
    models_to_check = [
        ('size_predictor_model.pth', 'Size Predictor Model'),
        ('baseline_real_t2_adc_3s_ep1.pth', 'Baseline T2/ADC Model'),
        ('yolov8s.pt', 'YOLO Detection Model'),
    ]
    
    results = {}
    for filename, display_name in models_to_check:
        model_path = model_dir / filename
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✓ {display_name:30} - {size_mb:.1f} MB")
            results[filename] = True
        else:
            print(f"⚠ {display_name:30} - NOT FOUND (optional)")
            results[filename] = False
    
    return True  # Don't fail if models missing (can be generated)


def test_source_files():
    """Test that all required source files exist"""
    print("\n" + "="*70)
    print("3️⃣  TESTING SOURCE FILES")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    
    files_to_check = [
        ('src/size_predictor_model.py', 'Size Predictor Model'),
        ('src/bbox_utils.py', 'Bounding Box Utils'),
        ('src/visualization_enhanced.py', 'Visualization Utils'),
        ('src/infer_with_bbox.py', 'Inference Pipeline'),
        ('src/utils_dicom.py', 'DICOM Utils'),
        ('webapp/fastapi_server.py', 'FastAPI Server'),
        ('scripts/demo_full_pipeline.py', 'Demo Script'),
        ('scripts/test_api_client.py', 'API Test Client'),
    ]
    
    results = {}
    for filepath, display_name in files_to_check:
        full_path = project_root / filepath
        if full_path.exists():
            size_kb = full_path.stat().st_size / 1024
            print(f"✓ {display_name:30} - {size_kb:.1f} KB")
            results[filepath] = True
        else:
            print(f"✗ {display_name:30} - NOT FOUND")
            results[filepath] = False
    
    return all(results.values())


def test_data_structure():
    """Test that data directory structure is correct"""
    print("\n" + "="*70)
    print("4️⃣  TESTING DATA STRUCTURE")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    # Check directories
    dirs_to_check = [
        ('PROSTATEx', 'DICOM Patient Data'),
        ('.', 'Data Root'),
    ]
    
    all_ok = True
    
    for dirname, display_name in dirs_to_check:
        dir_path = data_dir / dirname if dirname != '.' else data_dir
        if dir_path.exists():
            if dirname == '.':
                file_count = len(list(dir_path.glob('*')))
            else:
                patient_count = len(list(dir_path.glob('ProstateX-*')))
                print(f"✓ {display_name:30} - {patient_count} patients")
                continue
            print(f"✓ {display_name:30} - Found")
        else:
            print(f"⚠ {display_name:30} - Directory not found (optional)")
    
    # Check metadata
    metadata_files = [
        ('metadata.csv', 'Patient Metadata'),
        ('ProstateX-Findings-Test.csv', 'Findings Data'),
    ]
    
    for filename, display_name in metadata_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            lines = len(filepath.read_text().splitlines())
            print(f"✓ {display_name:30} - {lines} lines ({size_kb:.1f} KB)")
        else:
            print(f"⚠ {display_name:30} - NOT FOUND (optional)")
    
    return True


def test_python_code():
    """Test that Python code has no syntax errors"""
    print("\n" + "="*70)
    print("5️⃣  TESTING PYTHON SYNTAX")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    
    python_files = list((project_root / 'src').glob('*.py')) + \
                   list((project_root / 'scripts').glob('*.py')) + \
                   list((project_root / 'webapp').glob('*.py'))
    
    results = {}
    for py_file in sorted(python_files):
        try:
            with open(py_file, 'r') as f:
                compile(f.read(), str(py_file), 'exec')
            print(f"✓ {py_file.relative_to(project_root)}")
            results[str(py_file)] = True
        except SyntaxError as e:
            print(f"✗ {py_file.relative_to(project_root)} - {e}")
            results[str(py_file)] = False
    
    return all(results.values())


def test_size_predictor_module():
    """Test size predictor model loading"""
    print("\n" + "="*70)
    print("6️⃣  TESTING SIZE PREDICTOR MODULE")
    print("="*70)
    
    try:
        import torch
        from src.size_predictor_model import SizePredictorModel
        
        print("  Creating model...")
        model = SizePredictorModel(in_channels=3, out_channels=1)
        print(f"  ✓ Model created")
        
        print("  Testing forward pass...")
        x = torch.randn(2, 3, 32, 32, 16)  # batch=2, channels=3, D=32, H=32, W=16
        output = model(x)
        print(f"  ✓ Forward pass OK: input {x.shape} -> output {output.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False


def test_bbox_utils_module():
    """Test bbox utilities"""
    print("\n" + "="*70)
    print("7️⃣  TESTING BBOX UTILITIES")
    print("="*70)
    
    try:
        from src.bbox_utils import (
            BoundingBoxPostProcessor,
            SeverityClassifier,
            round_bbox_corners
        )
        
        print("  Testing BoundingBoxPostProcessor...")
        processor = BoundingBoxPostProcessor()
        bbox = processor.estimate_bbox_from_size(25.0, image_shape=(128, 128))
        print(f"  ✓ Estimated bbox for 25mm tumor: {bbox}")
        
        print("  Testing round_bbox_corners...")
        bbox_rounded = round_bbox_corners(bbox, grid_size=8)
        print(f"  ✓ Rounded bbox: {bbox_rounded}")
        
        print("  Testing SeverityClassifier...")
        classifier = SeverityClassifier()
        severity = classifier.classify_severity(25.0, 100, 80, 150)
        print(f"  ✓ Severity for 25mm: {severity['class']} ({severity['risk_level']})")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False


def test_visualization_module():
    """Test visualization utilities"""
    print("\n" + "="*70)
    print("8️⃣  TESTING VISUALIZATION MODULE")
    print("="*70)
    
    try:
        import numpy as np
        from src.visualization_enhanced import (
            get_severity_color,
            normalize_to_8bit,
            draw_bbox_on_image,
            SEVERITY_COLORS
        )
        
        print("  Testing severity colors...")
        for sev_class in ['T1a', 'T2', 'T3a', 'T4']:
            color = get_severity_color(sev_class)
            print(f"  ✓ {sev_class}: {color}")
        
        print("  Testing normalize_to_8bit...")
        data = np.random.rand(128, 128) * 1000
        normalized = normalize_to_8bit(data)
        print(f"  ✓ Normalized: min={normalized.min()}, max={normalized.max()}")
        
        print("  Testing draw_bbox_on_image...")
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        bbox = (40, 50, 80, 120)
        img_with_bbox = draw_bbox_on_image(img, bbox, 'T2')
        print(f"  ✓ Drew bbox on image: shape {img_with_bbox.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False


def test_comprehensive_pipeline():
    """Test complete inference pipeline"""
    print("\n" + "="*70)
    print("9️⃣  TESTING COMPREHENSIVE PIPELINE")
    print("="*70)
    
    try:
        import torch
        import numpy as np
        from src.size_predictor_model import SizePredictorModel
        from src.bbox_utils import BoundingBoxPostProcessor, SeverityClassifier
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Using device: {device}")
        
        # Create model
        model = SizePredictorModel(in_channels=3, out_channels=1)
        model = model.to(device)
        model.eval()
        print("  ✓ Model loaded")
        
        # Create synthetic input
        print("  Creating synthetic input...")
        sequences = {
            'T2': np.random.randint(50, 200, (128, 128, 20), dtype=np.uint8),
            'ADC': np.random.randint(30, 150, (128, 128, 20), dtype=np.uint8),
            'DWI': np.random.randint(100, 250, (128, 128, 20), dtype=np.uint8),
        }
        print(f"  ✓ Created sequences: T2 {sequences['T2'].shape}")
        
        # Normalize and stack
        x = np.stack([sequences[s].astype(np.float32) / 255.0 for s in ['T2', 'ADC', 'DWI']], axis=0)
        x = torch.FloatTensor(x).unsqueeze(0).to(device)
        print(f"  ✓ Prepared input tensor: {x.shape}")
        
        # Inference
        print("  Running inference...")
        with torch.no_grad():
            output = model(x)
        
        size_pred = output.item() if hasattr(output, 'item') else output
        print(f"  ✓ Predicted tumor size: {size_pred:.2f} mm")
        
        # Post-processing
        print("  Computing bounding box...")
        processor = BoundingBoxPostProcessor()
        bbox = processor.estimate_bbox_from_size(size_pred, image_shape=(128, 128))
        print(f"  ✓ Bounding box: {bbox}")
        
        # Severity
        print("  Classifying severity...")
        classifier = SeverityClassifier()
        severity = classifier.classify_severity(
            size_mm=size_pred,
            t2_intensity=np.mean(sequences['T2']),
            adc_intensity=np.mean(sequences['ADC']),
            dwi_intensity=np.mean(sequences['DWI'])
        )
        print(f"  ✓ Severity: {severity['class']} ({severity['risk_level']})")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False


def test_fastapi_imports():
    """Test FastAPI server can be imported"""
    print("\n" + "="*70)
    print("🔟 TESTING FASTAPI SERVER")
    print("="*70)
    
    try:
        # This will fail if imports in fastapi_server.py fail
        from webapp.fastapi_server import app
        print(f"  ✓ FastAPI app imported successfully")
        print(f"  ✓ Routes available: {len(app.routes)}")
        
        # List main endpoints
        endpoints = set()
        for route in app.routes:
            if hasattr(route, 'path'):
                endpoints.add(route.path)
        
        for endpoint in sorted(endpoints)[:10]:
            print(f"    - {endpoint}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False


def create_test_report(results: dict):
    """Create test report"""
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    percentage = (passed / total * 100) if total > 0 else 0
    
    print(f"\nResults: {passed}/{total} tests passed ({percentage:.1f}%)")
    print("\nTest Details:")
    for test_name, passed_flag in results.items():
        status = "✅" if passed_flag else "❌"
        print(f"  {status} {test_name}")
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path(__file__).parent.parent / f'test_report_{timestamp}.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TUMOR SIZE PREDICTION SYSTEM - TEST REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Results: {passed}/{total} tests passed ({percentage:.1f}%)\n\n")
        
        for test_name, passed_flag in results.items():
            status = "PASS" if passed_flag else "FAIL"
            f.write(f"{status:6} - {test_name}\n")
    
    print(f"\n💾 Report saved: {report_file}")


def main():
    """Run all tests"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  TUMOR SIZE PREDICTION SYSTEM - COMPREHENSIVE TEST SUITE".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    start_time = time.time()
    
    tests = [
        ("Imports", test_imports),
        ("Model Files", test_models),
        ("Source Files", test_source_files),
        ("Data Structure", test_data_structure),
        ("Python Syntax", test_python_code),
        ("Size Predictor Module", test_size_predictor_module),
        ("BBox Utils Module", test_bbox_utils_module),
        ("Visualization Module", test_visualization_module),
        ("Comprehensive Pipeline", test_comprehensive_pipeline),
        ("FastAPI Server", test_fastapi_imports),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    elapsed = time.time() - start_time
    
    create_test_report(results)
    
    print(f"\n⏱️  Total time: {elapsed:.2f}s")
    print("\n" + "█"*70)
    
    # Return exit code
    all_passed = all(results.values())
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    
    print("█"*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
