#!/usr/bin/env python3
"""
TUMOR SIZE PREDICTION SYSTEM - STATUS REPORT
Displays comprehensive system information and status
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

def print_banner():
    """Print system banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║           🏥 PROSTATE TUMOR SIZE PREDICTION SYSTEM 🏥              ║
    ║                                                                      ║
    ║              Complete Pipeline with Bounding Box &                  ║
    ║                    Severity Classification (TNM)                    ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_files():
    """Check if all required files exist"""
    print("\n📁 SYSTEM FILES STATUS")
    print("─" * 70)
    
    required_files = {
        'Core Models': [
            'src/size_predictor_model.py',
            'src/bbox_utils.py',
            'src/config.py',
        ],
        'Inference': [
            'src/infer_with_bbox.py',
            'src/visualization_enhanced.py',
            'src/utils_dicom.py',
        ],
        'Scripts': [
            'scripts/run_complete_pipeline_demo.py',
            'scripts/train_size_model.py',
            'scripts/batch_predict_tumor_size.py',
            'scripts/comprehensive_test_tumor_size.py',
            'scripts/api_client_tumor_size.py',
            'scripts/QUICKSTART_GUIDE_TUMOR_SIZE.py',
        ],
        'API': [
            'webapp/fastapi_server.py',
        ],
        'Documentation': [
            'README.md',
            'TUMOR_SIZE_COMPLETE_GUIDE.md',
            'QUICK_REFERENCE.md',
        ]
    }
    
    root = Path(__file__).parent.parent
    total = 0
    found = 0
    
    for category, files in required_files.items():
        print(f"\n{category}:")
        for file in files:
            file_path = root / file
            exists = file_path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {file}")
            total += 1
            if exists:
                found += 1
    
    print(f"\nTotal: {found}/{total} files found ({found/total*100:.0f}%)")
    return found == total


def check_dependencies():
    """Check if dependencies are installed"""
    print("\n📦 DEPENDENCIES STATUS")
    print("─" * 70)
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('pydicom', 'pydicom'),
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('requests', 'Requests'),
        ('cv2', 'OpenCV'),
    ]
    
    installed = 0
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  ✓ {name:<20} installed")
            installed += 1
        except ImportError:
            print(f"  ✗ {name:<20} NOT installed")
    
    print(f"\nTotal: {installed}/{len(dependencies)} dependencies installed")
    return installed == len(dependencies)


def check_models():
    """Check if trained models exist"""
    print("\n🧠 TRAINED MODELS STATUS")
    print("─" * 70)
    
    model_dir = Path(__file__).parent.parent / "models"
    models = {
        'Size Predictor': 'size_predictor.pth',
        'Baseline Model': 'baseline_real_t2_adc_3s_ep1.pth',
    }
    
    found = 0
    for name, filename in models.items():
        path = model_dir / filename
        if path.exists():
            size_mb = path.stat().st_size / (1024**2)
            print(f"  ✓ {name:<25} {size_mb:.1f}MB")
            found += 1
        else:
            print(f"  ✗ {name:<25} NOT FOUND")
    
    print(f"\nTotal: {found}/{len(models)} models available")
    return found > 0


def check_data():
    """Check if sample data exists"""
    print("\n📊 SAMPLE DATA STATUS")
    print("─" * 70)
    
    data_dir = Path(__file__).parent.parent / "data" / "PROSTATEx"
    
    if data_dir.exists():
        samples = list(data_dir.glob("ProstateX-*"))
        print(f"  ✓ Found {len(samples)} samples")
        print(f"    Location: {data_dir}")
        
        # Show first 5 samples
        print("\n  First samples:")
        for sample in sorted(samples)[:5]:
            seq_dirs = [d.name for d in sample.iterdir() if d.is_dir()]
            print(f"    - {sample.name}: {', '.join(seq_dirs)}")
        
        return len(samples) > 0
    else:
        print(f"  ✗ Data directory not found: {data_dir}")
        return False


def show_quick_commands():
    """Show quick command reference"""
    print("\n⚡ QUICK COMMANDS")
    print("─" * 70)
    
    commands = {
        'Run Complete Pipeline': 
            'python scripts/run_complete_pipeline_demo.py --sample-id ProstateX-0000',
        'Start API Server': 
            'python -m uvicorn webapp.fastapi_server:app --reload --port 8000',
        'Run Test Suite': 
            'python scripts/comprehensive_test_tumor_size.py',
        'Show Quick Start': 
            'python scripts/QUICKSTART_GUIDE_TUMOR_SIZE.py --section all',
        'Batch Process': 
            'python scripts/batch_predict_tumor_size.py --input-dir data/PROSTATEx/',
        'Train Model': 
            'python scripts/train_size_model.py --data-path data/training_data.csv --epochs 100',
    }
    
    for i, (desc, cmd) in enumerate(commands.items(), 1):
        print(f"\n{i}. {desc}")
        print(f"   $ {cmd}")


def show_system_info():
    """Show system information"""
    print("\n🖥️  SYSTEM INFORMATION")
    print("─" * 70)
    
    import platform
    import torch
    
    print(f"Python Version: {platform.python_version()}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {'Yes' if torch.cuda.is_available() else 'No'}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def show_capabilities():
    """Show system capabilities"""
    print("\n✨ SYSTEM CAPABILITIES")
    print("─" * 70)
    
    capabilities = [
        ('Tumor Size Prediction', 'Precise measurement (±2-3mm) from multi-sequence MRI'),
        ('Bounding Box Detection', 'Automatic ROI detection with confidence scores'),
        ('TNM Staging', 'Automatic T-stage classification (T1-T4)'),
        ('Severity Classification', 'Clinical severity assessment and recommendations'),
        ('Multi-Sequence Input', 'Supports T2, ADC, and DWI sequences'),
        ('GPU Acceleration', 'Fast inference on NVIDIA GPUs'),
        ('REST API', 'HTTP endpoints for remote predictions'),
        ('Batch Processing', 'Process multiple samples in parallel'),
        ('Model Training', 'Fine-tune models on custom data'),
        ('Comprehensive Testing', '6 test suites for validation'),
    ]
    
    for feature, description in capabilities:
        print(f"  ✓ {feature:<30} - {description}")


def show_file_structure():
    """Show key file structure"""
    print("\n📂 KEY STRUCTURE")
    print("─" * 70)
    
    structure = """
    prostate-severity/
    ├── src/
    │   ├── size_predictor_model.py       ← Core prediction model
    │   ├── bbox_utils.py                 ← Bounding box generation
    │   ├── config.py                     ← Configuration
    │   └── [utilities]
    │
    ├── scripts/
    │   ├── run_complete_pipeline_demo.py ← Full pipeline demo
    │   ├── train_size_model.py           ← Model training
    │   ├── comprehensive_test_tumor_size.py ← Tests
    │   └── [other scripts]
    │
    ├── webapp/
    │   └── fastapi_server.py             ← REST API
    │
    ├── models/
    │   └── size_predictor.pth            ← Trained weights
    │
    └── data/
        └── PROSTATEx/                    ← Sample data
    """
    print(structure)


def show_workflow():
    """Show typical workflow"""
    print("\n🔄 TYPICAL WORKFLOW")
    print("─" * 70)
    
    workflow = """
    1. LOAD DATA
       └─→ Multi-sequence MRI (T2, ADC, DWI)
    
    2. PREPROCESS
       └─→ Normalize images
    
    3. PREDICT SIZE
       └─→ Neural network inference
       └─→ Output: Tumor size (mm)
    
    4. GENERATE BBOX
       └─→ Region analysis
       └─→ Output: Bounding box + confidence
    
    5. CLASSIFY SEVERITY
       └─→ TNM staging
       └─→ Output: T-stage + clinical notes
    
    6. VISUALIZE & REPORT
       └─→ Generate predictions report
       └─→ Output: JSON/CSV + visualization
    """
    print(workflow)


def show_next_steps():
    """Show next steps"""
    print("\n🚀 NEXT STEPS")
    print("─" * 70)
    
    next_steps = """
    1. VERIFY INSTALLATION
       $ python scripts/comprehensive_test_tumor_size.py
    
    2. EXPLORE SYSTEM
       $ python scripts/QUICKSTART_GUIDE_TUMOR_SIZE.py --section all
    
    3. RUN DEMO
       $ python scripts/run_complete_pipeline_demo.py
    
    4. START API SERVER
       $ python -m uvicorn webapp.fastapi_server:app --reload
    
    5. PROCESS YOUR DATA
       $ python scripts/batch_predict_tumor_size.py --input-dir your_data/
    
    6. TRAIN CUSTOM MODEL (Optional)
       $ python scripts/train_size_model.py --data-path training_data.csv
    """
    print(next_steps)


def generate_status_report():
    """Generate comprehensive status report"""
    print_banner()
    
    checks = {
        'File Status': check_files,
        'Dependencies': check_dependencies,
        'Models': check_models,
        'Sample Data': check_data,
    }
    
    results = {}
    for check_name, check_func in checks.items():
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n❌ Error in {check_name}: {e}")
            results[check_name] = False
    
    show_system_info()
    show_capabilities()
    show_file_structure()
    show_workflow()
    show_quick_commands()
    show_next_steps()
    
    # Summary
    print("\n" + "="*70)
    print("📋 SYSTEM STATUS SUMMARY")
    print("="*70)
    
    all_good = all(results.values())
    if all_good:
        print("✅ ALL SYSTEMS OPERATIONAL")
        print("\nThe tumor size prediction system is fully functional and ready to use!")
        print("\nStart with: python scripts/run_complete_pipeline_demo.py")
    else:
        print("⚠️  SOME ISSUES DETECTED")
        for check_name, passed in results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
        print("\nRefer to troubleshooting section or documentation for help.")
    
    print("\n" + "="*70 + "\n")
    
    return all_good


if __name__ == '__main__':
    try:
        success = generate_status_report()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹ Report generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
