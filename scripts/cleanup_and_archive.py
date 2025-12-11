#!/usr/bin/env python3
"""
Cleanup script: Archives large/unnecessary files and prepares clean deploy structure.
"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / 'archive'
DEPLOY_CLEAN = ROOT / 'deploy_clean'

# Files/folders to archive (large or temp)
ARCHIVE_ITEMS = [
    'notebooks',  # Jupyter notebooks (often large, not needed in prod)
    'yolo_dataset',  # Large YOLO data
    'yolo_dataset_fixed',  # Duplicate
    'seg_dataset_full',  # Large segmentation data
    'seg_dataset_test2',  # Test data
    'seg_dataset_try',  # Experimental
    'models/yolov8s_prostate_smoke',  # YOLO model (not used)
    'models/yolov8s_prostate_smoke64',  # YOLO model (not used)
    'models/prototype_toy.pth',  # Toy model
    'models/prototype_toy.pt',  # Toy model
    'test_results',  # Test outputs
    '*.log',  # Log files (wildcard)
    'train_log.txt',
    'run_status.txt',
    'prepare_out.txt',
    'prepare_seg_output.txt',
    'server_run_output.txt',
    'python-manager-25.0.msix',  # Windows installer
]

# Core files to keep (minimal deploy)
KEEP_CORE = {
    'src',
    'models/baseline_real_t2_adc_3s_ep1.pth',
    'webapp',
    'scripts',  # Keep all scripts for reference
    'data',  # Data is needed for reference
    'deploy',
    'results',
    'requirements.txt',
    'README.md',
}

def main():
    print("=" * 70)
    print("CLEANUP & ARCHIVE SCRIPT")
    print("=" * 70)
    
    # Create archive directory
    ARCHIVE.mkdir(exist_ok=True)
    print(f"\n[+] Archive directory: {ARCHIVE}")
    
    archived_count = 0
    archived_size_mb = 0
    
    # Archive items
    for item in ARCHIVE_ITEMS:
        if '*' in item:  # Wildcard patterns
            import glob
            matches = glob.glob(str(ROOT / item))
            for m in matches:
                p = Path(m)
                if p.exists() and p.relative_to(ROOT).as_posix() != 'archive':
                    dest = ARCHIVE / p.name
                    try:
                        if p.is_dir():
                            if dest.exists():
                                shutil.rmtree(dest)
                            shutil.move(str(p), str(dest))
                        else:
                            shutil.move(str(p), str(dest))
                        size_mb = p.stat().st_size / (1024*1024) if p.is_file() else sum(
                            f.stat().st_size for f in p.rglob('*') if f.is_file()
                        ) / (1024*1024)
                        archived_size_mb += size_mb
                        archived_count += 1
                        print(f"    [archived] {p.name} ({size_mb:.1f} MB)")
                    except Exception as e:
                        print(f"    [ERROR] {p.name}: {e}")
        else:
            p = ROOT / item
            if p.exists() and p.relative_to(ROOT).as_posix() != 'archive':
                dest = ARCHIVE / p.name
                try:
                    if p.is_dir():
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.move(str(p), str(dest))
                    else:
                        shutil.move(str(p), str(dest))
                    size_mb = p.stat().st_size / (1024*1024) if p.is_file() else sum(
                        f.stat().st_size for f in p.rglob('*') if f.is_file()
                    ) / (1024*1024)
                    archived_size_mb += size_mb
                    archived_count += 1
                    print(f"    [archived] {item} ({size_mb:.1f} MB)")
                except Exception as e:
                    print(f"    [ERROR] {item}: {e}")
    
    print(f"\n[OK] Archived {archived_count} items ({archived_size_mb:.1f} MB total)")
    
    # Create clean deployment folder
    print(f"\n[+] Creating clean deployment structure: {DEPLOY_CLEAN}")
    DEPLOY_CLEAN.mkdir(exist_ok=True)
    
    # Copy essential structure
    essential = [
        'src',
        'webapp',
        'models/baseline_real_t2_adc_3s_ep1.pth',
        'requirements.txt',
        'README.md',
    ]
    
    for item in essential:
        src = ROOT / item
        if src.is_dir():
            dst = DEPLOY_CLEAN / item
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"    [copied] {item}/")
        elif src.is_file():
            dst_dir = DEPLOY_CLEAN / src.parent.name
            dst_dir.mkdir(exist_ok=True)
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            print(f"    [copied] {item}")
    
    # Create deployment README
    deploy_readme = DEPLOY_CLEAN / 'DEPLOYMENT.md'
    deploy_readme.write_text("""# Prostate Tumor Size Analyzer - Deployment Package

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Inference Server
```bash
python -m uvicorn webapp.fastapi_server:app --host 0.0.0.0 --port 8000
```

### 3. Run Interactive UI
```bash
python -m streamlit run webapp/streamlit_app.py
```

## Model
- **File:** `models/baseline_real_t2_adc_3s_ep1.pth`
- **Architecture:** ResNet18-based TumorSizePredictor
- **Input:** 3-channel MRI stack (T2, ADC, DWI) @ 224x224
- **Output:** Tumor size (mm), severity grade, confidence score

## API Endpoints
- `POST /predict` - Single image inference
- `POST /batch` - Batch prediction
- `GET /health` - Health check

## Dataset
Not included in this deployment. Provide your own DICOM/image data.

## Questions?
Refer to the full project README.md in the root directory.
""")
    print(f"    [created] DEPLOYMENT.md")
    
    print("\n" + "=" * 70)
    print("CLEANUP COMPLETE")
    print("=" * 70)
    print(f"\nArchive location: {ARCHIVE}/")
    print(f"Clean deploy folder: {DEPLOY_CLEAN}/")
    print(f"\nNext steps:")
    print(f"  1. Review archive/ contents")
    print(f"  2. Test deploy_clean/ with: cd deploy_clean && python -m streamlit run webapp/streamlit_app.py")
    print(f"  3. Delete archive/ if satisfied, or keep as backup")
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
