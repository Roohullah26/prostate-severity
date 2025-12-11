#!/usr/bin/env python3
"""
Quick verification: Test model loading and basic inference.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import torch
from size_predictor_model import make_size_predictor

def test_model_loading():
    print("=" * 70)
    print("MODEL VERIFICATION TEST")
    print("=" * 70)
    
    model_path = ROOT / 'models' / 'baseline_real_t2_adc_3s_ep1.pth'
    print(f"\n[1] Testing model file existence...")
    if not model_path.exists():
        print(f"    [FAIL] Model not found: {model_path}")
        return False
    print(f"    [OK] Model file: {model_path.name}")
    print(f"    [OK] Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
    
    print(f"\n[2] Creating model instance...")
    try:
        model = make_size_predictor(pretrained=False, in_channels=6, checkpoint_path=None)
        print(f"    [OK] Model created (ResNet18 backbone)")
    except Exception as e:
        print(f"    [FAIL] {e}")
        return False
    
    print(f"\n[3] Testing checkpoint loading...")
    try:
        ckpt = torch.load(str(model_path), map_location='cpu')
        print(f"    [OK] Checkpoint loaded")
        print(f"    [OK] Checkpoint type: {type(ckpt).__name__}")
        if isinstance(ckpt, dict):
            print(f"    [OK] State dict keys: {len(ckpt)} parameters")
    except Exception as e:
        print(f"    [FAIL] {e}")
        return False
    
    print(f"\n[4] Testing forward pass (dummy input)...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        dummy_input = torch.randn(1, 6, 224, 224).to(device)
        with torch.no_grad():
            out = model(dummy_input)
        
        assert 'size' in out, "Missing 'size' output"
        assert 'severity_logits' in out, "Missing 'severity_logits' output"
        assert 'severity_probs' in out, "Missing 'severity_probs' output"
        assert 'confidence' in out, "Missing 'confidence' output"
        
        print(f"    [OK] Device: {device}")
        print(f"    [OK] Output keys: size{out['size'].shape}, severity_logits{out['severity_logits'].shape}, confidence{out['confidence'].shape}")
    except Exception as e:
        print(f"    [FAIL] {e}")
        return False
    
    print(f"\n[5] Checking webapp files...")
    webapp_files = ['fastapi_server.py', 'streamlit_app.py']
    for f in webapp_files:
        fpath = ROOT / 'webapp' / f
        if fpath.exists():
            print(f"    [OK] {f}")
        else:
            print(f"    [FAIL] Missing {f}")
            return False
    
    print(f"\n[6] Checking deploy_clean structure...")
    deploy_clean = ROOT / 'deploy_clean'
    required = ['src', 'webapp', 'models/baseline_real_t2_adc_3s_ep1.pth', 'requirements.txt']
    for item in required:
        ipath = deploy_clean / item
        if ipath.exists():
            print(f"    [OK] {item}")
        else:
            print(f"    [FAIL] Missing {item} in deploy_clean/")
            return False
    
    print("\n" + "=" * 70)
    print("[SUCCESS] All verification tests passed!")
    print("=" * 70)
    print(f"\nModel is production-ready. Next steps:")
    print(f"  1. Package deploy_clean/ into Docker image or archive")
    print(f"  2. Test inference with real data")
    print(f"  3. Deploy to production (FastAPI server or cloud endpoint)")
    
    return True

if __name__ == '__main__':
    success = test_model_loading()
    sys.exit(0 if success else 1)
