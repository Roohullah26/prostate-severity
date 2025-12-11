#!/usr/bin/env python
"""
FastAPI Server Smoke Test
Generates a test image and sends it to the /predict-size endpoint.
"""

import requests
import numpy as np
from PIL import Image
import json
import io
import sys
from pathlib import Path
import time

def generate_test_mri_image(size=224):
    """Generate a synthetic MRI image."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    tumor_mask = (X**2 + Y**2) < 0.3
    base_intensity = np.full((size, size), 100, dtype=np.uint8)
    base_intensity[tumor_mask] = 180
    
    noise = np.random.normal(0, 10, (size, size))
    image = np.clip(base_intensity + noise, 0, 255).astype(np.uint8)
    
    return image


def run_api_test():
    """Test the FastAPI /predict-size endpoint."""
    print("\n" + "="*80)
    print("[API] FASTAPI SMOKE TEST")
    print("="*80)
    
    # Configuration
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/predict-size"
    
    print(f"\nAPI Endpoint: {endpoint}")
    print(f"Model path env var: PROSTATE_SIZE_MODEL")
    
    # Check if server is running
    try:
        health_response = requests.get(f"{base_url}/health", timeout=2)
        if health_response.status_code == 200:
            print(f"[OK] API server is running")
        else:
            print(f"[FAIL] API server returned status {health_response.status_code}")
            return False
    except requests.ConnectionError:
        print(f"[FAIL] Cannot connect to API server at {base_url}")
        print(f"\nTo start the server, run:")
        print(f"  python -m uvicorn webapp.fastapi_server:app --port 8000 --reload")
        return False
    except Exception as e:
        print(f"[FAIL] Error checking server health: {e}")
        return False
    
    # Generate test image
    print(f"\n[1] Generating test MRI image...")
    test_image = generate_test_mri_image(size=224)
    print(f"[OK] Generated image shape: {test_image.shape}, dtype: {test_image.dtype}")
    
    # Convert to PNG bytes
    pil_img = Image.fromarray(test_image, mode='L')
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    print(f"[OK] Image size: {img_bytes.getbuffer().nbytes} bytes")
    
    # Prepare request
    print(f"\n[2] Sending POST request to {endpoint}...")
    files = {
        'file': ('test_mri.png', img_bytes, 'image/png'),
    }
    params = {
        'model_path': 'models/baseline_real_t2_adc_3s_ep1.pth',
        'bbox_type': 'rect',
        'return_image': 'false',
    }
    
    try:
        response = requests.post(endpoint, files=files, params=params, timeout=10)
        
        if response.status_code == 200:
            print(f"[OK] Request successful (status 200)")
        else:
            print(f"[FAIL] Request returned status {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False
        
    except requests.exceptions.Timeout:
        print(f"[FAIL] Request timed out")
        return False
    except Exception as e:
        print(f"[FAIL] Request error: {e}")
        return False
    
    # Parse response
    print(f"\n[3] Parsing response JSON...")
    try:
        json_response = response.json()
        print(f"[OK] JSON parsed successfully")
        
        # Validate response structure
        required_keys = ['severity', 'width_mm', 'height_mm', 'depth_mm', 'confidence', 'bbox']
        missing_keys = [k for k in required_keys if k not in json_response]
        
        if missing_keys:
            print(f"[FAIL] Missing keys in response: {missing_keys}")
            return False
        
        print(f"[OK] Response has all required keys")
        
        # Print prediction
        print(f"\n[4] Prediction Results:")
        print(f"  Severity: {json_response['severity']}")
        print(f"  Size: {json_response['width_mm']:.2f}W x {json_response['height_mm']:.2f}H x {json_response['depth_mm']:.2f}D mm")
        print(f"  Confidence: {json_response['confidence']:.3f}")
        print(f"  Bbox type: {json_response['bbox'].get('type', 'unknown')}")
        
        if 'severity_probabilities' in json_response:
            probs = json_response['severity_probabilities']
            print(f"  Severity probabilities:")
            for stage, prob in probs.items():
                print(f"    {stage}: {prob:.3f}")
        
        # Save response
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        response_file = results_dir / 'api_response.json'
        with open(response_file, 'w') as f:
            json.dump(json_response, f, indent=2)
        
        print(f"\n[OK] Response saved to {response_file}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"[FAIL] Failed to parse JSON: {e}")
        print(f"Response text: {response.text[:500]}")
        return False
    except Exception as e:
        print(f"[FAIL] Error parsing response: {e}")
        return False


if __name__ == '__main__':
    success = run_api_test()
    
    print("\n" + "="*80)
    if success:
        print("[OK] API SMOKE TEST PASSED")
        print("="*80)
        sys.exit(0)
    else:
        print("[FAIL] API SMOKE TEST FAILED")
        print("="*80)
        sys.exit(1)
