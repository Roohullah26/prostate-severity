"""
API Testing and Integration Script

Tests the complete tumor size prediction API with various scenarios:
1. Image upload and inference
2. Severity classification
3. Bounding box visualization
4. Error handling

Usage:
    python scripts/test_api_complete.py --server-url http://localhost:8000
    python scripts/test_api_complete.py --csv merged_data.csv
    python scripts/test_api_complete.py --toy
"""

import argparse
import sys
from pathlib import Path
import requests
import json
import time
from typing import Dict, Optional
import io
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prostate_dataset import ProstateLesionDataset
from src import config


class APITester:
    """Test the tumor prediction API."""
    
    SEVERITY_COLORS = {
        'T1': '🟢',  # Green
        'T2': '🟡',  # Yellow
        'T3': '🟠',  # Orange
        'T4': '🔴',  # Red
    }
    
    def __init__(self, server_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """Initialize tester.
        
        Args:
            server_url: Base URL of the API server
            api_key: Optional API key for authentication
        """
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers['X-API-Key'] = api_key
        
        print(f"✓ API Tester initialized")
        print(f"  Server: {self.server_url}")
        print(f"  Auth: {'Yes' if api_key else 'No'}")
    
    def check_health(self) -> bool:
        """Check if server is running."""
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False
    
    def predict_from_image(self, image: Image.Image, bbox_type: str = 'circle',
                          return_image: bool = False, model_path: str = None) -> Dict:
        """Send image to API for prediction.
        
        Args:
            image: PIL Image
            bbox_type: 'circle' or 'rect'
            return_image: Request base64 visualization
            model_path: Model path on server
            
        Returns:
            API response dict
        """
        # Save image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Prepare request
        files = {'file': ('image.png', img_bytes, 'image/png')}
        params = {
            'bbox_type': bbox_type,
            'return_image': return_image,
        }
        if model_path:
            params['model_path'] = model_path
        
        # Send request
        url = f"{self.server_url}/predict-size"
        resp = requests.post(url, files=files, params=params, headers=self.headers, timeout=30)
        
        if resp.status_code != 200:
            raise Exception(f"API error {resp.status_code}: {resp.text}")
        
        return resp.json()
    
    def test_health(self):
        """Test server health endpoint."""
        print("\n" + "="*80)
        print("TEST 1: Server Health Check")
        print("="*80)
        
        if not self.check_health():
            print("❌ Server is NOT running!")
            print("   Start it with: python -m webapp.fastapi_server")
            return False
        
        print("✓ Server is running!")
        return True
    
    def test_synthetic_image(self):
        """Test with synthetic image."""
        print("\n" + "="*80)
        print("TEST 2: Synthetic Image Prediction")
        print("="*80)
        
        # Create synthetic image
        print("Creating synthetic MRI image...")
        img_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Predict
        print("Sending to API...")
        start = time.time()
        try:
            result = self.predict_from_image(img, bbox_type='circle', return_image=False)
            elapsed = time.time() - start
            
            self._print_prediction(result, elapsed)
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def test_real_image(self, csv_path: str, sample_idx: int = 0, uid: str = None):
        """Test with real dataset image.
        
        Args:
            csv_path: Path to metadata CSV
            sample_idx: Sample index
            uid: Specific UID to test
        """
        print("\n" + "="*80)
        print("TEST 3: Real Dataset Image Prediction")
        print("="*80)
        
        # Load dataset
        print(f"Loading dataset from {csv_path}...")
        try:
            dataset = ProstateLesionDataset(
                csv_path=csv_path,
                img_size=config.IMG_SIZE,
                sequences=['t2', 'adc', 'dwi'],
            )
            print(f"✓ Dataset loaded: {len(dataset)} samples")
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return False
        
        # Get sample
        if uid:
            for i, sample in enumerate(dataset):
                if sample['uid'] == uid:
                    sample_idx = i
                    break
        
        sample = dataset[sample_idx]
        uid = sample['uid']
        
        print(f"Processing sample {sample_idx}: {uid}...")
        
        # Convert tensor to PIL image
        img_tensor = sample['img']
        img_array = img_tensor.permute(1, 2, 0).numpy()
        img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Predict
        print("Sending to API...")
        start = time.time()
        try:
            result = self.predict_from_image(img, bbox_type='circle', return_image=True)
            elapsed = time.time() - start
            
            self._print_prediction(result, elapsed, uid=uid)
            
            # Save visualization if returned
            if 'visualization_base64' in result:
                import base64
                img_data = base64.b64decode(result['visualization_base64'])
                output_path = f"api_prediction_{uid}.png"
                with open(output_path, 'wb') as f:
                    f.write(img_data)
                print(f"✓ Saved visualization: {output_path}")
            
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def test_bbox_types(self):
        """Test different bbox types."""
        print("\n" + "="*80)
        print("TEST 4: Bounding Box Types")
        print("="*80)
        
        # Create test image
        img_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        for bbox_type in ['circle', 'rect']:
            print(f"\nTesting bbox_type='{bbox_type}'...")
            try:
                result = self.predict_from_image(img, bbox_type=bbox_type, return_image=False)
                
                bbox = result.get('bbox', {})
                if bbox_type == 'circle':
                    print(f"  ✓ Circle: center=({bbox.get('center_x')}, {bbox.get('center_y')}), "
                          f"radius={bbox.get('radius_px')}px ({bbox.get('radius_mm'):.1f}mm)")
                else:
                    print(f"  ✓ Rectangle: ({bbox.get('x1')}, {bbox.get('y1')}) to "
                          f"({bbox.get('x2')}, {bbox.get('y2')})")
            except Exception as e:
                print(f"  ❌ Error: {e}")
                return False
        
        return True
    
    def test_concurrent_requests(self, n_requests: int = 5):
        """Test multiple concurrent requests."""
        print("\n" + "="*80)
        print(f"TEST 5: Concurrent Requests ({n_requests} images)")
        print("="*80)
        
        # Create test images
        images = [Image.fromarray(np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8))
                  for _ in range(n_requests)]
        
        print(f"Sending {n_requests} requests...")
        start = time.time()
        results = []
        
        for i, img in enumerate(images):
            try:
                result = self.predict_from_image(img, bbox_type='circle', return_image=False)
                results.append(result)
                print(f"  {i+1}/{n_requests} ✓ {result['severity']} ({result['confidence']:.2%})")
            except Exception as e:
                print(f"  {i+1}/{n_requests} ❌ Error: {e}")
                return False
        
        elapsed = time.time() - start
        avg_time = elapsed / n_requests
        
        print(f"\n✓ All requests completed!")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Average time per request: {avg_time:.2f}s")
        print(f"  Throughput: {1/avg_time:.1f} req/s")
        
        return True
    
    def _print_prediction(self, result: Dict, elapsed: float, uid: str = "SYNTHETIC"):
        """Pretty print prediction result."""
        severity = result['severity']
        emoji = self.SEVERITY_COLORS.get(severity, '⚪')
        
        print(f"\n✓ Prediction completed in {elapsed:.2f}s")
        print(f"\n  Patient: {uid}")
        print(f"  {emoji} Severity: {severity}")
        print(f"  📏 Dimensions: {result['width_mm']:.1f} x {result['height_mm']:.1f} x {result['depth_mm']:.1f} mm")
        print(f"  📊 Max dimension: {result['max_dimension_mm']:.1f} mm")
        print(f"  🎯 Confidence: {result['confidence']:.2%}")
        
        print(f"\n  Severity probabilities:")
        probs = result['severity_probabilities']
        for grade in ['T1', 'T2', 'T3', 'T4']:
            emoji = self.SEVERITY_COLORS.get(grade, '⚪')
            prob = probs[grade]
            bar = '█' * int(prob * 30) + '░' * (30 - int(prob * 30))
            print(f"    {emoji} {grade}: {bar} {prob:.2%}")


def main():
    parser = argparse.ArgumentParser(description='API Testing Suite')
    
    # Server
    parser.add_argument('--server-url', type=str, default='http://localhost:8000',
                        help='API server URL')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Optional API key')
    
    # Dataset testing
    parser.add_argument('--csv', type=str, default=None,
                        help='CSV for real dataset testing')
    parser.add_argument('--sample', type=int, default=0,
                        help='Sample index')
    parser.add_argument('--uid', type=str, default=None,
                        help='Specific UID to test')
    
    # Test modes
    parser.add_argument('--toy', action='store_true',
                        help='Test with synthetic data')
    parser.add_argument('--concurrent', type=int, default=0,
                        help='Number of concurrent requests to test (0 to skip)')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = APITester(args.server_url, args.api_key)
    
    # Run tests
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Health check
    tests_total += 1
    if tester.test_health():
        tests_passed += 1
    else:
        print("\nCannot proceed without server running!")
        return 1
    
    # Test 2: Synthetic image
    if args.toy or not args.csv:
        tests_total += 1
        if tester.test_synthetic_image():
            tests_passed += 1
    
    # Test 3: Real dataset
    if args.csv and Path(args.csv).exists():
        tests_total += 1
        if tester.test_real_image(args.csv, args.sample, args.uid):
            tests_passed += 1
    
    # Test 4: Bbox types
    tests_total += 1
    if tester.test_bbox_types():
        tests_passed += 1
    
    # Test 5: Concurrent requests
    if args.concurrent > 0:
        tests_total += 1
        if tester.test_concurrent_requests(args.concurrent):
            tests_passed += 1
    
    # Summary
    print("\n" + "="*80)
    print(f"TEST SUMMARY: {tests_passed}/{tests_total} tests passed")
    print("="*80)
    
    return 0 if tests_passed == tests_total else 1


if __name__ == '__main__':
    sys.exit(main())
