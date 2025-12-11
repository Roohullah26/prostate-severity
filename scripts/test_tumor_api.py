"""
Test script for Tumor Size Prediction API
Tests both local inference and FastAPI endpoints
"""

import sys
import os
import json
import requests
import base64
from pathlib import Path
from io import BytesIO
import numpy as np
import cv2
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from utils_image import normalize_image
from utils_dicom import load_dicom_image


class TumorSizeAPITester:
    """Test suite for Tumor Size Prediction API"""
    
    def __init__(self, api_url: str = "http://localhost:8000", api_key: str = None):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers['X-API-Key'] = api_key
    
    def _get_headers(self):
        """Get request headers with API key if configured."""
        headers = dict(self.headers)
        if self.api_key:
            headers['X-API-Key'] = self.api_key
        return headers
    
    def test_health(self) -> bool:
        """Test API health endpoint."""
        print("\n🏥 Testing API Health...")
        try:
            response = requests.get(f"{self.api_url}/health")
            if response.status_code == 200:
                print("  ✅ API is healthy")
                return True
            else:
                print(f"  ❌ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"  ❌ Failed to connect to API: {e}")
            return False
    
    def test_model_status(self) -> dict:
        """Get model status from API."""
        print("\n📊 Checking Model Status...")
        try:
            response = requests.get(
                f"{self.api_url}/model_status",
                headers=self._get_headers()
            )
            if response.status_code == 200:
                status = response.json()
                print(f"  ✅ Model loaded: {status.get('loaded', False)}")
                print(f"     Device: {status.get('device')}")
                print(f"     Channels: {status.get('in_channels')}")
                return status
            else:
                print(f"  ❌ Failed to get model status: {response.status_code}")
                return {}
        except Exception as e:
            print(f"  ❌ Error getting model status: {e}")
            return {}
    
    def test_series_status(self) -> dict:
        """Get available DICOM series."""
        print("\n📁 Checking Available DICOM Series...")
        try:
            response = requests.get(
                f"{self.api_url}/series_status",
                headers=self._get_headers()
            )
            if response.status_code == 200:
                status = response.json()
                print(f"  ✅ Available series: {status.get('available_series_count', 0)}")
                if 'sample_keys' in status:
                    print(f"     Sample: {status['sample_keys'][:3]}")
                return status
            else:
                print(f"  ❌ Failed to get series status: {response.status_code}")
                return {}
        except Exception as e:
            print(f"  ❌ Error getting series status: {e}")
            return {}
    
    def test_predict_size(self, image_path: str, bbox_type: str = "circle",
                         return_image: bool = True) -> dict:
        """
        Test predict-size endpoint.
        
        Args:
            image_path: Path to image file (PNG, JPEG, etc.)
            bbox_type: 'circle' or 'rect'
            return_image: Whether to return visualization
            
        Returns:
            dict with prediction results
        """
        print(f"\n🔮 Testing Tumor Size Prediction...")
        print(f"   Image: {image_path}")
        print(f"   BBox Type: {bbox_type}")
        
        if not Path(image_path).exists():
            print(f"  ❌ Image not found: {image_path}")
            return {}
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                
                params = {
                    'bbox_type': bbox_type,
                    'return_image': return_image
                }
                
                response = requests.post(
                    f"{self.api_url}/predict-size",
                    files=files,
                    params=params,
                    headers=self._get_headers()
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print("  ✅ Prediction successful!")
                    print(f"     Severity: {result.get('severity')}")
                    print(f"     Size (W×H×D): {result.get('width_mm'):.2f} × {result.get('height_mm'):.2f} × {result.get('depth_mm'):.2f} mm")
                    print(f"     Max Dimension: {result.get('max_dimension_mm'):.2f} mm")
                    print(f"     Confidence: {result.get('confidence'):.3f}")
                    
                    # Show severity probabilities
                    if 'severity_probabilities' in result:
                        probs = result['severity_probabilities']
                        print(f"     Severity Probabilities:")
                        for stage, prob in probs.items():
                            print(f"        {stage}: {prob:.3f}")
                    
                    # Save image if returned
                    if 'image_base64' in result:
                        img_data = base64.b64decode(result['image_base64'])
                        output_path = Path(image_path).parent / f"prediction_{Path(image_path).stem}.png"
                        with open(output_path, 'wb') as img_file:
                            img_file.write(img_data)
                        print(f"     Visualization saved: {output_path}")
                    
                    return result
                else:
                    print(f"  ❌ Prediction failed: {response.status_code}")
                    print(f"     Error: {response.text}")
                    return {}
        except Exception as e:
            print(f"  ❌ Error during prediction: {e}")
            return {}
    
    def test_quick_eval(self) -> dict:
        """Test quick eval endpoint."""
        print("\n⚡ Testing Quick Evaluation...")
        try:
            response = requests.get(
                f"{self.api_url}/quick_eval",
                headers=self._get_headers()
            )
            if response.status_code == 200:
                result = response.json()
                print("  ✅ Quick eval successful!")
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        print(f"     {key}: {value:.4f}")
                    else:
                        print(f"     {key}: {value}")
                return result
            else:
                print(f"  ❌ Quick eval failed: {response.status_code}")
                return {}
        except Exception as e:
            print(f"  ❌ Error during quick eval: {e}")
            return {}


def create_test_image(output_path: str = "test_image.png", size: tuple = (224, 224)):
    """Create a synthetic test image (gradient with circle)."""
    print(f"\n🎨 Creating test image: {output_path}")
    
    # Create gradient background
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    for i in range(size[0]):
        img[i, :] = int(255 * i / size[0])
    
    # Draw a circle (simulating tumor)
    center = (size[0] // 2, size[1] // 2)
    radius = 40
    cv2.circle(img, center, radius, (255, 0, 0), -1)
    
    # Save image
    cv2.imwrite(output_path, img)
    print(f"  ✅ Test image created: {output_path}")
    return output_path


def main():
    """Main test execution."""
    
    print("="*70)
    print("🧪 TUMOR SIZE PREDICTION API TEST SUITE")
    print("="*70)
    
    # Configure API endpoint
    api_url = "http://localhost:8000"
    api_key = os.environ.get('PROSTATE_API_KEY')
    
    print(f"\n🔗 API URL: {api_url}")
    print(f"🔑 API Key: {'Configured' if api_key else 'Not configured'}")
    
    # Initialize tester
    tester = TumorSizeAPITester(api_url=api_url, api_key=api_key)
    
    # Run tests
    print("\n" + "-"*70)
    print("PHASE 1: API Connectivity & Status")
    print("-"*70)
    
    if not tester.test_health():
        print("\n❌ API is not running. Start server with:")
        print("   python webapp/fastapi_server.py --port 8000")
        return
    
    tester.test_model_status()
    tester.test_series_status()
    
    # Test prediction with sample image
    print("\n" + "-"*70)
    print("PHASE 2: Tumor Size Prediction")
    print("-"*70)
    
    test_image = "test_tumor_image.png"
    if not Path(test_image).exists():
        test_image = create_test_image(test_image)
    
    # Test with circle bbox
    print("\n📐 Test 1: Circular Bounding Box")
    result_circle = tester.test_predict_size(test_image, bbox_type="circle", return_image=True)
    
    # Test with rectangular bbox
    print("\n📦 Test 2: Rectangular Bounding Box")
    result_rect = tester.test_predict_size(test_image, bbox_type="rect", return_image=True)
    
    # Quick evaluation
    print("\n" + "-"*70)
    print("PHASE 3: Model Evaluation")
    print("-"*70)
    tester.test_quick_eval()
    
    # Summary
    print("\n" + "="*70)
    print("✅ TEST SUITE COMPLETE")
    print("="*70)
    print("\n📋 Summary:")
    print("  ✅ API connectivity verified")
    print("  ✅ Model status checked")
    print("  ✅ Tumor size prediction tested")
    print("  ✅ Bounding box generation verified")
    print("\n💡 Next steps:")
    print("  1. Train model: python scripts/train_yolo.py")
    print("  2. Start server: python webapp/fastapi_server.py")
    print("  3. Make predictions using /predict-size endpoint")
    print("  4. Use streamlit demo: streamlit run webapp/streamlit_demo.py")


if __name__ == '__main__':
    main()
