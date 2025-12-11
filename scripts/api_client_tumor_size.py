#!/usr/bin/env python3
"""
API Client for Tumor Size Prediction
"""

import requests
import json
import base64
import cv2
import numpy as np
from pathlib import Path


class TumorSizePredictionClient:
    """Client for tumor size prediction API"""
    
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.session = requests.Session()
    
    def predict_from_files(self, t2_path, adc_path, dwi_path):
        """
        Predict tumor size from image files
        """
        print("\n[API Client] Loading images...")
        
        # Load images
        t2_img = cv2.imread(str(t2_path), cv2.IMREAD_GRAYSCALE)
        adc_img = cv2.imread(str(adc_path), cv2.IMREAD_GRAYSCALE)
        dwi_img = cv2.imread(str(dwi_path), cv2.IMREAD_GRAYSCALE)
        
        # Encode to base64
        print("[API Client] Encoding images...")
        t2_b64 = base64.b64encode(cv2.imencode('.png', t2_img)[1]).decode()
        adc_b64 = base64.b64encode(cv2.imencode('.png', adc_img)[1]).decode()
        dwi_b64 = base64.b64encode(cv2.imencode('.png', dwi_img)[1]).decode()
        
        # Prepare request
        payload = {
            "t2_image_base64": t2_b64,
            "adc_image_base64": adc_b64,
            "dwi_image_base64": dwi_b64
        }
        
        print("[API Client] Sending request to API...")
        endpoint = f"{self.api_url}/predict-size"
        
        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            
            print("[API Client] ✓ Response received")
            return result
            
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to {endpoint}")
            print("   Make sure the FastAPI server is running:")
            print("   python -m uvicorn webapp.fastapi_server:app --reload")
            return None
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
    
    def predict_from_arrays(self, t2_array, adc_array, dwi_array):
        """
        Predict tumor size from numpy arrays
        """
        print("[API Client] Encoding image arrays...")
        
        # Normalize to 0-255 range if needed
        t2_array = (t2_array / t2_array.max() * 255).astype(np.uint8) if t2_array.max() <= 1 else t2_array.astype(np.uint8)
        adc_array = (adc_array / adc_array.max() * 255).astype(np.uint8) if adc_array.max() <= 1 else adc_array.astype(np.uint8)
        dwi_array = (dwi_array / dwi_array.max() * 255).astype(np.uint8) if dwi_array.max() <= 1 else dwi_array.astype(np.uint8)
        
        # Encode to base64
        t2_b64 = base64.b64encode(cv2.imencode('.png', t2_array)[1]).decode()
        adc_b64 = base64.b64encode(cv2.imencode('.png', adc_array)[1]).decode()
        dwi_b64 = base64.b64encode(cv2.imencode('.png', dwi_array)[1]).decode()
        
        # Prepare request
        payload = {
            "t2_image_base64": t2_b64,
            "adc_image_base64": adc_b64,
            "dwi_image_base64": dwi_b64
        }
        
        print("[API Client] Sending request to API...")
        endpoint = f"{self.api_url}/predict-size"
        
        try:
            response = self.session.post(endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            
            print("[API Client] ✓ Response received")
            return result
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
    
    def health_check(self):
        """Check if API is running"""
        try:
            response = self.session.get(f"{self.api_url}/health")
            response.raise_for_status()
            return True
        except:
            return False


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python api_client_tumor_size.py <t2_path> <adc_path> <dwi_path>")
        sys.exit(1)
    
    t2_path = sys.argv[1]
    adc_path = sys.argv[2]
    dwi_path = sys.argv[3]
    
    client = TumorSizePredictionClient()
    
    # Check health
    print("[API Client] Checking API health...")
    if not client.health_check():
        print("❌ API is not running!")
        print("Start the server with: python -m uvicorn webapp.fastapi_server:app --reload")
        sys.exit(1)
    
    print("✓ API is healthy")
    
    # Make prediction
    result = client.predict_from_files(t2_path, adc_path, dwi_path)
    
    if result:
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(json.dumps(result, indent=2))
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
