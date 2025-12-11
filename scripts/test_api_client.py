"""
API Client for Tumor Size Prediction & Severity Analysis
Tests the FastAPI endpoints with sample data
"""

import requests
import json
import numpy as np
from pathlib import Path
import time
from datetime import datetime


class TumorPredictionClient:
    """Client for tumor prediction API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> dict:
        """Check if API is running"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.json() if response.status_code == 200 else {"status": "error"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def predict_size(self, dicom_paths: dict, patient_id: str = "test") -> dict:
        """
        Predict tumor size and severity
        
        Args:
            dicom_paths: Dict with 'T2', 'ADC', 'DWI' file paths
            patient_id: Patient identifier
        
        Returns:
            Prediction results
        """
        files = {}
        
        # Prepare files for upload
        for seq_type, path in dicom_paths.items():
            if Path(path).exists():
                files[seq_type] = open(path, 'rb')
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict-size",
                files=files,
                data={"patient_id": patient_id}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "error",
                    "code": response.status_code,
                    "message": response.text
                }
        finally:
            # Close files
            for f in files.values():
                f.close()
    
    def predict_size_json(self, sequences_data: dict, patient_id: str = "test") -> dict:
        """
        Predict using JSON data (for testing without files)
        """
        payload = {
            "patient_id": patient_id,
            "sequences": sequences_data
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict-size-json",
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "error",
                    "code": response.status_code,
                    "message": response.text
                }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_severity_info(self, size_mm: float) -> dict:
        """Get severity information for a given size"""
        try:
            response = self.session.get(
                f"{self.base_url}/severity-info",
                params={"size_mm": size_mm}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


def test_api_with_synthetic_data():
    """Test API with synthetic data"""
    print("\n" + "="*70)
    print("API CLIENT TEST - TUMOR SIZE PREDICTION")
    print("="*70)
    
    client = TumorPredictionClient()
    
    # Health check
    print("\n🏥 Health Check...")
    health = client.health_check()
    print(f"  Status: {health}")
    
    if health.get("status") == "error":
        print("\n❌ API is not running!")
        print("   Start the API with: python webapp/fastapi_server.py")
        return
    
    print("✓ API is running")
    
    # Test with synthetic data
    print("\n📊 Testing with synthetic data...")
    
    # Create synthetic sequences
    sequences = {
        'T2': {
            'data': np.random.randint(50, 200, (128, 128, 20)).tolist(),
            'shape': [128, 128, 20]
        },
        'ADC': {
            'data': np.random.randint(30, 150, (128, 128, 20)).tolist(),
            'shape': [128, 128, 20]
        },
        'DWI': {
            'data': np.random.randint(100, 250, (128, 128, 20)).tolist(),
            'shape': [128, 128, 20]
        }
    }
    
    # Add tumor signal
    size_mm = 25.0
    size_px = int(size_mm / 0.5)
    
    for seq_type in ['T2', 'ADC', 'DWI']:
        data = np.array(sequences[seq_type]['data'])
        # Add tumor blob at center
        center_y, center_x = 64, 64
        y_start = max(0, center_y - size_px)
        y_end = min(128, center_y + size_px)
        x_start = max(0, center_x - size_px)
        x_end = min(128, center_x + size_px)
        
        data[10:15, y_start:y_end, x_start:x_end] += 50
        sequences[seq_type]['data'] = data.tolist()
    
    # Make prediction
    print("  Sending prediction request...")
    start_time = time.time()
    
    result = client.predict_size_json(sequences, patient_id="synthetic_test")
    
    elapsed = time.time() - start_time
    
    # Display results
    print(f"\n✓ Prediction complete ({elapsed:.2f}s)")
    print("\n" + "-"*70)
    print("PREDICTION RESULTS")
    print("-"*70)
    
    if "status" in result and result["status"] == "error":
        print(f"❌ Error: {result.get('message', 'Unknown error')}")
        return
    
    # Parse results
    print(f"\n📏 Tumor Size: {result.get('tumor_size_mm', 'N/A'):.2f} mm")
    
    if 'bounding_box' in result:
        bbox = result['bounding_box']
        print(f"\n📦 Bounding Box (rounded):")
        print(f"   Y: [{bbox.get('y_min', 'N/A')}, {bbox.get('y_max', 'N/A')}]")
        print(f"   X: [{bbox.get('x_min', 'N/A')}, {bbox.get('x_max', 'N/A')}]")
        print(f"   Width: {bbox.get('width', 'N/A')}px, Height: {bbox.get('height', 'N/A')}px")
    
    if 'severity' in result:
        sev = result['severity']
        print(f"\n⚠️  Severity Classification:")
        print(f"   Class: {sev.get('class', 'N/A')}")
        print(f"   Risk Level: {sev.get('risk_level', 'N/A')}")
        print(f"   Score: {sev.get('score', 'N/A'):.3f}")
        print(f"   Confidence: {sev.get('confidence', 'N/A'):.1%}")
    
    print("\n" + "-"*70)
    
    # Test severity info
    print("\n🔍 Severity Information...")
    for size in [15, 20, 25, 30, 35, 40]:
        sev_info = client.get_severity_info(size)
        if "status" not in sev_info or sev_info["status"] != "error":
            print(f"  {size}mm: {sev_info.get('severity_class', 'N/A')} "
                  f"({sev_info.get('risk_level', 'N/A')})")
    
    print("\n✅ API test complete!\n")


def create_test_config():
    """Create configuration file for API testing"""
    config = {
        "api_base_url": "http://localhost:8000",
        "test_cases": [
            {
                "name": "Small tumor (low risk)",
                "size_mm": 10,
                "expected_severity": "T1a"
            },
            {
                "name": "Medium tumor (medium risk)",
                "size_mm": 25,
                "expected_severity": "T2"
            },
            {
                "name": "Large tumor (high risk)",
                "size_mm": 40,
                "expected_severity": "T3"
            }
        ],
        "timeout_seconds": 30,
        "retry_attempts": 3
    }
    
    config_file = Path(__file__).parent.parent / 'api_test_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📝 Test configuration saved to: {config_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Tumor Prediction API Client")
    parser.add_argument('--test', action='store_true', help='Run API test')
    parser.add_argument('--config', action='store_true', help='Create test configuration')
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    
    args = parser.parse_args()
    
    if args.config:
        create_test_config()
    elif args.test:
        test_api_with_synthetic_data()
    else:
        print("Usage: python test_api_client.py [--test] [--config] [--url URL]")
        print("\nExamples:")
        print("  python test_api_client.py --test")
        print("  python test_api_client.py --config")
        print("  python test_api_client.py --test --url http://localhost:8000")
