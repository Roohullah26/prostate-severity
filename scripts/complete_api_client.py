"""
Complete API Client for Tumor Size Prediction
Handles T2, ADC, DWI sequences and provides predictions with severity
"""
import requests
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import time

class TumorSizeAPIClient:
    """Complete API client for tumor size prediction"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def predict_tumor_size(
        self,
        t2_image: np.ndarray,
        adc_image: np.ndarray,
        dwi_image: np.ndarray,
        patient_id: str = "unknown"
    ) -> Dict:
        """
        Predict tumor size from multi-sequence MRI images
        
        Args:
            t2_image: T2-weighted image (numpy array)
            adc_image: ADC map (numpy array)
            dwi_image: DWI image (numpy array)
            patient_id: Patient identifier
            
        Returns:
            Dictionary with predictions including:
            - tumor_size: Predicted tumor size in mm
            - severity: TNM classification (T1, T2, T3, T4)
            - confidence: Confidence score
            - bounding_box: Detected bounding box coordinates
            - visualizations: Marked images
        """
        payload = {
            "t2_image": t2_image.tolist() if isinstance(t2_image, np.ndarray) else t2_image,
            "adc_image": adc_image.tolist() if isinstance(adc_image, np.ndarray) else adc_image,
            "dwi_image": dwi_image.tolist() if isinstance(dwi_image, np.ndarray) else dwi_image,
            "patient_id": patient_id
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict-size",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def get_system_status(self) -> Dict:
        """Get system status and available models"""
        try:
            response = self.session.get(
                f"{self.base_url}/status",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "offline"}
    
    def health_check(self) -> bool:
        """Check if API is running"""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False


class TumorAnalysisPipeline:
    """Complete pipeline for tumor analysis"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.client = TumorSizeAPIClient(base_url)
        self.results = []
    
    def load_images(self, t2_path: str, adc_path: str, dwi_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load images from file paths"""
        import nibabel as nib
        from PIL import Image
        
        def load_image(path):
            path = Path(path)
            if path.suffix == '.nii':
                img = nib.load(str(path)).get_fdata()
            elif path.suffix in ['.jpg', '.png']:
                img = np.array(Image.open(str(path)))
            elif path.suffix == '.npy':
                img = np.load(str(path))
            else:
                raise ValueError(f"Unsupported image format: {path.suffix}")
            return img
        
        t2 = load_image(t2_path)
        adc = load_image(adc_path)
        dwi = load_image(dwi_path)
        
        return t2, adc, dwi
    
    def analyze_case(
        self,
        t2_path: str,
        adc_path: str,
        dwi_path: str,
        patient_id: str
    ) -> Dict:
        """Analyze a single case"""
        try:
            # Load images
            t2, adc, dwi = self.load_images(t2_path, adc_path, dwi_path)
            
            # Predict
            result = self.client.predict_tumor_size(t2, adc, dwi, patient_id)
            result['patient_id'] = patient_id
            result['timestamp'] = time.time()
            
            self.results.append(result)
            return result
        except Exception as e:
            return {
                "patient_id": patient_id,
                "error": str(e),
                "status": "failed"
            }
    
    def batch_analyze(self, cases: List[Dict]) -> List[Dict]:
        """Analyze multiple cases"""
        results = []
        for i, case in enumerate(cases, 1):
            print(f"Processing case {i}/{len(cases)}: {case.get('patient_id', 'unknown')}")
            result = self.analyze_case(
                case['t2_path'],
                case['adc_path'],
                case['dwi_path'],
                case.get('patient_id', f'case_{i}')
            )
            results.append(result)
        return results
    
    def generate_report(self) -> str:
        """Generate analysis report"""
        report = "TUMOR SIZE ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        successful = [r for r in self.results if 'error' not in r]
        failed = [r for r in self.results if 'error' in r]
        
        report += f"Total Cases: {len(self.results)}\n"
        report += f"Successful: {len(successful)}\n"
        report += f"Failed: {len(failed)}\n\n"
        
        report += "CASE RESULTS:\n"
        report += "-" * 60 + "\n"
        
        for result in successful:
            report += f"\nPatient ID: {result.get('patient_id', 'unknown')}\n"
            report += f"  Tumor Size: {result.get('tumor_size', 'N/A'):.2f} mm\n"
            report += f"  Severity (TNM): {result.get('severity', 'N/A')}\n"
            report += f"  Confidence: {result.get('confidence', 'N/A'):.2%}\n"
            if 'bounding_box' in result:
                bbox = result['bounding_box']
                report += f"  Bounding Box: x={bbox[0]:.0f}, y={bbox[1]:.0f}, w={bbox[2]:.0f}, h={bbox[3]:.0f}\n"
        
        if failed:
            report += "\n\nFAILED CASES:\n"
            report += "-" * 60 + "\n"
            for result in failed:
                report += f"\nPatient ID: {result.get('patient_id', 'unknown')}\n"
                report += f"  Error: {result.get('error', 'Unknown error')}\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    print("Tumor Size API Client")
    print("=" * 60)
    
    # Initialize client
    client = TumorSizeAPIClient()
    
    # Check API health
    print("\nChecking API status...")
    if client.health_check():
        print("✓ API is running")
        
        # Get system status
        status = client.get_system_status()
        print(f"\nSystem Status:")
        print(json.dumps(status, indent=2))
    else:
        print("✗ API is not running. Start it with: python -m uvicorn webapp.fastapi_server:app --reload")
    
    print("\n\nAPI Client ready. Use like this:")
    print("-" * 60)
    print("""
# Create client
client = TumorSizeAPIClient("http://localhost:8000")

# Load images
import numpy as np
t2 = np.load("path/to/t2.npy")
adc = np.load("path/to/adc.npy")
dwi = np.load("path/to/dwi.npy")

# Get prediction
result = client.predict_tumor_size(t2, adc, dwi, patient_id="P001")

# Check results
print(f"Tumor Size: {result['tumor_size']} mm")
print(f"Severity: {result['severity']}")
print(f"Confidence: {result['confidence']}")
print(f"Bounding Box: {result['bounding_box']}")
""")
