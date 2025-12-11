"""
Batch Processing Script for Multiple Tumor Cases
Processes multiple patients' MRI data and generates analysis report
"""
import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import time
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class BatchTumorAnalyzer:
    """Process multiple tumor cases in batch"""
    
    def __init__(self, output_dir: str = "batch_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.start_time = None
    
    def create_synthetic_case(self, patient_id: str, tumor_size_mm: float = None):
        """Create synthetic case for demo/testing"""
        if tumor_size_mm is None:
            tumor_size_mm = np.random.uniform(10, 60)
        
        size = 256
        
        # Generate synthetic images
        t2 = np.ones((size, size)) * 100
        adc = np.ones((size, size)) * 1200
        dwi = np.ones((size, size)) * 80
        
        # Add tumor
        cy, cx = 128, 128
        r = int(tumor_size_mm / 2)
        y, x = np.ogrid[:size, :size]
        tumor_mask = (x - cx)**2 + (y - cy)**2 <= r**2
        
        t2[tumor_mask] = 200
        adc[tumor_mask] = 600
        dwi[tumor_mask] = 180
        
        # Add noise
        t2 += np.random.normal(0, 10, t2.shape)
        adc += np.random.normal(0, 50, adc.shape)
        dwi += np.random.normal(0, 10, dwi.shape)
        
        return np.clip(t2, 0, 255), np.clip(adc, 0, 2000), np.clip(dwi, 0, 255)
    
    def analyze_case(self, patient_id: str, t2: np.ndarray, adc: np.ndarray, dwi: np.ndarray) -> Dict:
        """Analyze single case"""
        # Get tumor mask
        adc_norm = (adc - adc.min()) / (adc.max() - adc.min() + 1e-8)
        tumor_mask = adc_norm < 0.5
        
        # Extract features
        features = {
            "t2_mean": float(t2[tumor_mask].mean()),
            "adc_mean": float(adc[tumor_mask].mean()),
            "dwi_mean": float(dwi[tumor_mask].mean()),
            "tumor_pixels": int(tumor_mask.sum())
        }
        
        # Estimate size
        area_mm2 = tumor_mask.sum() * 0.64
        size_mm = 2 * np.sqrt(area_mm2 / np.pi)
        
        # Classify severity
        if size_mm <= 5:
            severity = "T1a"
        elif size_mm <= 10:
            severity = "T1b"
        elif size_mm <= 20:
            severity = "T2a"
        elif size_mm <= 35:
            severity = "T2b"
        elif size_mm <= 50:
            severity = "T2c"
        elif size_mm <= 70:
            severity = "T3a"
        elif size_mm <= 100:
            severity = "T3b"
        else:
            severity = "T4"
        
        # Detect bbox
        rows = np.any(tumor_mask, axis=1)
        cols = np.any(tumor_mask, axis=0)
        
        if rows.any() and cols.any():
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        else:
            bbox = None
        
        result = {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "tumor_size_mm": round(size_mm, 2),
            "severity_tnm": severity,
            "confidence": round(0.5 + (1 - abs(size_mm - 25) / 100), 3),
            "bounding_box": bbox,
            "features": features,
            "status": "success"
        }
        
        return result
    
    def process_batch(self, cases: List[Dict]) -> List[Dict]:
        """Process batch of cases"""
        self.start_time = time.time()
        self.results = []
        
        print("\n" + "=" * 70)
        print("BATCH TUMOR ANALYSIS")
        print("=" * 70)
        print(f"Processing {len(cases)} cases...\n")
        
        for i, case in enumerate(cases, 1):
            patient_id = case.get('patient_id', f'P{i:04d}')
            
            try:
                # Create or load case
                if case.get('synthetic', True):
                    t2, adc, dwi = self.create_synthetic_case(
                        patient_id,
                        case.get('tumor_size_mm')
                    )
                else:
                    # Load from file paths
                    import nibabel as nib
                    from PIL import Image
                    
                    def load_img(path):
                        if path.endswith('.nii') or path.endswith('.nii.gz'):
                            return nib.load(path).get_fdata()
                        else:
                            return np.array(Image.open(path))
                    
                    t2 = load_img(case['t2_path'])
                    adc = load_img(case['adc_path'])
                    dwi = load_img(case['dwi_path'])
                
                # Analyze
                result = self.analyze_case(patient_id, t2, adc, dwi)
                self.results.append(result)
                
                # Print progress
                status_symbol = "✓" if result['status'] == 'success' else "✗"
                print(f"[{i:3d}/{len(cases)}] {status_symbol} {patient_id:12s} | "
                      f"Size: {result['tumor_size_mm']:6.2f}mm | "
                      f"TNM: {result['severity_tnm']:3s} | "
                      f"Conf: {result['confidence']:.1%}")
                
            except Exception as e:
                result = {
                    "patient_id": patient_id,
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                self.results.append(result)
                print(f"[{i:3d}/{len(cases)}] ✗ {patient_id:12s} | ERROR: {str(e)}")
        
        elapsed = time.time() - self.start_time
        
        print("\n" + "=" * 70)
        print(f"BATCH PROCESSING COMPLETE")
        print(f"Total time: {elapsed:.2f}s ({elapsed/len(cases):.2f}s per case)")
        print("=" * 70)
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report"""
        successful = [r for r in self.results if r.get('status') == 'success']
        failed = [r for r in self.results if r.get('status') != 'success']
        
        report = "BATCH ANALYSIS REPORT\n"
        report += f"Generated: {datetime.now().isoformat()}\n"
        report += "=" * 70 + "\n\n"
        
        # Summary statistics
        report += "SUMMARY\n"
        report += "-" * 70 + "\n"
        report += f"Total cases: {len(self.results)}\n"
        report += f"Successful: {len(successful)}\n"
        report += f"Failed: {len(failed)}\n"
        
        if successful:
            sizes = [r['tumor_size_mm'] for r in successful]
            report += f"\nTumor Size Statistics (mm):\n"
            report += f"  Mean: {np.mean(sizes):.2f}\n"
            report += f"  Std: {np.std(sizes):.2f}\n"
            report += f"  Min: {np.min(sizes):.2f}\n"
            report += f"  Max: {np.max(sizes):.2f}\n"
            
            # Severity distribution
            severities = {}
            for r in successful:
                tnm = r['severity_tnm']
                severities[tnm] = severities.get(tnm, 0) + 1
            
            report += f"\nTNM Stage Distribution:\n"
            for stage in sorted(severities.keys()):
                count = severities[stage]
                pct = count / len(successful) * 100
                report += f"  {stage}: {count} ({pct:.1f}%)\n"
        
        # Detailed results
        report += "\n" + "=" * 70 + "\n"
        report += "DETAILED RESULTS\n"
        report += "=" * 70 + "\n\n"
        
        for result in successful:
            report += f"Patient: {result['patient_id']}\n"
            report += f"  Tumor Size: {result['tumor_size_mm']:.2f} mm\n"
            report += f"  TNM Stage: {result['severity_tnm']}\n"
            report += f"  Confidence: {result['confidence']:.1%}\n"
            if result.get('bounding_box'):
                bbox = result['bounding_box']
                report += f"  Bbox: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}\n"
            report += "\n"
        
        if failed:
            report += "\n" + "=" * 70 + "\n"
            report += "FAILED CASES\n"
            report += "=" * 70 + "\n\n"
            for result in failed:
                report += f"Patient: {result['patient_id']}\n"
                report += f"  Error: {result.get('error', 'Unknown error')}\n\n"
        
        return report
    
    def save_results(self):
        """Save results to files"""
        # JSON results
        json_file = self.output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Results saved to {json_file}")
        
        # CSV summary
        csv_file = self.output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        successful = [r for r in self.results if r.get('status') == 'success']
        if successful:
            df = pd.DataFrame([{
                'patient_id': r['patient_id'],
                'tumor_size_mm': r['tumor_size_mm'],
                'severity_tnm': r['severity_tnm'],
                'confidence': r['confidence'],
                'timestamp': r['timestamp']
            } for r in successful])
            df.to_csv(csv_file, index=False)
            print(f"✓ Summary saved to {csv_file}")
        
        # Text report
        report_file = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report = self.generate_report()
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"✓ Report saved to {report_file}")
        
        return json_file, csv_file, report_file


def main():
    """Run batch processing demo"""
    
    # Create sample cases
    cases = [
        {"patient_id": "P001", "synthetic": True, "tumor_size_mm": 8},
        {"patient_id": "P002", "synthetic": True, "tumor_size_mm": 15},
        {"patient_id": "P003", "synthetic": True, "tumor_size_mm": 25},
        {"patient_id": "P004", "synthetic": True, "tumor_size_mm": 40},
        {"patient_id": "P005", "synthetic": True, "tumor_size_mm": 55},
        {"patient_id": "P006", "synthetic": True},  # Random size
        {"patient_id": "P007", "synthetic": True},
        {"patient_id": "P008", "synthetic": True},
    ]
    
    # Run batch analysis
    analyzer = BatchTumorAnalyzer()
    results = analyzer.process_batch(cases)
    
    # Print report
    print("\n" + analyzer.generate_report())
    
    # Save results
    analyzer.save_results()


if __name__ == "__main__":
    main()
