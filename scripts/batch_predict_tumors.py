"""
Batch Processing Script for Tumor Size Predictions

Process multiple patients and generate comprehensive reports.

Usage:
    # Process all samples
    python scripts/batch_predict_tumors.py --csv merged_data.csv --output results/

    # Process subset with specific sequence
    python scripts/batch_predict_tumors.py --csv merged_data.csv --output results/ --limit 10

    # Save detailed report
    python scripts/batch_predict_tumors.py --csv merged_data.csv --output results/ --report-html
"""

import argparse
import sys
from pathlib import Path
import json
import csv
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.size_predictor_model import TumorSizePredictor
from src.bbox_utils import BoundingBoxGenerator, TumorPrediction
from src.prostate_dataset import ProstateLesionDataset
from src.utils_image import pil_to_tensor
from src import config


class BatchPredictor:
    """Batch prediction processor."""
    
    def __init__(self, model_path: str, device: str = 'cuda', output_dir: str = 'results'):
        """Initialize batch predictor.
        
        Args:
            model_path: Path to model weights
            device: 'cuda' or 'cpu'
            output_dir: Output directory for results
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = TumorSizePredictor(pretrained=False, in_channels=3)
        if Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.bbox_gen = BoundingBoxGenerator(pixel_spacing_mm=(1.0, 1.0))
        
        print(f"✓ Batch predictor initialized")
        print(f"  Device: {self.device}")
        print(f"  Output: {self.output_dir}")
    
    def predict_batch(self, csv_path: str, limit: int = None) -> List[Dict]:
        """Process dataset.
        
        Args:
            csv_path: Path to metadata CSV
            limit: Max samples to process (None = all)
            
        Returns:
            List of prediction dicts
        """
        print(f"\nLoading dataset...")
        dataset = ProstateLesionDataset(
            csv_path=csv_path,
            img_size=config.IMG_SIZE,
            sequences=['t2', 'adc', 'dwi'],
        )
        
        n_samples = min(limit, len(dataset)) if limit else len(dataset)
        print(f"Processing {n_samples}/{len(dataset)} samples...")
        
        results = []
        
        for i in tqdm(range(n_samples)):
            sample = dataset[i]
            uid = sample['uid']
            
            # Predict
            img_tensor = sample['img'].unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(img_tensor)
            
            size = output['size'][0].cpu().numpy()
            severity_probs = output['severity_probs'][0].cpu().numpy()
            confidence = output['confidence'][0, 0].item()
            
            severity_idx = np.argmax(severity_probs)
            severity = ['T1', 'T2', 'T3', 'T4'][severity_idx]
            
            # Create result
            result = {
                'uid': uid,
                'width_mm': float(size[0]),
                'height_mm': float(size[1]),
                'depth_mm': float(size[2]),
                'max_dimension_mm': float(np.max(size)),
                'severity': severity,
                'severity_T1': float(severity_probs[0]),
                'severity_T2': float(severity_probs[1]),
                'severity_T3': float(severity_probs[2]),
                'severity_T4': float(severity_probs[3]),
                'confidence': float(confidence),
                'processed_at': datetime.now().isoformat(),
            }
            
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], format: str = 'both'):
        """Save prediction results.
        
        Args:
            results: List of prediction dicts
            format: 'json', 'csv', or 'both'
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format in ['json', 'both']:
            json_path = self.output_dir / f'predictions_{timestamp}.json'
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Saved: {json_path}")
        
        if format in ['csv', 'both']:
            csv_path = self.output_dir / f'predictions_{timestamp}.csv'
            
            fieldnames = [
                'uid', 'width_mm', 'height_mm', 'depth_mm', 'max_dimension_mm',
                'severity', 'severity_T1', 'severity_T2', 'severity_T3', 'severity_T4',
                'confidence', 'processed_at'
            ]
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            print(f"✓ Saved: {csv_path}")
    
    def generate_report(self, results: List[Dict]) -> str:
        """Generate text report.
        
        Args:
            results: List of prediction dicts
            
        Returns:
            Report text
        """
        n = len(results)
        
        # Statistics
        sizes = [r['max_dimension_mm'] for r in results]
        severities = [r['severity'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        severity_counts = {
            'T1': severities.count('T1'),
            'T2': severities.count('T2'),
            'T3': severities.count('T3'),
            'T4': severities.count('T4'),
        }
        
        report = f"""
{'='*80}
TUMOR SIZE PREDICTION BATCH REPORT
{'='*80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

STATISTICS
----------
Total samples: {n}

Tumor Size Distribution:
  Mean: {np.mean(sizes):.2f} mm
  Median: {np.median(sizes):.2f} mm
  Std Dev: {np.std(sizes):.2f} mm
  Min: {np.min(sizes):.2f} mm
  Max: {np.max(sizes):.2f} mm

Severity Distribution:
  T1 (≤20mm): {severity_counts['T1']:3d} ({severity_counts['T1']/n*100:5.1f}%) 🟢
  T2 (20-40mm): {severity_counts['T2']:3d} ({severity_counts['T2']/n*100:5.1f}%) 🟡
  T3 (40-60mm): {severity_counts['T3']:3d} ({severity_counts['T3']/n*100:5.1f}%) 🟠
  T4 (>60mm): {severity_counts['T4']:3d} ({severity_counts['T4']/n*100:5.1f}%) 🔴

Prediction Confidence:
  Mean: {np.mean(confidences):.2%}
  Median: {np.median(confidences):.2%}
  Min: {np.min(confidences):.2%}
  Max: {np.max(confidences):.2%}

SAMPLE PREDICTIONS (First 10)
{'─'*80}
"""
        
        for result in results[:10]:
            report += f"""
{result['uid']}:
  Size: {result['width_mm']:.1f} x {result['height_mm']:.1f} x {result['depth_mm']:.1f} mm (max: {result['max_dimension_mm']:.1f} mm)
  Severity: {result['severity']}
  Confidence: {result['confidence']:.2%}
  Probs: T1={result['severity_T1']:.2%} T2={result['severity_T2']:.2%} T3={result['severity_T3']:.2%} T4={result['severity_T4']:.2%}
"""
        
        report += f"\n{'='*80}\n"
        
        return report
    
    def generate_html_report(self, results: List[Dict]) -> str:
        """Generate HTML report.
        
        Args:
            results: List of prediction dicts
            
        Returns:
            HTML string
        """
        sizes = [r['max_dimension_mm'] for r in results]
        severities = [r['severity'] for r in results]
        
        severity_counts = {
            'T1': severities.count('T1'),
            'T2': severities.count('T2'),
            'T3': severities.count('T3'),
            'T4': severities.count('T4'),
        }
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Tumor Size Prediction Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background-color: #007bff;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f9f9f9;
        }}
        .stat-box {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #f9f9f9;
            border-left: 4px solid #007bff;
        }}
        .severity-T1 {{ color: #28a745; font-weight: bold; }}
        .severity-T2 {{ color: #ffc107; font-weight: bold; }}
        .severity-T3 {{ color: #ff9800; font-weight: bold; }}
        .severity-T4 {{ color: #dc3545; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Tumor Size Prediction Batch Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Statistics</h2>
        <div class="stat-box">
            <strong>Total Samples:</strong> {len(results)}
        </div>
        <div class="stat-box">
            <strong>Mean Size:</strong> {np.mean(sizes):.2f} mm
        </div>
        <div class="stat-box">
            <strong>Size Range:</strong> {np.min(sizes):.2f} - {np.max(sizes):.2f} mm
        </div>
        
        <h2>Severity Distribution</h2>
        <table>
            <tr>
                <th>Severity</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
            <tr><td><span class="severity-T1">T1 (≤20mm)</span></td>
                <td>{severity_counts['T1']}</td>
                <td>{severity_counts['T1']/len(results)*100:.1f}%</td></tr>
            <tr><td><span class="severity-T2">T2 (20-40mm)</span></td>
                <td>{severity_counts['T2']}</td>
                <td>{severity_counts['T2']/len(results)*100:.1f}%</td></tr>
            <tr><td><span class="severity-T3">T3 (40-60mm)</span></td>
                <td>{severity_counts['T3']}</td>
                <td>{severity_counts['T3']/len(results)*100:.1f}%</td></tr>
            <tr><td><span class="severity-T4">T4 (>60mm)</span></td>
                <td>{severity_counts['T4']}</td>
                <td>{severity_counts['T4']/len(results)*100:.1f}%</td></tr>
        </table>
        
        <h2>Sample Predictions</h2>
        <table>
            <tr>
                <th>UID</th>
                <th>Size (mm)</th>
                <th>Severity</th>
                <th>Confidence</th>
            </tr>
"""
        
        for result in results[:50]:
            html += f"""            <tr>
                <td>{result['uid']}</td>
                <td>{result['max_dimension_mm']:.1f}</td>
                <td><span class="severity-{result['severity']}">{result['severity']}</span></td>
                <td>{result['confidence']:.2%}</td>
            </tr>
"""
        
        html += """        </table>
    </div>
</body>
</html>
"""
        return html


def main():
    parser = argparse.ArgumentParser(description='Batch Tumor Size Prediction')
    
    parser.add_argument('--csv', type=str, required=True,
                        help='Dataset CSV file')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--model-path', type=str,
                        default='models/tumor_size_predictor_best.pth',
                        help='Model weights path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--limit', type=int, default=None,
                        help='Max samples to process')
    parser.add_argument('--report-html', action='store_true',
                        help='Generate HTML report')
    parser.add_argument('--format', type=str, default='both',
                        choices=['json', 'csv', 'both'],
                        help='Output format')
    
    args = parser.parse_args()
    
    # Check CSV exists
    if not Path(args.csv).exists():
        print(f"❌ CSV not found: {args.csv}")
        return 1
    
    # Initialize and run
    predictor = BatchPredictor(args.model_path, device=args.device, output_dir=args.output)
    
    # Process
    results = predictor.predict_batch(args.csv, limit=args.limit)
    
    # Save results
    predictor.save_results(results, format=args.format)
    
    # Generate reports
    text_report = predictor.generate_report(results)
    print(text_report)
    
    # Save text report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = Path(args.output) / f'report_{timestamp}.txt'
    with open(report_path, 'w') as f:
        f.write(text_report)
    print(f"✓ Saved: {report_path}")
    
    # HTML report
    if args.report_html:
        html_report = predictor.generate_html_report(results)
        html_path = Path(args.output) / f'report_{timestamp}.html'
        with open(html_path, 'w') as f:
            f.write(html_report)
        print(f"✓ Saved: {html_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
