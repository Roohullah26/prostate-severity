#!/usr/bin/env python3
"""
Batch Tumor Size Prediction
Process multiple cases efficiently
"""

import os
import sys
import csv
import glob
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from size_predictor_model import TumorSizePredictorMultiSeq
from bbox_utils import generate_bounding_box, apply_bbox_to_image
from utils_image import load_dicom_series


class BatchTumorSizePredictor:
    """Batch processor for tumor size prediction"""
    
    def __init__(self, model_path=None):
        """Initialize predictor"""
        self.model = TumorSizePredictorMultiSeq(model_path=model_path)
        self.results = []
    
    def classify_severity(self, tumor_size_mm):
        """Classify severity based on size"""
        if tumor_size_mm < 10:
            return "T1a", "Small (<10mm)"
        elif tumor_size_mm < 20:
            return "T1b", "Medium (10-20mm)"
        elif tumor_size_mm < 30:
            return "T1c", "Medium-Large (20-30mm)"
        elif tumor_size_mm < 50:
            return "T2", "Large (30-50mm)"
        else:
            return "T3+", "Very Large (>50mm)"
    
    def load_image(self, path):
        """Load image from file or directory"""
        if os.path.isdir(path):
            return load_dicom_series(path)
        else:
            return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    
    def predict_single(self, case_id, t2_path, adc_path, dwi_path, 
                       output_dir=None, save_viz=False):
        """Predict tumor size for single case"""
        
        try:
            # Load images
            t2_img = self.load_image(t2_path)
            adc_img = self.load_image(adc_path)
            dwi_img = self.load_image(dwi_path)
            
            if t2_img is None or adc_img is None or dwi_img is None:
                return {
                    'case_id': case_id,
                    'status': 'FAILED',
                    'error': 'Failed to load images'
                }
            
            # Prepare input
            input_data = np.stack([t2_img, adc_img, dwi_img], axis=0)
            input_data = np.expand_dims(input_data, axis=0)
            
            # Predict
            tumor_size_mm = self.model.predict(input_data)
            
            # Generate bbox
            bbox = generate_bounding_box(dwi_img, tumor_size_mm)
            
            # Classify severity
            tnm_stage, severity = self.classify_severity(tumor_size_mm)
            
            result = {
                'case_id': case_id,
                'status': 'SUCCESS',
                'tumor_size_mm': float(tumor_size_mm),
                'tnm_stage': tnm_stage,
                'severity': severity,
                'bbox': bbox,
                't2_shape': str(t2_img.shape),
                'adc_shape': str(adc_img.shape),
                'dwi_shape': str(dwi_img.shape)
            }
            
            # Save visualizations
            if save_viz and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                # Save bbox visualization
                dwi_with_bbox = apply_bbox_to_image(dwi_img, bbox)
                viz_path = os.path.join(output_dir, f"{case_id}_bbox.png")
                cv2.imwrite(viz_path, dwi_with_bbox)
                
                # Save composite
                h, w = dwi_img.shape
                composite = np.zeros((h, w*3), dtype=np.uint8)
                composite[:, :w] = t2_img
                composite[:, w:w*2] = adc_img
                composite[:, w*2:] = dwi_with_bbox
                
                composite_path = os.path.join(output_dir, f"{case_id}_composite.png")
                cv2.imwrite(composite_path, composite)
            
            return result
            
        except Exception as e:
            return {
                'case_id': case_id,
                'status': 'FAILED',
                'error': str(e)
            }
    
    def predict_batch_from_directory(self, data_dir, output_dir=None, 
                                     save_viz=False, t2_pattern="t2",
                                     adc_pattern="adc", dwi_pattern="dwi"):
        """
        Predict for all cases in directory
        
        Directory structure expected:
        data_dir/
            case_001/
                t2.png (or t2_*.dcm)
                adc.png
                dwi.png
            case_002/
                ...
        """
        
        print(f"\nBatch Processing: {data_dir}")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        case_dirs = sorted(glob.glob(os.path.join(data_dir, "*/")))
        
        self.results = []
        
        for case_dir in tqdm(case_dirs, desc="Processing"):
            case_id = os.path.basename(case_dir.rstrip('/'))
            
            # Find image files
            t2_files = glob.glob(os.path.join(case_dir, f"*{t2_pattern}*"))
            adc_files = glob.glob(os.path.join(case_dir, f"*{adc_pattern}*"))
            dwi_files = glob.glob(os.path.join(case_dir, f"*{dwi_pattern}*"))
            
            if not (t2_files and adc_files and dwi_files):
                self.results.append({
                    'case_id': case_id,
                    'status': 'SKIPPED',
                    'error': 'Missing image files'
                })
                continue
            
            result = self.predict_single(
                case_id,
                t2_files[0],
                adc_files[0],
                dwi_files[0],
                output_dir=output_dir if save_viz else None,
                save_viz=save_viz
            )
            
            self.results.append(result)
        
        return self.results
    
    def predict_batch_from_csv(self, csv_path, output_dir=None, save_viz=False):
        """
        Predict from CSV file
        
        CSV format:
        case_id,t2_path,adc_path,dwi_path
        case_001,data/case_001/t2.png,data/case_001/adc.png,data/case_001/dwi.png
        ...
        """
        
        print(f"\nBatch Processing from CSV: {csv_path}")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            cases = list(reader)
        
        for row in tqdm(cases, desc="Processing"):
            result = self.predict_single(
                row['case_id'],
                row['t2_path'],
                row['adc_path'],
                row['dwi_path'],
                output_dir=output_dir if save_viz else None,
                save_viz=save_viz
            )
            
            self.results.append(result)
        
        return self.results
    
    def save_results_csv(self, output_path="predictions.csv"):
        """Save results to CSV"""
        
        if not self.results:
            print("No results to save")
            return
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Get all unique keys
        keys = set()
        for r in self.results:
            keys.update(r.keys())
        
        keys = sorted(list(keys))
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"\n✓ Results saved: {output_path}")
    
    def print_summary(self):
        """Print results summary"""
        
        if not self.results:
            print("No results to summarize")
            return
        
        successful = [r for r in self.results if r['status'] == 'SUCCESS']
        failed = [r for r in self.results if r['status'] == 'FAILED']
        skipped = [r for r in self.results if r['status'] == 'SKIPPED']
        
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total Cases:      {len(self.results)}")
        print(f"Successful:       {len(successful)}")
        print(f"Failed:           {len(failed)}")
        print(f"Skipped:          {len(skipped)}")
        print("="*60)
        
        if successful:
            sizes = [r['tumor_size_mm'] for r in successful]
            print(f"\nTumor Size Statistics (mm):")
            print(f"  Mean:           {np.mean(sizes):.2f}")
            print(f"  Std:            {np.std(sizes):.2f}")
            print(f"  Min:            {np.min(sizes):.2f}")
            print(f"  Max:            {np.max(sizes):.2f}")
            print(f"  Median:         {np.median(sizes):.2f}")
            
            # TNM distribution
            tnm_counts = {}
            for r in successful:
                tnm = r['tnm_stage']
                tnm_counts[tnm] = tnm_counts.get(tnm, 0) + 1
            
            print(f"\nTNM Stage Distribution:")
            for tnm, count in sorted(tnm_counts.items()):
                pct = (count / len(successful)) * 100
                print(f"  {tnm:4s}: {count:3d} ({pct:5.1f}%)")
        
        if failed:
            print(f"\nFailed Cases:")
            for r in failed[:5]:  # Show first 5
                print(f"  {r['case_id']}: {r['error']}")
            if len(failed) > 5:
                print(f"  ... and {len(failed)-5} more")
        
        print("="*60 + "\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch tumor size prediction")
    parser.add_argument("--input-dir", type=str, help="Input directory with cases")
    parser.add_argument("--input-csv", type=str, help="Input CSV file")
    parser.add_argument("--output-dir", type=str, default="batch_output",
                        help="Output directory")
    parser.add_argument("--save-viz", action="store_true", help="Save visualizations")
    parser.add_argument("--model-path", type=str, help="Path to model file")
    parser.add_argument("--results-csv", type=str, default="predictions.csv",
                        help="Output CSV file")
    
    args = parser.parse_args()
    
    if not args.input_dir and not args.input_csv:
        parser.print_help()
        sys.exit(1)
    
    # Initialize predictor
    predictor = BatchTumorSizePredictor(model_path=args.model_path)
    
    # Run batch processing
    if args.input_dir:
        predictor.predict_batch_from_directory(
            args.input_dir,
            output_dir=args.output_dir,
            save_viz=args.save_viz
        )
    elif args.input_csv:
        predictor.predict_batch_from_csv(
            args.input_csv,
            output_dir=args.output_dir,
            save_viz=args.save_viz
        )
    
    # Save results
    predictor.save_results_csv(args.results_csv)
    predictor.print_summary()


if __name__ == "__main__":
    main()
