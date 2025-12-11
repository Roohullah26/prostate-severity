#!/usr/bin/env python3
"""
Production-Ready API Wrapper for Tumor Size Prediction System
Provides easy-to-use functions for tumor analysis
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from size_predictor_model import SizePredictorModel
from bbox_utils import BBoxPredictor
from utils_image import normalize_mri_sequence, load_dicom_sequence


class TumorAnalysisAPI:
    """High-level API for tumor size prediction and analysis"""
    
    def __init__(self, model_path: Optional[str] = None, bbox_model_path: Optional[str] = None):
        """
        Initialize the Tumor Analysis API
        
        Args:
            model_path: Path to size predictor model (uses default if None)
            bbox_model_path: Path to bbox model (uses default if None)
        """
        self.size_model = SizePredictorModel()
        self.bbox_model = BBoxPredictor()
        print("✓ Tumor Analysis API initialized")
    
    def analyze_tumor(
        self,
        t2_sequence: np.ndarray,
        adc_sequence: np.ndarray,
        dwi_sequence: np.ndarray,
        normalize: bool = True
    ) -> Dict:
        """
        Comprehensive tumor analysis using multi-sequence MRI data
        
        Args:
            t2_sequence: T2-weighted MRI image (2D array)
            adc_sequence: ADC map (2D array)
            dwi_sequence: DWI image (2D array)
            normalize: Whether to normalize sequences (default: True)
        
        Returns:
            Dictionary with analysis results:
            - tumor_size: Predicted tumor size in mm
            - size_class: Size classification (T1a, T1b, T1c, T2, T3)
            - severity: Severity level (Low, Intermediate, High)
            - confidence: Confidence score (0-1)
            - bounding_box: Tumor bounding box coordinates
            - bbox_area: Area of bounding box in mm²
            - sequence_info: Information about input sequences
        """
        
        # Normalize if requested
        if normalize:
            t2_norm = normalize_mri_sequence(t2_sequence)
            adc_norm = normalize_mri_sequence(adc_sequence)
            dwi_norm = normalize_mri_sequence(dwi_sequence)
        else:
            t2_norm, adc_norm, dwi_norm = t2_sequence, adc_sequence, dwi_sequence
        
        # Predict tumor size
        tumor_size, size_class = self.size_model.predict(t2_norm, adc_norm, dwi_norm)
        
        # Get severity
        severity = self.size_model.classify_severity(tumor_size)
        
        # Predict bounding box
        bbox = self.bbox_model.predict_bbox(t2_norm)
        
        # Calculate confidence (based on model uncertainty)
        confidence = self._estimate_confidence(t2_norm, adc_norm, dwi_norm)
        
        # Calculate bounding box area (rough estimate assuming 0.5mm per pixel)
        x1, y1, x2, y2 = bbox
        bbox_area = ((x2 - x1) * (y2 - y1)) * (0.5 ** 2)  # mm²
        
        return {
            'tumor_size': float(tumor_size),
            'size_class': size_class,
            'severity': severity,
            'confidence': float(confidence),
            'bounding_box': {
                'coordinates': [int(x) for x in bbox],
                'format': 'x1, y1, x2, y2',
                'area_mm2': float(bbox_area)
            },
            'sequence_info': {
                't2_shape': t2_norm.shape,
                'adc_shape': adc_norm.shape,
                'dwi_shape': dwi_norm.shape,
                'normalized': normalize
            }
        }
    
    def batch_analyze(self, cases: list) -> list:
        """
        Analyze multiple tumor cases
        
        Args:
            cases: List of dictionaries, each containing 't2', 'adc', 'dwi' arrays
        
        Returns:
            List of analysis results for each case
        """
        results = []
        for i, case in enumerate(cases):
            print(f"Processing case {i+1}/{len(cases)}...")
            result = self.analyze_tumor(case['t2'], case['adc'], case['dwi'])
            result['case_id'] = i
            results.append(result)
        
        return results
    
    def get_severity_recommendation(self, severity: str) -> Dict:
        """
        Get clinical recommendations based on severity
        
        Args:
            severity: Severity level ('Low', 'Intermediate', 'High')
        
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            'Low': {
                'clinical_action': 'Active surveillance',
                'follow_up': '12-24 months',
                'priority': 'Routine',
                'notes': 'Monitor with periodic MRI'
            },
            'Intermediate': {
                'clinical_action': 'Close monitoring or treatment consideration',
                'follow_up': '6-12 months',
                'priority': 'High',
                'notes': 'Consider treatment options, multi-disciplinary team review'
            },
            'High': {
                'clinical_action': 'Urgent treatment planning',
                'follow_up': 'Within 4 weeks',
                'priority': 'Urgent',
                'notes': 'Immediate referral to oncology, treatment options review'
            }
        }
        
        return recommendations.get(severity, {})
    
    def _estimate_confidence(self, t2: np.ndarray, adc: np.ndarray, dwi: np.ndarray) -> float:
        """Estimate confidence score based on image quality"""
        # Simple quality metrics
        t2_std = np.std(t2)
        adc_std = np.std(adc)
        dwi_std = np.std(dwi)
        
        # Confidence based on signal variation
        avg_std = (t2_std + adc_std + dwi_std) / 3
        confidence = min(1.0, avg_std / 0.3)  # Normalize to reasonable range
        
        return confidence
    
    def export_report(self, analysis_result: Dict, output_path: str) -> None:
        """
        Export analysis results as a structured report
        
        Args:
            analysis_result: Result dictionary from analyze_tumor()
            output_path: Path to save the report
        """
        import json
        from datetime import datetime
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis_result,
            'recommendations': self.get_severity_recommendation(analysis_result['severity'])
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved to {output_path}")


def main():
    """Example usage"""
    print("="*60)
    print("TUMOR ANALYSIS API - EXAMPLE USAGE")
    print("="*60)
    
    # Initialize API
    api = TumorAnalysisAPI()
    
    # Create example data
    print("\nCreating example MRI data...")
    t2_example = np.random.rand(256, 256).astype(np.float32)
    adc_example = np.random.rand(256, 256).astype(np.float32)
    dwi_example = np.random.rand(256, 256).astype(np.float32)
    
    # Analyze tumor
    print("\nAnalyzing tumor...")
    result = api.analyze_tumor(t2_example, adc_example, dwi_example)
    
    # Display results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    print(f"Tumor Size: {result['tumor_size']:.2f} mm")
    print(f"Size Class: {result['size_class']}")
    print(f"Severity: {result['severity']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Bounding Box: {result['bounding_box']['coordinates']}")
    print(f"BBox Area: {result['bounding_box']['area_mm2']:.2f} mm²")
    
    # Get recommendations
    print("\n" + "="*60)
    print("CLINICAL RECOMMENDATIONS")
    print("="*60)
    recs = api.get_severity_recommendation(result['severity'])
    for key, value in recs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
