"""
Complete Tumor Size & Severity Prediction Demo

This script demonstrates the full pipeline:
1. Load multi-sequence MRI data (T2, ADC, DWI)
2. Predict tumor dimensions in mm
3. Generate rectangular and circular bounding boxes
4. Classify tumor severity (T1/T2/T3/T4)
5. Visualize predictions on MRI images

Usage:
    python scripts/demo_complete_pipeline.py --sample 0
    python scripts/demo_complete_pipeline.py --csv merged_data.csv --uid ProstateX-0000
    python scripts/demo_complete_pipeline.py --toy  # Demo with synthetic data
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from typing import Dict, Tuple, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.size_predictor_model import TumorSizePredictor
from src.bbox_utils import TumorPrediction, BoundingBoxGenerator, VisualizationHelper
from src.prostate_dataset import ProstateLesionDataset
from src.utils_image import pil_to_tensor
from src import config


class CompletePipeline:
    """Complete prediction and visualization pipeline."""
    
    # Severity color mapping for visualization
    SEVERITY_COLORS = {
        'T1': (0, 255, 0),      # Green - benign
        'T2': (255, 255, 0),    # Yellow - intermediate
        'T3': (255, 165, 0),    # Orange - significant
        'T4': (255, 0, 0),      # Red - severe
    }
    
    # Severity descriptions
    SEVERITY_DESCRIPTIONS = {
        'T1': 'Small tumor (≤20mm) - Low risk',
        'T2': 'Intermediate tumor (20-40mm) - Moderate risk',
        'T3': 'Large tumor (40-60mm) - High risk',
        'T4': 'Very large tumor (>60mm) - Very high risk',
    }
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """Initialize pipeline.
        
        Args:
            model_path: Path to trained model weights
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = TumorSizePredictor(pretrained=False, in_channels=3)
        
        # Try to load weights if they exist
        model_file = Path(model_path)
        if model_file.exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("✓ Model weights loaded")
        else:
            print(f"⚠ Model weights not found at {model_path} - using untrained model")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize utilities
        self.bbox_gen = BoundingBoxGenerator(pixel_spacing_mm=(1.0, 1.0))
        self.vis_helper = VisualizationHelper()
    
    def predict(self, image_tensor: torch.Tensor) -> Dict:
        """Run prediction on a tensor.
        
        Args:
            image_tensor: (B, 3, 224, 224) tensor
            
        Returns:
            Dict with predictions
        """
        with torch.no_grad():
            output = self.model(image_tensor.to(self.device))
        
        # Extract batch (use first element if batch)
        size = output['size'][0].cpu().numpy()
        severity_probs = output['severity_probs'][0].cpu().numpy()
        confidence = output['confidence'][0, 0].item()
        
        # Determine severity
        severity_idx = np.argmax(severity_probs)
        severity_grades = ['T1', 'T2', 'T3', 'T4']
        severity = severity_grades[severity_idx]
        
        return {
            'width_mm': float(size[0]),
            'height_mm': float(size[1]),
            'depth_mm': float(size[2]),
            'max_dimension_mm': float(np.max(size)),
            'severity': severity,
            'severity_probs': {
                'T1': float(severity_probs[0]),
                'T2': float(severity_probs[1]),
                'T3': float(severity_probs[2]),
                'T4': float(severity_probs[3]),
            },
            'confidence': float(confidence),
        }
    
    def visualize_prediction(self, image: Image.Image, prediction: Dict,
                            uid: str = "Unknown") -> Image.Image:
        """Create visualization with bounding boxes and severity info.
        
        Args:
            image: PIL Image (224x224)
            prediction: Prediction dict from predict()
            uid: Patient ID for title
            
        Returns:
            Annotated PIL Image
        """
        # Create a copy for drawing
        vis_img = image.copy()
        draw = ImageDraw.Draw(vis_img)
        
        # Get severity color
        severity = prediction['severity']
        color = self.SEVERITY_COLORS[severity]
        rgb_color = tuple(color[::-1])  # BGR to RGB for PIL
        
        # Get bounding boxes
        tumor_pred = TumorPrediction(
            width_mm=prediction['width_mm'],
            height_mm=prediction['height_mm'],
            depth_mm=prediction['depth_mm'],
            severity=severity,
            severity_logits=np.array(list(prediction['severity_probs'].values())),
            confidence=prediction['confidence'],
            image_size=(224, 224),
        )
        
        # Rectangular bbox
        rect_bbox = self.bbox_gen.get_rectangular_bbox(tumor_pred, center_x=112, center_y=112)
        x1, y1, x2, y2 = rect_bbox['x1'], rect_bbox['y1'], rect_bbox['x2'], rect_bbox['y2']
        draw.rectangle([x1, y1, x2, y2], outline=rgb_color, width=3)
        
        # Circular bbox
        circ_bbox = self.bbox_gen.get_circular_bbox(tumor_pred, center_x=112, center_y=112)
        cx, cy = circ_bbox['center_x'], circ_bbox['center_y']
        r = circ_bbox['radius_px']
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=rgb_color, width=2)
        
        # Draw center point
        draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=rgb_color, outline=rgb_color)
        
        # Add text annotations
        info_lines = [
            f"Patient: {uid}",
            f"Severity: {severity} - {self.SEVERITY_DESCRIPTIONS[severity]}",
            f"Size: W={prediction['width_mm']:.1f}mm, H={prediction['height_mm']:.1f}mm, D={prediction['depth_mm']:.1f}mm",
            f"Max Dimension: {prediction['max_dimension_mm']:.1f}mm",
            f"Confidence: {prediction['confidence']:.2%}",
            f"Severity Probs: T1={prediction['severity_probs']['T1']:.2%}, T2={prediction['severity_probs']['T2']:.2%}, T3={prediction['severity_probs']['T3']:.2%}, T4={prediction['severity_probs']['T4']:.2%}",
        ]
        
        # Draw text with background
        y_offset = 5
        for line in info_lines:
            bbox = draw.textbbox((0, 0), line)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw background
            draw.rectangle(
                [5, y_offset, 5 + text_width + 4, y_offset + text_height + 4],
                fill=(0, 0, 0)
            )
            # Draw text
            draw.text((7, y_offset + 2), line, fill=(255, 255, 255))
            y_offset += text_height + 6
        
        return vis_img
    
    def demo_synthetic(self):
        """Run demo with synthetic data."""
        print("\n" + "="*80)
        print("SYNTHETIC DATA DEMO")
        print("="*80)
        
        # Create synthetic multi-channel image (simulating T2/ADC/DWI stack)
        print("\nGenerating synthetic MRI data (T2/ADC/DWI stack)...")
        synthetic = torch.randn(1, 3, 224, 224)
        
        # Predict
        print("Running prediction...")
        prediction = self.predict(synthetic)
        
        # Print results
        self._print_results(prediction, uid="SYNTHETIC")
        
        # Create visualization
        img = Image.new('RGB', (224, 224), color='black')
        img_array = np.random.randint(50, 150, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        vis_img = self.visualize_prediction(img, prediction, uid="SYNTHETIC-DATA")
        vis_img.save("synthetic_prediction.png")
        print("✓ Saved: synthetic_prediction.png")
    
    def demo_from_dataset(self, csv_path: str, sample_idx: int = 0, uid: str = None):
        """Run demo with real dataset.
        
        Args:
            csv_path: Path to metadata CSV
            sample_idx: Sample index to process
            uid: Specific UID to process (overrides sample_idx)
        """
        print("\n" + "="*80)
        print("DATASET DEMO")
        print("="*80)
        
        # Load dataset
        print(f"Loading dataset from {csv_path}...")
        dataset = ProstateLesionDataset(
            csv_path=csv_path,
            img_size=config.IMG_SIZE,
            sequences=['t2', 'adc', 'dwi'],
        )
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Get sample
        if uid:
            # Find by UID
            for i, sample in enumerate(dataset):
                if sample['uid'] == uid:
                    sample_idx = i
                    break
        
        print(f"Processing sample {sample_idx}...")
        sample = dataset[sample_idx]
        
        # Predict
        print("Running prediction...")
        image_tensor = sample['img'].unsqueeze(0)
        prediction = self.predict(image_tensor)
        
        # Print results
        self._print_results(prediction, uid=sample['uid'])
        
        # Visualize
        print("Creating visualization...")
        pil_image = image_tensor[0, :3].permute(1, 2, 0).numpy()  # Convert to PIL format
        pil_image = ((pil_image - pil_image.min()) / (pil_image.max() - pil_image.min()) * 255).astype(np.uint8)
        img = Image.fromarray(pil_image)
        
        vis_img = self.visualize_prediction(img, prediction, uid=sample['uid'])
        
        output_path = f"prediction_{sample['uid']}.png"
        vis_img.save(output_path)
        print(f"✓ Saved: {output_path}")
        
        # Save JSON results
        json_path = output_path.replace('.png', '.json')
        with open(json_path, 'w') as f:
            json.dump(prediction, f, indent=2)
        print(f"✓ Saved: {json_path}")
    
    def _print_results(self, prediction: Dict, uid: str = "Unknown"):
        """Pretty print prediction results."""
        print(f"\n{'─'*80}")
        print(f"PREDICTION RESULTS - {uid}")
        print(f"{'─'*80}")
        print(f"\n📏 DIMENSIONS:")
        print(f"   Width:  {prediction['width_mm']:.2f} mm")
        print(f"   Height: {prediction['height_mm']:.2f} mm")
        print(f"   Depth:  {prediction['depth_mm']:.2f} mm")
        print(f"   Max:    {prediction['max_dimension_mm']:.2f} mm")
        
        print(f"\n🔴 SEVERITY CLASSIFICATION:")
        severity = prediction['severity']
        print(f"   Predicted: {severity}")
        print(f"   Description: {self.SEVERITY_DESCRIPTIONS[severity]}")
        
        print(f"\n📊 SEVERITY PROBABILITIES:")
        for grade, prob in prediction['severity_probs'].items():
            bar_length = int(prob * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"   {grade}: {bar} {prob:.2%}")
        
        print(f"\n🎯 CONFIDENCE:")
        print(f"   {prediction['confidence']:.2%}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Complete Tumor Size & Severity Pipeline Demo')
    
    # Demo modes
    parser.add_argument('--toy', action='store_true', help='Run with synthetic data')
    parser.add_argument('--csv', type=str, default='merged_data.csv',
                        help='CSV file with dataset metadata')
    parser.add_argument('--sample', type=int, default=0,
                        help='Sample index to process')
    parser.add_argument('--uid', type=str, default=None,
                        help='Specific UID to process')
    
    # Model
    parser.add_argument('--model-path', type=str,
                        default='models/tumor_size_predictor_best.pth',
                        help='Path to model weights')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CompletePipeline(args.model_path, device=args.device)
    
    # Run demo
    if args.toy:
        pipeline.demo_synthetic()
    else:
        if Path(args.csv).exists():
            pipeline.demo_from_dataset(args.csv, sample_idx=args.sample, uid=args.uid)
        else:
            print(f"CSV not found: {args.csv}")
            print("Running synthetic demo instead...")
            pipeline.demo_synthetic()


if __name__ == '__main__':
    main()
