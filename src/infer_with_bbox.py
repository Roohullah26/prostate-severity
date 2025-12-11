"""
End-to-end inference pipeline for tumor size prediction with visualization.

This module combines:
1. Multi-sequence DICOM loading
2. Size prediction
3. Bounding box generation
4. Severity classification
5. Visualization

Usage (toy):
    python -m src.infer_with_bbox --toy --model-path models/tumor_size_predictor_best.pth

Usage (real data):
    python -m src.infer_with_bbox \\
        --csv merged_data.csv \\
        --sequences t2,adc,dwi \\
        --model-path models/tumor_size_predictor_best.pth \\
        --output results/
"""

import argparse
from pathlib import Path
import torch
import numpy as np
from typing import Optional, Dict, Tuple
import cv2
from PIL import Image
import json

from src.size_predictor_model import TumorSizePredictor
from src.prostate_dataset import ProstateLesionDataset
from src.bbox_utils import (
    TumorPrediction, BoundingBoxGenerator, VisualizationHelper
)
from src.utils_image import pil_to_tensor, get_eval_tensor_transform
from src import config


class TumorInferencePipeline:
    """End-to-end inference pipeline for tumor detection and sizing."""
    
    def __init__(self, model_path: str, device: str = 'cuda', 
                 pixel_spacing_mm: Optional[Tuple[float, float]] = None):
        """Initialize pipeline.
        
        Args:
            model_path: Path to saved TumorSizePredictor weights
            device: 'cuda' or 'cpu'
            pixel_spacing_mm: DICOM pixel spacing for mm conversion
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model = TumorSizePredictor(pretrained=False, in_channels=3)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Bbox generator
        self.bbox_gen = BoundingBoxGenerator(pixel_spacing_mm=pixel_spacing_mm)
        self.vis_helper = VisualizationHelper()
        
        print(f"Pipeline initialized. Model: {model_path}, Device: {self.device}")
    
    def predict_from_image(self, image: Image.Image) -> TumorPrediction:
        """Predict tumor size from a PIL image.
        
        Args:
            image: PIL Image (224x224 recommended)
            
        Returns:
            TumorPrediction object
        """
        # Prepare input
        img_tensor = pil_to_tensor(image, img_size=config.IMG_SIZE)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Extract predictions
        size = output['size'][0].cpu().numpy()  # (3,)
        severity_probs = output['severity_probs'][0].cpu().numpy()  # (4,)
        confidence = output['confidence'][0, 0].item()
        
        # Determine severity
        severity_idx = np.argmax(severity_probs)
        severity_grades = ['T1', 'T2', 'T3', 'T4']
        severity = severity_grades[severity_idx]
        
        # Create prediction object
        prediction = TumorPrediction(
            width_mm=float(size[0]),
            height_mm=float(size[1]),
            depth_mm=float(size[2]),
            severity=severity,
            severity_logits=severity_probs,
            confidence=confidence,
            image_size=(224, 224),
            pixel_spacing_mm=self.bbox_gen.pixel_spacing_mm,
        )
        
        return prediction
    
    def predict_with_visualization(self, image: Image.Image, 
                                   bbox_type: str = 'circle',
                                   return_numpy: bool = False):
        """Predict and visualize on image.
        
        Args:
            image: PIL Image
            bbox_type: 'circle' or 'rect'
            return_numpy: Return as numpy array (BGR) if True
            
        Returns:
            Dict with keys:
                - prediction: TumorPrediction object
                - image: PIL Image or numpy array with drawn bbox
                - bbox: Bounding box dict
        """
        # Get prediction
        prediction = self.predict_from_image(image)
        
        # Generate bbox
        if bbox_type == 'circle':
            bbox = self.bbox_gen.get_circular_bbox(prediction)
            vis_image = self.vis_helper.draw_circular_bbox_pil(
                image, bbox, prediction, thickness=2
            )
        else:  # rect
            bbox = self.bbox_gen.get_rectangular_bbox(prediction)
            vis_image = self.vis_helper.draw_rectangular_bbox_pil(
                image, bbox, prediction, thickness=2
            )
        
        # Convert if requested
        if return_numpy:
            vis_array = cv2.cvtColor(np.array(vis_image), cv2.COLOR_RGB2BGR)
            return {
                'prediction': prediction,
                'image': vis_array,
                'bbox': bbox,
            }
        
        return {
            'prediction': prediction,
            'image': vis_image,
            'bbox': bbox,
        }
    
    def predict_batch(self, images: list, bbox_type: str = 'circle') -> list:
        """Batch predict on multiple images.
        
        Args:
            images: List of PIL Images
            bbox_type: 'circle' or 'rect'
            
        Returns:
            List of prediction dicts (same as predict_with_visualization)
        """
        results = []
        for img in images:
            result = self.predict_with_visualization(img, bbox_type=bbox_type)
            results.append(result)
        return results


def infer_from_dataset(model_path: str, csv_path: Optional[str] = None,
                       sequences: Optional[str] = None, toy: bool = False,
                       toy_len: int = 10, output_dir: Optional[str] = None,
                       max_samples: int = 0):
    """Run inference on dataset samples.
    
    Args:
        model_path: Path to saved model
        csv_path: Path to merged_data.csv
        sequences: Comma-separated sequence keywords
        toy: Use toy dataset
        toy_len: Toy dataset size
        output_dir: Optional directory to save visualizations
        max_samples: Max samples to process (0 = all)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset
    if toy:
        ds = ProstateLesionDataset(toy=True, toy_len=toy_len)
    else:
        seqs = None if sequences is None else [s.strip().lower() for s in sequences.split(",") if s.strip()]
        ds = ProstateLesionDataset(csv_path=csv_path, sequences=seqs)
        ds.transform = get_eval_tensor_transform(img_size=config.IMG_SIZE)
    
    # Initialize pipeline
    pipeline = TumorInferencePipeline(model_path, device=device)
    
    # Create output dir
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process samples
    num_samples = len(ds) if max_samples <= 0 else min(max_samples, len(ds))
    predictions_list = []
    
    print(f"Running inference on {num_samples} samples...")
    
    for idx in range(num_samples):
        try:
            item = ds[idx]
            if isinstance(item, tuple):
                img, label = item
            else:
                img = item
            
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img.astype('uint8'))
            elif not isinstance(img, Image.Image):
                continue
            
            # Inference
            result = pipeline.predict_with_visualization(img, bbox_type='circle')
            prediction = result['prediction']
            
            # Store results
            pred_dict = {
                'sample_id': idx,
                'width_mm': prediction.width_mm,
                'height_mm': prediction.height_mm,
                'depth_mm': prediction.depth_mm,
                'severity': prediction.severity,
                'confidence': prediction.confidence,
                'severity_probs': {
                    'T1': float(prediction.severity_logits[0]),
                    'T2': float(prediction.severity_logits[1]),
                    'T3': float(prediction.severity_logits[2]),
                    'T4': float(prediction.severity_logits[3]),
                },
            }
            predictions_list.append(pred_dict)
            
            # Save visualization
            if output_dir:
                vis_img = result['image']
                save_path = output_dir / f"sample_{idx:04d}_pred.png"
                if isinstance(vis_img, Image.Image):
                    vis_img.save(save_path)
                else:
                    cv2.imwrite(str(save_path), vis_img)
            
            if (idx + 1) % max(1, num_samples // 10) == 0:
                print(f"  Processed {idx + 1}/{num_samples}")
        
        except Exception as e:
            print(f"  Error on sample {idx}: {e}")
            continue
    
    # Save results JSON
    if output_dir:
        results_path = output_dir / 'predictions.json'
        with open(results_path, 'w') as f:
            json.dump(predictions_list, f, indent=2)
        print(f"Saved {len(predictions_list)} predictions to {results_path}")
    
    # Print summary statistics
    if predictions_list:
        sizes = [p['width_mm'] for p in predictions_list]
        severities = [p['severity'] for p in predictions_list]
        from collections import Counter
        
        print("\n=== Inference Summary ===")
        print(f"Total processed: {len(predictions_list)}")
        print(f"Size stats (mm):")
        print(f"  Mean: {np.mean(sizes):.2f}")
        print(f"  Std: {np.std(sizes):.2f}")
        print(f"  Min: {np.min(sizes):.2f}")
        print(f"  Max: {np.max(sizes):.2f}")
        print(f"Severity distribution:")
        for severity, count in Counter(severities).most_common():
            print(f"  {severity}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Inference with bbox visualization")
    
    parser.add_argument("--model-path", required=True, help="Path to model weights")
    parser.add_argument("--csv", default=None, help="Path to merged_data.csv")
    parser.add_argument("--sequences", default="t2,adc", help="Sequence keywords")
    parser.add_argument("--toy", action="store_true", help="Use toy dataset")
    parser.add_argument("--toy-len", type=int, default=10, help="Toy dataset size")
    parser.add_argument("--output", default=None, help="Output directory for visualizations")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples (0=all)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    infer_from_dataset(
        model_path=args.model_path,
        csv_path=args.csv,
        sequences=args.sequences,
        toy=args.toy,
        toy_len=args.toy_len,
        output_dir=args.output,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
