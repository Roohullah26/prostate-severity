"""
Comprehensive demonstration of tumor size prediction with severity classification.

This script shows how to:
1. Load multi-sequence MRI data (T2, ADC, DWI)
2. Predict tumor dimensions (width, height, depth in mm)
3. Classify severity (T1, T2, T3, T4)
4. Generate bounding boxes (circular or rectangular)
5. Visualize predictions

Usage:
    # Toy demo
    python scripts/demo_tumor_size_prediction.py --toy --output demo_results/
    
    # Real data with trained model
    python scripts/demo_tumor_size_prediction.py \\
        --model-path models/tumor_size_predictor_best.pth \\
        --csv merged_data.csv \\
        --sequences t2,adc,dwi \\
        --output demo_results/
        
    # Single image inference
    python scripts/demo_tumor_size_prediction.py \\
        --model-path models/tumor_size_predictor_best.pth \\
        --image path/to/image.png \\
        --output demo_results/
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from typing import Optional, Dict, List
import json
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infer_with_bbox import TumorInferencePipeline
from src.size_predictor_model import TumorSizePredictor
from src.bbox_utils import BoundingBoxGenerator, VisualizationHelper, TumorPrediction
from src.prostate_dataset import ProstateLesionDataset
from src.utils_image import get_eval_tensor_transform
from src import config


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_model_architecture():
    """Demonstrate the model architecture."""
    print_header("1. Model Architecture")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    model = TumorSizePredictor(pretrained=True, in_channels=3)
    model = model.to(device)
    
    # Print architecture
    print("\nModel Overview:")
    print(f"  Backbone: ResNet18 (pretrained)")
    print(f"  Input: (B, 3, 224, 224) - Multi-sequence MRI stack (T2/ADC/DWI)")
    print(f"  Feature Dimension: {model.feature_dim}")
    print("\nOutput Heads:")
    print("  1. Size Head: predicts [width_mm, height_mm, depth_mm]")
    print("  2. Severity Head: predicts [T1, T2, T3, T4] logits")
    print("  3. Confidence Head: predicts confidence score [0, 1]")
    
    # Test forward pass
    print("\nTesting forward pass with dummy input...")
    with torch.no_grad():
        x = torch.randn(2, 3, 224, 224).to(device)
        output = model(x)
    
    print("Output shapes:")
    for key, val in output.items():
        print(f"  {key}: {val.shape}")
    
    return model, device


def demo_severity_classification():
    """Demonstrate severity classification based on tumor size."""
    print_header("2. Severity Classification Scheme")
    
    bbox_gen = BoundingBoxGenerator()
    
    print("\nSeverity Classification based on maximum tumor dimension:")
    print("  T1: ≤ 20 mm  - Small, clinically insignificant")
    print("  T2: 20-40 mm - Medium, locally confined")
    print("  T3: 40-60 mm - Large, may extend beyond organ")
    print("  T4: > 60 mm  - Very large, extensive local invasion")
    
    # Test classification
    test_sizes = [10, 15, 25, 35, 50, 65, 80]
    print("\nExample classifications:")
    for size in test_sizes:
        severity = bbox_gen.classify_severity(size)
        print(f"  {size:2d} mm → {severity}")


def demo_bounding_box_generation():
    """Demonstrate bounding box generation."""
    print_header("3. Bounding Box Generation")
    
    # Create sample prediction
    pred = TumorPrediction(
        width_mm=18.5,
        height_mm=22.3,
        depth_mm=20.1,
        severity='T2',
        severity_logits=np.array([0.05, 0.80, 0.12, 0.03]),
        confidence=0.94,
        image_size=(224, 224),
        pixel_spacing_mm=(1.0, 1.0)
    )
    
    bbox_gen = BoundingBoxGenerator(pixel_spacing_mm=(1.0, 1.0))
    
    print("\nSample Prediction:")
    print(f"  Dimensions: {pred.width_mm:.1f} x {pred.height_mm:.1f} x {pred.depth_mm:.1f} mm")
    print(f"  Severity: {pred.severity}")
    print(f"  Confidence: {pred.confidence:.3f}")
    
    # Rectangular bbox
    rect_bbox = bbox_gen.get_rectangular_bbox(pred)
    print("\nRectangular Bounding Box:")
    print(f"  Top-left: ({rect_bbox['x1']}, {rect_bbox['y1']})")
    print(f"  Bottom-right: ({rect_bbox['x2']}, {rect_bbox['y2']})")
    print(f"  Size in pixels: {rect_bbox['width_px']} x {rect_bbox['height_px']}")
    
    # Circular bbox
    circ_bbox = bbox_gen.get_circular_bbox(pred)
    print("\nCircular Bounding Box:")
    print(f"  Center: ({circ_bbox['center_x']}, {circ_bbox['center_y']})")
    print(f"  Radius: {circ_bbox['radius_px']} pixels ({circ_bbox['radius_mm']:.1f} mm)")
    print(f"  Diameter: {circ_bbox['diameter_mm']:.1f} mm")


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print_header("4. Visualization Examples")
    
    # Create dummy image
    print("Creating sample visualization...")
    img = Image.new('RGB', (224, 224), color=(100, 100, 150))
    
    pred = TumorPrediction(
        width_mm=15.0,
        height_mm=18.0,
        depth_mm=16.5,
        severity='T2',
        severity_logits=np.array([0.05, 0.82, 0.10, 0.03]),
        confidence=0.96,
        image_size=(224, 224),
        pixel_spacing_mm=(1.0, 1.0)
    )
    
    bbox_gen = BoundingBoxGenerator()
    vis_helper = VisualizationHelper()
    
    # Circular bbox
    circ_bbox = bbox_gen.get_circular_bbox(pred)
    vis_circ = vis_helper.draw_circular_bbox_pil(img, circ_bbox, pred)
    print(f"  ✓ Circular bbox visualization created")
    
    # Rectangular bbox
    rect_bbox = bbox_gen.get_rectangular_bbox(pred)
    vis_rect = vis_helper.draw_rectangular_bbox_pil(img, rect_bbox, pred)
    print(f"  ✓ Rectangular bbox visualization created")
    
    print("\nVisualization color scheme by severity:")
    print("  T1 (≤20mm): Green")
    print("  T2 (20-40mm): Yellow")
    print("  T3 (40-60mm): Orange")
    print("  T4 (>60mm): Red")
    
    return vis_circ, vis_rect


def demo_inference_on_toy_data(model_path: Optional[str] = None, 
                               output_dir: Optional[Path] = None):
    """Run inference on toy dataset."""
    print_header("5. Inference on Toy Dataset")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load toy dataset
    print("\nLoading toy dataset (10 samples)...")
    toy_ds = ProstateLesionDataset(toy=True, toy_len=10)
    print(f"  Loaded {len(toy_ds)} toy samples")
    
    # Initialize or load model
    if model_path and Path(model_path).exists():
        print(f"Loading pretrained model from {model_path}...")
        pipeline = TumorInferencePipeline(model_path, device=device)
    else:
        print("No pretrained model provided. Creating untrained model for demo...")
        model = TumorSizePredictor(pretrained=False, in_channels=3)
        model = model.to(device)
        model.eval()
        # Manual pipeline setup
        pipeline = TumorInferencePipeline.__new__(TumorInferencePipeline)
        pipeline.device = device
        pipeline.model = model
        pipeline.bbox_gen = BoundingBoxGenerator()
        pipeline.vis_helper = VisualizationHelper()
    
    # Run inference
    print("\nRunning inference on toy samples...")
    predictions = []
    
    for idx in range(min(5, len(toy_ds))):
        item = toy_ds[idx]
        if isinstance(item, tuple):
            img, _ = item
        else:
            img = item
        
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype('uint8'))
        elif not isinstance(img, Image.Image):
            continue
        
        # Inference
        result = pipeline.predict_with_visualization(img, bbox_type='circle')
        prediction = result['prediction']
        
        predictions.append({
            'sample_id': idx,
            'width_mm': prediction.width_mm,
            'height_mm': prediction.height_mm,
            'depth_mm': prediction.depth_mm,
            'severity': prediction.severity,
            'confidence': prediction.confidence,
        })
        
        # Save visualization
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            vis_img = result['image']
            save_path = output_dir / f"toy_sample_{idx:02d}.png"
            if isinstance(vis_img, Image.Image):
                vis_img.save(save_path)
            print(f"  Sample {idx}: Saved to {save_path}")
    
    # Print summary
    print("\nInference Results:")
    print(f"{'ID':<4} {'Width':<10} {'Height':<10} {'Depth':<10} {'Severity':<10} {'Confidence':<10}")
    print("-" * 55)
    for p in predictions:
        print(f"{p['sample_id']:<4} {p['width_mm']:<10.2f} {p['height_mm']:<10.2f} "
              f"{p['depth_mm']:<10.2f} {p['severity']:<10} {p['confidence']:<10.3f}")
    
    # Summary statistics
    if predictions:
        sizes = [max(p['width_mm'], p['height_mm'], p['depth_mm']) for p in predictions]
        print(f"\nSize Statistics (max dimension):")
        print(f"  Mean: {np.mean(sizes):.2f} mm")
        print(f"  Std: {np.std(sizes):.2f} mm")
        print(f"  Range: {np.min(sizes):.2f} - {np.max(sizes):.2f} mm")
        
        severities = [p['severity'] for p in predictions]
        print(f"Severity Distribution:")
        for severity, count in Counter(severities).most_common():
            print(f"  {severity}: {count}")


def demo_single_image_inference(image_path: str, model_path: str, 
                               output_dir: Optional[Path] = None):
    """Run inference on a single image file."""
    print_header("5. Single Image Inference")
    
    # Load image
    print(f"Loading image from {image_path}...")
    if not Path(image_path).exists():
        print(f"ERROR: Image not found at {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    # Resize to model input size
    img = img.resize(config.IMG_SIZE)
    print(f"  Image shape: {img.size}")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_path}...")
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        return
    
    pipeline = TumorInferencePipeline(model_path, device=device)
    
    # Run inference
    print("Running inference...")
    result = pipeline.predict_with_visualization(img, bbox_type='circle')
    prediction = result['prediction']
    
    # Print results
    print("\n" + VisualizationHelper.create_prediction_summary(prediction))
    
    # Save visualization
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        vis_img = result['image']
        save_path = output_dir / "single_image_prediction.png"
        if isinstance(vis_img, Image.Image):
            vis_img.save(save_path)
        else:
            import cv2
            cv2.imwrite(str(save_path), vis_img)
        print(f"Visualization saved to {save_path}")


def create_summary_report(output_dir: Path):
    """Create a summary report of the demo."""
    print_header("Summary Report")
    
    report = """
================================================================================
                    TUMOR SIZE PREDICTION SYSTEM DEMO
================================================================================

SYSTEM OVERVIEW:
  This system uses a deep learning model (ResNet18) to predict tumor dimensions
  and severity classification from multi-sequence MRI data (T2, ADC, DWI).

KEY FEATURES:
  ✓ Multi-sequence MRI processing (T2, ADC, DWI stacking)
  ✓ 3D tumor dimension prediction (width, height, depth in mm)
  ✓ Severity classification (T1/T2/T3/T4 based on size)
  ✓ Confidence scoring for predictions
  ✓ Bounding box generation (circular or rectangular)
  ✓ Visualization with severity-based color coding

MODEL ARCHITECTURE:
  Backbone: ResNet18 (ImageNet-pretrained)
  Input: (B, 3, 224, 224) - stacked multi-sequence MRI
  Outputs:
    - Size: [width_mm, height_mm, depth_mm]
    - Severity Logits: [T1, T2, T3, T4] probabilities
    - Confidence: [0, 1] prediction confidence

SEVERITY CLASSIFICATION:
  T1: ≤ 20 mm   - Small, clinically insignificant (Green)
  T2: 20-40 mm  - Medium, locally confined (Yellow)
  T3: 40-60 mm  - Large, may extend beyond organ (Orange)
  T4: > 60 mm   - Very large, extensive invasion (Red)

API ENDPOINT (FastAPI):
  POST /predict-size
  
  Parameters:
    - file: MRI image upload
    - model_path: Path to model weights
    - bbox_type: "circle" or "rect"
    - pixel_spacing: JSON [row_mm, col_mm]
    - return_image: Include base64-encoded visualization
  
  Returns:
    - severity: Predicted severity grade
    - width_mm, height_mm, depth_mm: Tumor dimensions
    - max_dimension_mm: Maximum dimension (for quick assessment)
    - confidence: Prediction confidence
    - severity_probabilities: Probabilities for each grade
    - bbox: Bounding box coordinates/radius
    - image_base64 (optional): Base64-encoded visualization

USAGE EXAMPLES:

1. Toy Dataset Demo (no pretrained model needed):
   python scripts/demo_tumor_size_prediction.py --toy

2. Real Data Inference:
   python scripts/demo_tumor_size_prediction.py \\
     --model-path models/tumor_size_predictor_best.pth \\
     --csv merged_data.csv \\
     --sequences t2,adc,dwi

3. Single Image:
   python scripts/demo_tumor_size_prediction.py \\
     --model-path models/tumor_size_predictor_best.pth \\
     --image path/to/image.png

4. Via FastAPI Server:
   # Start server
   python -m uvicorn webapp.fastapi_server:app --reload
   
   # Make prediction
   curl -X POST http://localhost:8000/predict-size \\
     -F "file=@mri_image.png" \\
     -F "model_path=models/tumor_size_predictor_best.pth" \\
     -F "return_image=true"

TRAINING:
  To train the model on your data:
  
  python -m src.train_size_model \\
    --csv merged_data.csv \\
    --size-csv tumor_sizes.csv \\
    --sequences t2,adc,dwi \\
    --epochs 20 \\
    --bs 8

  Expected output: models/tumor_size_predictor_best.pth

OUTPUT FILES:
  Demo results saved to:
"""
    
    if output_dir and output_dir.exists():
        report += f"  {output_dir}/\n"
        files = list(output_dir.glob("*.png"))
        if files:
            report += f"\n  Generated visualizations ({len(files)} files):\n"
            for f in files[:5]:
                report += f"    - {f.name}\n"
            if len(files) > 5:
                report += f"    ... and {len(files) - 5} more\n"
    
    report += """
================================================================================
                                   END OF DEMO
================================================================================
"""
    
    print(report)
    
    # Save report
    if output_dir:
        report_path = output_dir / "DEMO_REPORT.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive demo of tumor size prediction system"
    )
    parser.add_argument("--toy", action="store_true", 
                       help="Use toy dataset")
    parser.add_argument("--model-path", default=None,
                       help="Path to trained model weights")
    parser.add_argument("--image", default=None,
                       help="Single image for inference")
    parser.add_argument("--csv", default=None,
                       help="Path to dataset CSV")
    parser.add_argument("--sequences", default="t2,adc",
                       help="Sequences to use (comma-separated)")
    parser.add_argument("--output", default="demo_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    output_dir = Path(args.output) if args.output else None
    
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█  TUMOR SIZE PREDICTION SYSTEM - COMPREHENSIVE DEMO" + " " * 5 + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60 + "\n")
    
    try:
        # Run demos
        demo_model_architecture()
        demo_severity_classification()
        demo_bounding_box_generation()
        demo_visualization()
        
        # Inference demos
        if args.image:
            demo_single_image_inference(args.image, args.model_path or 
                                       str(config.MODELS_DIR / "tumor_size_predictor_best.pth"),
                                       output_dir)
        elif args.toy or not args.model_path:
            demo_inference_on_toy_data(args.model_path, output_dir)
        else:
            from src.infer_with_bbox import infer_from_dataset
            infer_from_dataset(
                model_path=args.model_path,
                csv_path=args.csv,
                sequences=args.sequences,
                toy=False,
                output_dir=str(output_dir) if output_dir else None,
                max_samples=10
            )
        
        # Summary
        create_summary_report(output_dir)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
