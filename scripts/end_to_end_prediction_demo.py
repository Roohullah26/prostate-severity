#!/usr/bin/env python
"""
End-to-end tumor size prediction pipeline.

This script:
1. Generates synthetic multi-sequence MRI images (T2, ADC, DWI)
2. Loads the trained size predictor model
3. Runs predictions
4. Generates bounding boxes
5. Creates visualizations
6. Outputs JSON results
"""

import torch
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from size_predictor_model import TumorSizePredictor
from bbox_utils import BoundingBoxGenerator, VisualizationHelper, TumorPrediction


def create_synthetic_mri_image(size=224, intensity_range=(50, 200)):
    """Create a synthetic MRI-like image with Gaussian noise."""
    # Create base image with smooth gradients
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Create circular region (simulating tumor area)
    tumor_mask = (X**2 + Y**2) < 0.3
    
    # Create base intensity
    base_intensity = np.full((size, size), intensity_range[0], dtype=np.uint8)
    
    # Add intensity in tumor region
    base_intensity[tumor_mask] = intensity_range[1]
    
    # Add Gaussian noise
    noise = np.random.normal(0, 10, (size, size))
    image = np.clip(base_intensity + noise, 0, 255).astype(np.uint8)
    
    return image


def create_stacked_mri(size=224, seed=None):
    """Create a 3-channel stacked MRI (simulating T2, ADC, DWI)."""
    if seed is not None:
        np.random.seed(seed)
    
    # Create 3 channels with slight variations
    t2 = create_synthetic_mri_image(size, intensity_range=(60, 200))
    adc = create_synthetic_mri_image(size, intensity_range=(40, 150))
    dwi = create_synthetic_mri_image(size, intensity_range=(70, 220))
    
    # Stack channels
    stacked = np.stack([t2, adc, dwi], axis=2)
    
    return stacked, (t2, adc, dwi)


def predict_tumor_size(model, device, image_array):
    """Run inference on stacked MRI image."""
    # Convert to tensor
    img_tensor = torch.from_numpy(image_array).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    img_tensor = img_tensor.to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)
    
    return output


def overlay_bbox_on_image(image, bbox, prediction, bbox_type='rect'):
    """Draw bbox on image and return PIL Image."""
    # Convert grayscale to RGB for consistent coloring
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)
    
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    
    # Color scheme (RGB)
    colors_rgb = {
        'T1': (0, 255, 0),      # Green
        'T2': (255, 255, 0),    # Yellow
        'T3': (255, 165, 0),    # Orange
        'T4': (255, 0, 0),      # Red
    }
    
    color = colors_rgb.get(prediction.severity, (200, 200, 200))
    
    if bbox_type == 'circle':
        x = bbox['center_x']
        y = bbox['center_y']
        r = bbox['radius_px']
        draw.ellipse([x-r, y-r, x+r, y+r], outline=color, width=2)
        draw.ellipse([x-3, y-3, x+3, y+3], fill=color)
        label = f"{prediction.severity} {prediction.width_mm:.1f}mm"
    else:  # rect
        x1, y1 = bbox['x1'], bbox['y1']
        x2, y2 = bbox['x2'], bbox['y2']
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{prediction.severity} {prediction.width_mm:.1f}x{prediction.height_mm:.1f}mm"
    
    # Draw label
    try:
        draw.text((10, 10), label, fill=color)
    except:
        pass
    
    return img


def run_full_pipeline():
    """Run complete end-to-end pipeline."""
    print("\n" + "="*80)
    print("[>>] END-TO-END TUMOR SIZE PREDICTION PIPELINE")
    print("="*80)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_dir = Path(__file__).resolve().parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    print(f"\nDevice: {device}")
    print(f"Results directory: {results_dir}")
    
    # Step 1: Generate synthetic MRI
    print("\n[1/6] Generating synthetic MRI images...")
    stacked_mri, (t2, adc, dwi) = create_stacked_mri(size=224, seed=42)
    print(f"  [OK] Created T2 ({t2.shape}), ADC ({adc.shape}), DWI ({dwi.shape})")
    
    # Step 2: Load model
    print("\n[2/6] Loading model...")
    model = TumorSizePredictor(pretrained=False, in_channels=3, dropout_rate=0.3)
    
    model_path = Path(__file__).resolve().parent.parent / 'models' / 'baseline_real_t2_adc_3s_ep1.pth'
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"  ✓ Loaded weights from {model_path.name}")
    else:
        print(f"  ⚠ No weights found, using random initialization")
    
    model = model.to(device)
    model.eval()
    
    # Step 3: Run inference
    print("\n[3/6] Running inference...")
    output = predict_tumor_size(model, device, stacked_mri)
    
    size = output['size'][0].cpu().numpy()
    severity_probs = output['severity_probs'][0].cpu().numpy()
    confidence = output['confidence'][0, 0].item()
    
    print(f"  ✓ Predicted size: {size[0]:.2f}W x {size[1]:.2f}H x {size[2]:.2f}D mm")
    print(f"  ✓ Confidence: {confidence:.3f}")
    
    # Step 4: Create severity prediction
    print("\n[4/6] Creating severity prediction...")
    severity_grades = ['T1', 'T2', 'T3', 'T4']
    severity_idx = int(np.argmax(severity_probs))
    
    pred = TumorPrediction(
        width_mm=float(size[0]),
        height_mm=float(size[1]),
        depth_mm=float(size[2]),
        severity=severity_grades[severity_idx],
        severity_logits=severity_probs,
        confidence=confidence,
        image_size=(224, 224),
        pixel_spacing_mm=(1.0, 1.0)
    )
    
    print(f"  ✓ Severity: {pred.severity}")
    print(f"  ✓ Probabilities: T1={severity_probs[0]:.3f}, T2={severity_probs[1]:.3f}, "
          f"T3={severity_probs[2]:.3f}, T4={severity_probs[3]:.3f}")
    
    # Step 5: Generate bounding boxes
    print("\n[5/6] Generating bounding boxes...")
    bbox_gen = BoundingBoxGenerator(pixel_spacing_mm=(1.0, 1.0))
    
    rect_bbox = bbox_gen.get_rectangular_bbox(pred)
    circ_bbox = bbox_gen.get_circular_bbox(pred)
    
    print(f"  ✓ Rectangular bbox: ({rect_bbox['x1']}, {rect_bbox['y1']}) to ({rect_bbox['x2']}, {rect_bbox['y2']})")
    print(f"  ✓ Circular bbox: center=({circ_bbox['center_x']}, {circ_bbox['center_y']}), radius={circ_bbox['radius_px']}px")
    
    # Step 6: Create visualizations and save
    print("\n[6/6] Creating visualizations...")
    
    # Overlay bbox on images
    rect_viz = overlay_bbox_on_image(stacked_mri[:, :, 0], rect_bbox, pred, 'rect')
    circ_viz = overlay_bbox_on_image(stacked_mri[:, :, 1], circ_bbox, pred, 'circle')
    
    # Save visualizations
    rect_path = results_dir / 'prediction_rectangular_bbox.png'
    circ_path = results_dir / 'prediction_circular_bbox.png'
    
    rect_viz.save(rect_path)
    circ_viz.save(circ_path)
    
    print(f"  ✓ Saved rectangular bbox visualization: {rect_path}")
    print(f"  ✓ Saved circular bbox visualization: {circ_path}")
    
    # Save JSON results
    json_output = {
        'timestamp': datetime.now().isoformat(),
        'sample_id': 'synthetic_001',
        'prediction': {
            'width_mm': pred.width_mm,
            'height_mm': pred.height_mm,
            'depth_mm': pred.depth_mm,
            'max_dimension_mm': max(pred.width_mm, pred.height_mm, pred.depth_mm),
            'confidence': pred.confidence,
        },
        'severity': {
            'stage': pred.severity,
            'probabilities': {
                'T1': float(severity_probs[0]),
                'T2': float(severity_probs[1]),
                'T3': float(severity_probs[2]),
                'T4': float(severity_probs[3]),
            }
        },
        'bounding_boxes': {
            'rectangular': {
                'type': 'rect',
                'x1': int(rect_bbox['x1']),
                'y1': int(rect_bbox['y1']),
                'x2': int(rect_bbox['x2']),
                'y2': int(rect_bbox['y2']),
                'center_x': int(rect_bbox['center_x']),
                'center_y': int(rect_bbox['center_y']),
                'width_px': int(rect_bbox['width_px']),
                'height_px': int(rect_bbox['height_px']),
            },
            'circular': {
                'type': 'circle',
                'center_x': int(circ_bbox['center_x']),
                'center_y': int(circ_bbox['center_y']),
                'radius_px': int(circ_bbox['radius_px']),
                'radius_mm': float(circ_bbox['radius_mm']),
                'diameter_mm': float(circ_bbox['diameter_mm']),
            }
        }
    }
    
    json_path = results_dir / 'prediction_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"  ✓ Saved JSON results: {json_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("✨ PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {results_dir}/")
    print(f"  - prediction_results.json (JSON output)")
    print(f"  - prediction_rectangular_bbox.png (Visualization)")
    print(f"  - prediction_circular_bbox.png (Visualization)")
    
    print(f"\n📊 PREDICTION SUMMARY")
    print(f"  Tumor Size: {pred.width_mm:.2f} × {pred.height_mm:.2f} × {pred.depth_mm:.2f} mm")
    print(f"  Severity: {pred.severity}")
    print(f"  Confidence: {pred.confidence:.3f}")
    
    print(f"\n🎯 CLINICAL INTERPRETATION")
    if pred.severity == 'T1':
        print(f"  Small tumor (≤20mm) - Early stage")
        print(f"  Recommendation: Monitor, active surveillance")
    elif pred.severity == 'T2':
        print(f"  Medium tumor (20-40mm) - Localized disease")
        print(f"  Recommendation: Active treatment recommended")
    elif pred.severity == 'T3':
        print(f"  Large tumor (40-60mm) - Advanced localized")
        print(f"  Recommendation: Aggressive treatment required")
    else:  # T4
        print(f"  Very large tumor (>60mm) - Advanced disease")
        print(f"  Recommendation: Multi-disciplinary urgent intervention")
    
    return True


if __name__ == '__main__':
    try:
        success = run_full_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
