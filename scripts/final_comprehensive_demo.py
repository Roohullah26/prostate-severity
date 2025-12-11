#!/usr/bin/env python
"""
FINAL COMPREHENSIVE DEMO - Tumor Size Prediction Pipeline
Demonstrates the complete workflow:
1. Load multi-sequence MRI (synthetic T2, ADC, DWI)
2. Predict tumor size
3. Generate bounding boxes (rectangular & circular)
4. Classify TNM severity (T1-T4)
5. Output JSON results with clinical interpretation
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys
from datetime import datetime
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from size_predictor_model import TumorSizePredictor
from bbox_utils import BoundingBoxGenerator, TumorPrediction, VisualizationHelper


def generate_test_mri(size=224, seed=42):
    """Generate synthetic multi-sequence MRI."""
    np.random.seed(seed)
    
    # T2 sequence
    t2 = np.random.randint(60, 200, (size, size), dtype=np.uint8)
    # ADC sequence
    adc = np.random.randint(40, 150, (size, size), dtype=np.uint8)
    # DWI sequence
    dwi = np.random.randint(70, 220, (size, size), dtype=np.uint8)
    
    # Stack into 3-channel image
    stacked = np.stack([t2, adc, dwi], axis=2)
    return stacked


def main():
    print("\n" + "="*90)
    print("PROSTATE TUMOR SIZE PREDICTION - COMPREHENSIVE DEMO")
    print("="*90)
    print("\nThis demo shows the complete end-to-end pipeline:")
    print("  1. Multi-sequence MRI analysis (T2, ADC, DWI)")
    print("  2. Precise tumor size prediction (mm)")
    print("  3. Automatic bounding box generation")
    print("  4. TNM severity classification (T1-T4)")
    print("  5. Clinical recommendations")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_dir = Path(__file__).resolve().parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    print(f"\n[CONFIG]")
    print(f"  Device: {device}")
    print(f"  Results dir: {results_dir}")
    
    # ============================================================================
    # STEP 1: Load Model
    # ============================================================================
    print(f"\n[STEP 1] Loading Tumor Size Predictor Model...")
    
    model = TumorSizePredictor(pretrained=False, in_channels=3, dropout_rate=0.3)
    
    model_path = Path(__file__).resolve().parent.parent / 'models' / 'baseline_real_t2_adc_3s_ep1.pth'
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"  [OK] Loaded pre-trained weights: {model_path.name}")
    else:
        print(f"  [WARN] Using random initialization (weights not found)")
    
    model = model.to(device)
    model.eval()
    
    print(f"  [OK] Model ready on {device}")
    print(f"  [OK] Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ============================================================================
    # STEP 2: Generate Synthetic Multi-Sequence MRI
    # ============================================================================
    print(f"\n[STEP 2] Generating Synthetic Multi-Sequence MRI...")
    
    mri_stacked = generate_test_mri(size=224, seed=42)
    print(f"  [OK] T2 channel shape: {mri_stacked[:,:,0].shape}")
    print(f"  [OK] ADC channel shape: {mri_stacked[:,:,1].shape}")
    print(f"  [OK] DWI channel shape: {mri_stacked[:,:,2].shape}")
    print(f"  [OK] Stacked input shape: {mri_stacked.shape}")
    
    # ============================================================================
    # STEP 3: Run Inference
    # ============================================================================
    print(f"\n[STEP 3] Running Tumor Size Prediction...")
    
    # Prepare input tensor
    img_tensor = torch.from_numpy(mri_stacked).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    img_tensor = img_tensor.to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)
    
    # Extract predictions
    size = output['size'][0].cpu().numpy()
    severity_probs = output['severity_probs'][0].cpu().numpy()
    confidence = output['confidence'][0, 0].item()
    
    print(f"  [OK] Predicted tumor size:")
    print(f"       Width:  {size[0]:.2f} mm")
    print(f"       Height: {size[1]:.2f} mm")
    print(f"       Depth:  {size[2]:.2f} mm")
    print(f"  [OK] Prediction confidence: {confidence:.3f}")
    
    # ============================================================================
    # STEP 4: Classify Severity
    # ============================================================================
    print(f"\n[STEP 4] Classifying TNM Severity...")
    
    severity_grades = ['T1 (Small <=20mm)', 'T2 (Medium 20-40mm)', 'T3 (Large 40-60mm)', 'T4 (Very Large >60mm)']
    severity_idx = int(np.argmax(severity_probs))
    severity_stage = ['T1', 'T2', 'T3', 'T4'][severity_idx]
    
    print(f"  [OK] Severity stage: {severity_stage}")
    print(f"  [OK] Stage probabilities:")
    for i, (grade, prob) in enumerate(zip(severity_grades, severity_probs)):
        marker = " <-- PREDICTED" if i == severity_idx else ""
        print(f"       {grade}: {prob:.3f}{marker}")
    
    # ============================================================================
    # STEP 5: Generate Bounding Boxes
    # ============================================================================
    print(f"\n[STEP 5] Generating Bounding Boxes...")
    
    pred = TumorPrediction(
        width_mm=float(size[0]),
        height_mm=float(size[1]),
        depth_mm=float(size[2]),
        severity=severity_stage,
        severity_logits=severity_probs,
        confidence=confidence,
        image_size=(224, 224),
        pixel_spacing_mm=(1.0, 1.0)
    )
    
    bbox_gen = BoundingBoxGenerator(pixel_spacing_mm=(1.0, 1.0))
    rect_bbox = bbox_gen.get_rectangular_bbox(pred)
    circ_bbox = bbox_gen.get_circular_bbox(pred)
    
    print(f"  [OK] Rectangular bounding box:")
    print(f"       Top-left: ({rect_bbox['x1']}, {rect_bbox['y1']})")
    print(f"       Bottom-right: ({rect_bbox['x2']}, {rect_bbox['y2']})")
    print(f"       Width: {rect_bbox['width_px']} px, Height: {rect_bbox['height_px']} px")
    print(f"  [OK] Circular bounding box:")
    print(f"       Center: ({circ_bbox['center_x']}, {circ_bbox['center_y']})")
    print(f"       Radius: {circ_bbox['radius_px']} px ({circ_bbox['radius_mm']:.1f} mm)")
    print(f"       Diameter: {circ_bbox['diameter_mm']:.1f} mm")
    
    # ============================================================================
    # STEP 6: Generate Clinical Report
    # ============================================================================
    print(f"\n[STEP 6] Generating Clinical Report...")
    
    clinical_notes = ""
    recommendations = ""
    
    if severity_stage == 'T1':
        clinical_notes = "Small tumor, minimal disease burden. Excellent prognosis with treatment."
        recommendations = "Active surveillance, repeat imaging in 3-6 months. Consider biopsy if growth detected."
    elif severity_stage == 'T2':
        clinical_notes = "Medium tumor, localized disease. Good prognosis with aggressive treatment."
        recommendations = "Active treatment recommended (radiation/brachytherapy/surgery). MRI follow-up every 3 months."
    elif severity_stage == 'T3':
        clinical_notes = "Large tumor, advanced localized disease. Guarded prognosis, requires intervention."
        recommendations = "Aggressive treatment essential. Multi-modality approach (surgery + radiation). Close follow-up."
    else:  # T4
        clinical_notes = "Very large tumor, extensive disease. High risk for spread. Poor prognosis without treatment."
        recommendations = "Urgent intervention required. Multidisciplinary team consultation. Consider systemic therapy."
    
    print(f"  [OK] Clinical Notes: {clinical_notes}")
    print(f"  [OK] Recommendations: {recommendations}")
    
    # ============================================================================
    # STEP 7: Save Results to JSON
    # ============================================================================
    print(f"\n[STEP 7] Saving Results to JSON...")
    
    json_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'sample_id': 'demo_001',
            'model': 'baseline_real_t2_adc_3s_ep1',
            'device': device,
        },
        'mri_sequences': {
            'T2': {'present': True, 'role': 'Anatomical reference'},
            'ADC': {'present': True, 'role': 'Diffusion-weighted'},
            'DWI': {'present': True, 'role': 'Diffusion-weighted'},
        },
        'predictions': {
            'tumor_size_mm': {
                'width': float(size[0]),
                'height': float(size[1]),
                'depth': float(size[2]),
                'max_dimension': float(max(size)),
            },
            'confidence': float(confidence),
        },
        'severity': {
            'tnm_stage': severity_stage,
            'stage_probabilities': {
                'T1': float(severity_probs[0]),
                'T2': float(severity_probs[1]),
                'T3': float(severity_probs[2]),
                'T4': float(severity_probs[3]),
            },
            'clinical_notes': clinical_notes,
            'recommendations': recommendations,
        },
        'bounding_boxes': {
            'rectangular': {
                'type': 'rect',
                'x1': int(rect_bbox['x1']),
                'y1': int(rect_bbox['y1']),
                'x2': int(rect_bbox['x2']),
                'y2': int(rect_bbox['y2']),
                'center': [int(rect_bbox['center_x']), int(rect_bbox['center_y'])],
                'dimensions': {'width_px': int(rect_bbox['width_px']), 'height_px': int(rect_bbox['height_px'])},
            },
            'circular': {
                'type': 'circle',
                'center': [int(circ_bbox['center_x']), int(circ_bbox['center_y'])],
                'radius_px': int(circ_bbox['radius_px']),
                'radius_mm': float(circ_bbox['radius_mm']),
                'diameter_mm': float(circ_bbox['diameter_mm']),
            }
        }
    }
    
    # Save JSON
    json_file = results_dir / 'final_demo_results.json'
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"  [OK] Saved to: {json_file}")
    
    # ============================================================================
    # STEP 8: Create Visualization
    # ============================================================================
    print(f"\n[STEP 8] Creating Visualizations...")
    
    try:
        # Overlay bounding box on T2 image
        t2_image = mri_stacked[:, :, 0]
        t2_rgb = np.stack([t2_image, t2_image, t2_image], axis=2)
        
        pil_img = Image.fromarray(t2_rgb)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(pil_img)
        
        # Color based on severity
        colors = {'T1': (0, 255, 0), 'T2': (255, 255, 0), 'T3': (255, 165, 0), 'T4': (255, 0, 0)}
        color = colors.get(severity_stage, (200, 200, 200))
        
        # Draw rectangular bbox
        draw.rectangle(
            [(rect_bbox['x1'], rect_bbox['y1']), (rect_bbox['x2'], rect_bbox['y2'])],
            outline=color, width=2
        )
        
        # Draw label
        label = f"{severity_stage} {max(size):.1f}mm"
        try:
            draw.text((10, 10), label, fill=color)
        except:
            pass
        
        # Save
        viz_file = results_dir / 'demo_visualization.png'
        pil_img.save(viz_file)
        print(f"  [OK] Saved visualization: {viz_file}")
        
    except Exception as e:
        print(f"  [WARN] Could not create visualization: {e}")
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print(f"\n" + "="*90)
    print("DEMO COMPLETE - RESULTS SUMMARY")
    print("="*90)
    
    print(f"\n[TUMOR ANALYSIS]")
    print(f"  Size: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} mm")
    print(f"  Max dimension: {max(size):.2f} mm")
    print(f"  Severity: {severity_stage}")
    print(f"  Confidence: {confidence:.3f}")
    
    print(f"\n[CLINICAL DECISION]")
    print(f"  {clinical_notes}")
    
    print(f"\n[TREATMENT PLAN]")
    print(f"  {recommendations}")
    
    print(f"\n[OUTPUT FILES]")
    print(f"  JSON results: {json_file}")
    if Path(results_dir / 'demo_visualization.png').exists():
        print(f"  Visualization: {results_dir / 'demo_visualization.png'}")
    
    print(f"\n[NEXT STEPS]")
    print(f"  1. Review JSON results for integration with PACS/EHR")
    print(f"  2. Start API server: python -m uvicorn webapp.fastapi_server:app --port 8000")
    print(f"  3. Process batch of samples: python scripts/batch_predict_tumor_size.py")
    print(f"  4. Fine-tune model: python scripts/train_size_model.py --data-path your_data.csv")
    
    print(f"\n" + "="*90)
    
    return True


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
