#!/usr/bin/env python3
"""
Quick Start: Tumor Size Prediction System
Minimal example showing how to use the complete system
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def demo_1_basic_prediction():
    """Demo 1: Basic size prediction"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Tumor Size Prediction")
    print("="*70)
    
    import numpy as np
    from utils_image import normalize_image
    from size_predictor_model import TumorSizePredictor
    
    # Create sample image (grayscale)
    sample_image = np.random.rand(256, 256) * 0.5 + 0.25
    
    # Normalize
    img_norm = normalize_image(sample_image)
    
    # Load model
    print("\n🤖 Loading TumorSizePredictor...")
    predictor = TumorSizePredictor()
    
    # Predict size using T2 image only
    print("🔮 Predicting tumor size...")
    result = predictor.predict(t2_image=img_norm)
    
    print(f"\n✅ Results:")
    print(f"   Predicted Size: {result['predicted_size']:.2f} mm")
    print(f"   Confidence: {result.get('confidence', 'N/A')}")
    

def demo_2_multi_sequence():
    """Demo 2: Multi-sequence prediction (T2, ADC, DWI)"""
    print("\n" + "="*70)
    print("DEMO 2: Multi-Sequence Tumor Analysis")
    print("="*70)
    
    import numpy as np
    from utils_image import normalize_image
    from size_predictor_model import TumorSizePredictor
    
    print("\n📊 Creating simulated multi-sequence MRI data...")
    
    # Create sample sequences
    t2_img = np.random.rand(256, 256) * 0.6 + 0.2
    adc_img = np.random.rand(256, 256) * 0.4 + 0.1
    dwi_img = np.random.rand(256, 256) * 0.5 + 0.3
    
    # Normalize
    t2_norm = normalize_image(t2_img)
    adc_norm = normalize_image(adc_img)
    dwi_norm = normalize_image(dwi_img)
    
    # Predict with all sequences
    print("🤖 Loading predictor...")
    predictor = TumorSizePredictor()
    
    print("🔮 Predicting with T2 + ADC + DWI...")
    result = predictor.predict(
        t2_image=t2_norm,
        adc_image=adc_norm,
        dwi_image=dwi_norm
    )
    
    print(f"\n✅ Results:")
    print(f"   Size: {result['predicted_size']:.2f} mm")
    print(f"   Confidence: {result.get('confidence', 'N/A')}")
    

def demo_3_bounding_box():
    """Demo 3: Bounding box generation"""
    print("\n" + "="*70)
    print("DEMO 3: Bounding Box Generation")
    print("="*70)
    
    import numpy as np
    from utils_image import normalize_image
    from bbox_utils import BoundingBoxGenerator
    
    # Create sample image
    sample_image = np.ones((256, 256)) * 0.3
    img_norm = normalize_image(sample_image)
    
    print("\n📦 Initializing BoundingBoxGenerator...")
    bbox_gen = BoundingBoxGenerator()
    
    # Test different tumor sizes
    sizes = [8, 15, 30, 60]
    
    for size in sizes:
        print(f"\n🎯 Generating bbox for {size}mm tumor...")
        
        # Circular bbox
        circle_bbox = bbox_gen.generate_bbox(
            image=img_norm,
            tumor_size_mm=size,
            bbox_type='circle'
        )
        
        # Rectangular bbox
        rect_bbox = bbox_gen.generate_bbox(
            image=img_norm,
            tumor_size_mm=size,
            bbox_type='rect'
        )
        
        print(f"   Circle: {circle_bbox['bbox_coords']}")
        print(f"   Rect:   {rect_bbox['bbox_coords']}")
        

def demo_4_severity_classification():
    """Demo 4: Severity classification"""
    print("\n" + "="*70)
    print("DEMO 4: Severity Classification")
    print("="*70)
    
    sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
    from demo_tumor_complete import TumorSeverityClassifier
    
    print("\n⚕️  Classifying tumors by size...")
    
    test_sizes = [5, 12, 25, 55]
    
    for size_mm in test_sizes:
        severity = TumorSeverityClassifier.classify(size_mm)
        
        print(f"\n  Size: {size_mm:.1f}mm")
        print(f"    Stage: {severity['severity']}")
        print(f"    Range: {severity['range']}")
        print(f"    Desc:  {severity['description']}")
        

def demo_5_visualization():
    """Demo 5: Create visualization with bounding box"""
    print("\n" + "="*70)
    print("DEMO 5: Visualization with Annotations")
    print("="*70)
    
    try:
        import numpy as np
        import cv2
        from utils_image import normalize_image
        from bbox_utils import BoundingBoxGenerator, VisualizationHelper
        
        print("\n🎨 Creating visualization...")
        
        # Create sample image
        img = np.ones((300, 300)) * 0.4
        img_norm = normalize_image(img)
        
        # Generate bbox
        bbox_gen = BoundingBoxGenerator()
        bbox = bbox_gen.generate_bbox(img_norm, tumor_size_mm=20, bbox_type='circle')
        
        # Visualize
        vis_helper = VisualizationHelper()
        
        # Convert to BGR for visualization
        img_bgr = cv2.cvtColor((img_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Draw bbox
        x1, y1, x2, y2 = bbox['bbox_coords']
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img_bgr, "Tumor: 20.0mm T2", (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save
        output_path = "demo_visualization.png"
        cv2.imwrite(output_path, img_bgr)
        
        print(f"   ✅ Visualization saved: {output_path}")
        
    except Exception as e:
        print(f"   ⚠️  Skipping visualization (dependencies needed): {e}")
        

def main():
    """Run all demos"""
    
    print("\n" + "="*70)
    print("🏥 TUMOR SIZE PREDICTION - QUICK START DEMOS")
    print("="*70)
    print("\nThis script demonstrates the complete tumor analysis system:")
    print("  1. Basic size prediction")
    print("  2. Multi-sequence analysis (T2, ADC, DWI)")
    print("  3. Bounding box generation")
    print("  4. Severity classification")
    print("  5. Visualization with annotations")
    
    try:
        demo_1_basic_prediction()
    except Exception as e:
        print(f"\n❌ Demo 1 failed: {e}")
    
    try:
        demo_2_multi_sequence()
    except Exception as e:
        print(f"\n❌ Demo 2 failed: {e}")
    
    try:
        demo_3_bounding_box()
    except Exception as e:
        print(f"\n❌ Demo 3 failed: {e}")
    
    try:
        demo_4_severity_classification()
    except Exception as e:
        print(f"\n❌ Demo 4 failed: {e}")
    
    try:
        demo_5_visualization()
    except Exception as e:
        print(f"\n❌ Demo 5 failed: {e}")
    
    print("\n" + "="*70)
    print("✅ DEMOS COMPLETE")
    print("="*70)
    print("\n📚 Next Steps:")
    print("   1. Review TUMOR_SIZE_COMPLETE_GUIDE.md for full documentation")
    print("   2. Run: python scripts/demo_tumor_complete.py (full analysis)")
    print("   3. Start server: python webapp/fastapi_server.py --port 8000")
    print("   4. Test API: python scripts/test_tumor_api.py")
    print("   5. Train model: python scripts/train_size_model.py --epochs 100")


if __name__ == '__main__':
    main()
