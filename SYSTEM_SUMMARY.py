#!/usr/bin/env python3
"""
Display comprehensive system summary and status
"""

import os
import sys
from pathlib import Path

def print_banner():
    """Print beautiful banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║              🏥 PROSTATE TUMOR SIZE PREDICTION SYSTEM 🏥                       ║
║                      Complete Implementation Guide                            ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_system_overview():
    """Print system overview"""
    overview = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ SYSTEM OVERVIEW                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

The Prostate Tumor Size Prediction System provides:

✓ Multi-Sequence MRI Analysis
  • T2-weighted imaging (anatomical reference)
  • ADC maps (apparent diffusion coefficient)
  • DWI imaging (diffusion-weighted imaging)

✓ Tumor Size Prediction
  • Deep learning model (ResNet-50 backbone)
  • Continuous output: tumor size in millimeters (mm)
  • Multi-scale attention mechanisms
  • Transfer learning support

✓ Bounding Box Generation
  • Automatic detection and localization
  • Pixel-to-millimeter conversion
  • Visualization overlay

✓ Severity Classification
  • TNM staging (T1a, T1b, T1c, T2, T3+)
  • Based on WHO/AJCC guidelines
  • Integrated severity descriptors

✓ Multiple Interfaces
  • Command-line inference
  • FastAPI REST endpoints
  • Python API client
  • Batch processing

"""
    print(overview)


def print_key_components():
    """Print key components"""
    components = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ KEY COMPONENTS                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Core Models & Utilities:
  📄 src/size_predictor_model.py       - Main prediction model
  📄 src/bbox_utils.py                 - Bounding box generation
  📄 src/infer_with_bbox.py            - Inference pipeline
  📄 src/utils_image.py                - Image utilities
  📄 src/utils_dicom.py                - DICOM handling

Training & Evaluation:
  📄 src/train_size_model.py           - Training script
  📄 scripts/batch_predict_tumor_size.py - Batch inference
  📄 scripts/training_guide_tumor_size.py - Training documentation

API & Server:
  📄 webapp/fastapi_server.py          - REST API endpoints
  📄 scripts/api_client_tumor_size.py  - Python API client

Pipeline Scripts:
  📄 scripts/run_tumor_size_pipeline.py - Complete inference pipeline
  📄 scripts/demo_tumor_size_prediction.py - Interactive demo

Models (Pre-trained):
  📁 models/baseline_real_t2_adc_3s_ep1.pth - Multi-sequence model
  📁 models/prototype_toy.pth               - Test model

Datasets:
  📁 data/PROSTATEx/                   - Raw DICOM data
  📁 seg_dataset_*/                    - Segmentation datasets
  📁 yolo_dataset*/                    - YOLO training data

"""
    print(components)


def print_quick_start():
    """Print quick start guide"""
    quickstart = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ QUICK START                                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

1️⃣  SINGLE CASE PREDICTION
   ─────────────────────────────────────────────────────────────────────────
   
   Command:
   $ python scripts/run_tumor_size_pipeline.py t2.png adc.png dwi.png
   
   Output:
   • Predicted tumor size in mm
   • TNM classification
   • Bounding box coordinates
   • Visualization with bounding box

2️⃣  BATCH PROCESSING
   ─────────────────────────────────────────────────────────────────────────
   
   Command:
   $ python scripts/batch_predict_tumor_size.py \\
       --input-dir data/test \\
       --output-dir results/ \\
       --save-viz
   
   Or from CSV:
   $ python scripts/batch_predict_tumor_size.py \\
       --input-csv cases.csv \\
       --output-dir results/
   
   Output:
   • predictions.csv with results
   • Visualization images (optional)
   • Statistics summary

3️⃣  API SERVER
   ─────────────────────────────────────────────────────────────────────────
   
   Start Server:
   $ python -m uvicorn webapp.fastapi_server:app --reload
   
   API Endpoint:
   POST /predict-size
   
   Request Format:
   {
     "t2_image_base64": "...",
     "adc_image_base64": "...",
     "dwi_image_base64": "..."
   }
   
   Response Format:
   {
     "tumor_size_mm": 15.4,
     "tnm_stage": "T1b",
     "severity": "Medium (10-20mm)",
     "bbox": [x1, y1, x2, y2],
     "confidence": 0.92
   }

4️⃣  PYTHON API CLIENT
   ─────────────────────────────────────────────────────────────────────────
   
   Code Example:
   ```python
   from scripts.api_client_tumor_size import TumorSizePredictionClient
   
   client = TumorSizePredictionClient("http://localhost:8000")
   result = client.predict_from_files("t2.png", "adc.png", "dwi.png")
   
   print(f"Tumor Size: {result['tumor_size_mm']:.2f} mm")
   print(f"Severity: {result['severity']}")
   ```

5️⃣  TRAINING NEW MODEL
   ─────────────────────────────────────────────────────────────────────────
   
   Prepare Training Data:
   $ python scripts/training_guide_tumor_size.py
   
   Run Training:
   $ python src/train_size_model.py \\
       --data-dir data/ \\
       --epochs 100 \\
       --batch-size 32 \\
       --save-dir models/my_model/

"""
    print(quickstart)


def print_api_reference():
    """Print API reference"""
    api_ref = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ API REFERENCE                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

Endpoints Available:

1. Tumor Size Prediction
   ─────────────────────────────────────────────────────────────────────────
   
   POST /predict-size
   
   Request:
   {
     "t2_image_base64": "string",      # Base64 encoded T2 image
     "adc_image_base64": "string",     # Base64 encoded ADC image
     "dwi_image_base64": "string"      # Base64 encoded DWI image
   }
   
   Response:
   {
     "success": true,
     "tumor_size_mm": 18.5,            # Predicted size in mm
     "tnm_stage": "T1b",               # TNM classification
     "severity": "Medium (10-20mm)",   # Severity description
     "bbox": [100, 150, 250, 300],     # Bounding box [x1,y1,x2,y2]
     "confidence": 0.87,               # Confidence score
     "processing_time_ms": 245
   }

2. Health Check
   ─────────────────────────────────────────────────────────────────────────
   
   GET /health
   
   Response:
   {
     "status": "healthy",
     "timestamp": "2025-12-04T10:30:00Z"
   }

3. Model Info
   ─────────────────────────────────────────────────────────────────────────
   
   GET /model-info
   
   Response:
   {
     "model_name": "TumorSizePredictorMultiSeq",
     "input_sequences": ["T2", "ADC", "DWI"],
     "output_format": "tumor_size_mm",
     "tnm_stages": ["T1a", "T1b", "T1c", "T2", "T3+"]
   }

Error Codes:

400 Bad Request
  • Missing required fields
  • Invalid image format
  • Image decode error

422 Unprocessable Entity
  • Invalid request format
  • Image size mismatch

500 Internal Server Error
  • Model inference error
  • GPU memory issues

"""
    print(api_ref)


def print_tnm_classification():
    """Print TNM classification guide"""
    tnm_guide = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ TNM CLASSIFICATION GUIDE                                                    │
└─────────────────────────────────────────────────────────────────────────────┘

Classification Based on Tumor Size (WHO/AJCC):

┌───────────────────────────────────────────────────────────────────────────┐
│ TNM Stage │ Size Range    │ Category            │ Clinical Features       │
├───────────────────────────────────────────────────────────────────────────┤
│ T1a       │ <10 mm        │ Small               │ Early detection         │
│           │               │                     │ Excellent prognosis     │
├───────────────────────────────────────────────────────────────────────────┤
│ T1b       │ 10-20 mm      │ Medium              │ Localized               │
│           │               │                     │ Good prognosis          │
├───────────────────────────────────────────────────────────────────────────┤
│ T1c       │ 20-30 mm      │ Medium-Large        │ Localized growth        │
│           │               │                     │ Moderate prognosis      │
├───────────────────────────────────────────────────────────────────────────┤
│ T2        │ 30-50 mm      │ Large               │ Clinically significant  │
│           │               │                     │ Requires treatment      │
├───────────────────────────────────────────────────────────────────────────┤
│ T3+       │ >50 mm        │ Very Large/Advanced │ Advanced disease        │
│           │               │                     │ Aggressive intervention │
└───────────────────────────────────────────────────────────────────────────┘

Key Points:
  • T1a: Incidental finding, smallest measurable
  • T1b-T1c: Localized, contained within prostate
  • T2: Extends beyond prostate capsule
  • T3+: Involves seminal vesicles or other structures

Clinical Implications:
  • T1a/T1b: Monitoring or minimally invasive treatment
  • T1c: Active surveillance or focal therapy
  • T2: Radical treatment often indicated
  • T3+: Multimodal treatment required

"""
    print(tnm_guide)


def print_performance_metrics():
    """Print expected performance metrics"""
    perf = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ EXPECTED PERFORMANCE METRICS                                                │
└─────────────────────────────────────────────────────────────────────────────┘

Model Performance (with adequate training data):

Size Prediction Accuracy:
  • MAE (Mean Absolute Error):      ±2-3 mm (good)
  • RMSE:                            3-5 mm (good)
  • Correlation (R²):               0.85-0.95

Bounding Box Accuracy:
  • IoU (Intersection over Union):  0.75-0.85
  • Dice Coefficient:               0.80-0.90

TNM Classification:
  • Accuracy:                       90-95%
  • Precision:                      90-93%
  • Recall:                         85-92%

Inference Speed:
  • Single case:                    0.5-2 seconds (GPU)
  • Single case:                    2-5 seconds (CPU)
  • Batch (32 cases):               10-20 seconds (GPU)

Factors Affecting Performance:
  ✓ Training data quality and quantity
  ✓ Image resolution and preprocessing
  ✓ Radiologist measurement accuracy (ground truth)
  ✓ Inter-observer variability
  ✓ Patient diversity in training set
  ✓ Hardware (GPU vs CPU)

"""
    print(perf)


def print_troubleshooting():
    """Print troubleshooting guide"""
    troubleshoot = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ TROUBLESHOOTING & FAQ                                                       │
└─────────────────────────────────────────────────────────────────────────────┘

Problem: "Model file not found"
Solution:
  • Check model path: models/baseline_real_t2_adc_3s_ep1.pth
  • Ensure model is downloaded or trained
  • Verify file permissions

Problem: "CUDA out of memory"
Solution:
  • Reduce batch size
  • Use CPU: CUDA_VISIBLE_DEVICES="" python ...
  • Clear GPU cache: nvidia-smi

Problem: "Images have wrong shape"
Solution:
  • Verify images are 2D grayscale
  • Check image dimensions match model input
  • Resize if needed: cv2.resize(img, (256, 256))

Problem: "API returns 500 error"
Solution:
  • Check server logs
  • Verify image encoding (base64)
  • Check image dimensions
  • Monitor GPU memory usage

Problem: "Predictions seem inaccurate"
Solution:
  • Verify ground truth labels
  • Check image preprocessing
  • Use cross-validation
  • Consider transfer learning or retraining
  • Validate with radiologists

Problem: "High variance between runs"
Solution:
  • Set random seed: np.random.seed(42), torch.manual_seed(42)
  • Check data augmentation intensity
  • Verify test set consistency
  • Monitor training stability

Problem: "Slow inference on CPU"
Solution:
  • Use GPU if available
  • Optimize model size
  • Use batch processing
  • Consider quantization

"""
    print(troubleshoot)


def print_file_structure():
    """Print directory structure"""
    structure = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ PROJECT STRUCTURE                                                           │
└─────────────────────────────────────────────────────────────────────────────┘

prostate-severity/
│
├── src/                           # Core implementation
│   ├── size_predictor_model.py    # Main model
│   ├── bbox_utils.py              # Bounding box utilities
│   ├── infer_with_bbox.py         # Inference pipeline
│   ├── train_size_model.py        # Training script
│   ├── utils_image.py             # Image utilities
│   ├── utils_dicom.py             # DICOM utilities
│   └── ...
│
├── scripts/                       # Executable scripts
│   ├── run_tumor_size_pipeline.py     # Main inference
│   ├── batch_predict_tumor_size.py    # Batch processing
│   ├── api_client_tumor_size.py       # API client
│   ├── training_guide_tumor_size.py   # Training guide
│   ├── demo_tumor_size_prediction.py  # Demo
│   └── ...
│
├── webapp/                        # Web server
│   ├── fastapi_server.py          # REST API
│   ├── streamlit_demo.py          # Streamlit UI
│   └── ...
│
├── models/                        # Pre-trained models
│   ├── baseline_real_t2_adc_3s_ep1.pth
│   ├── yolov8s_prostate_smoke/
│   └── ...
│
├── data/                          # Training & test data
│   ├── PROSTATEx/                 # DICOM dataset
│   ├── metadata.csv
│   └── ...
│
└── output/                        # Results directory (created on run)
    ├── predictions.csv
    ├── tumor_with_bbox.png
    └── ...

"""
    print(structure)


def print_next_steps():
    """Print next steps"""
    steps = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ NEXT STEPS & RECOMMENDATIONS                                                │
└─────────────────────────────────────────────────────────────────────────────┘

Immediate:
  1. ✓ Review system components
  2. ✓ Run single case prediction: python scripts/run_tumor_size_pipeline.py
  3. ✓ Start API server: python -m uvicorn webapp.fastapi_server:app
  4. ✓ Test API with client script

Short-term (1-2 weeks):
  1. Train on your dataset: python src/train_size_model.py
  2. Validate predictions with radiologists
  3. Measure performance metrics
  4. Create clinical validation protocol

Medium-term (1-3 months):
  1. Integrate into clinical workflow
  2. Deploy to production server
  3. Set up monitoring and logging
  4. Create user documentation

Long-term (3+ months):
  1. Continuous model improvement
  2. Multi-center validation
  3. FDA/CE approval (if applicable)
  4. Knowledge base of edge cases

Performance Optimization:
  1. Profile inference time
  2. Consider model quantization
  3. Implement batch processing
  4. Use GPU acceleration

Robustness Improvement:
  1. Test on diverse patient populations
  2. Validate cross-institutional data
  3. Add uncertainty quantification
  4. Implement confidence thresholds

"""
    print(steps)


def main():
    """Main entry point"""
    print_banner()
    print_system_overview()
    print_key_components()
    print_quick_start()
    print_api_reference()
    print_tnm_classification()
    print_performance_metrics()
    print_file_structure()
    print_troubleshooting()
    print_next_steps()
    
    # Print footer
    footer = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║  For detailed documentation, see:                                            ║
║  • TUMOR_SIZE_COMPLETE_GUIDE.md                                              ║
║  • README.md                                                                 ║
║  • scripts/training_guide_tumor_size.py                                      ║
║                                                                               ║
║  Questions or Issues?                                                        ║
║  • Check troubleshooting guide above                                         ║
║  • Review example scripts in scripts/                                        ║
║  • Run demo: python scripts/demo_tumor_size_prediction.py                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

"""
    print(footer)


if __name__ == "__main__":
    main()
