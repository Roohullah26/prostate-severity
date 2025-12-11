"""
TUMOR SIZE PREDICTION - QUICK START GUIDE

Complete system for:
✓ Multi-sequence MRI tumor size prediction (T2/ADC/DWI)
✓ Precise bounding box generation (circular & rectangular)
✓ Automated severity grading (T1/T2/T3/T4)
✓ Confidence scoring
✓ REST API endpoints
✓ Interactive web interface

=============================================================================
SETUP
=============================================================================

1. Install dependencies:
   pip install -r requirements.txt

2. Train the size predictor (optional, trained models available):
   python scripts/auto_train_unet.py --epochs 50 --lr 0.001

   OR use pre-trained model:
   # Model already available at: models/tumor_size_predictor_best.pth

=============================================================================
USAGE - LOCAL INFERENCE
=============================================================================

Option 1: Simple Demo (Synthetic Data)
---------------------------------------
python scripts/demo_complete_pipeline.py --toy

Output:
- synthetic_prediction.png       # Visualization with bounding boxes
- Terminal output showing size, severity, and probabilities


Option 2: Real Dataset Demo
----------------------------
python scripts/demo_complete_pipeline.py --csv merged_data.csv --sample 0

Output:
- prediction_ProstateX-XXXX.png   # Visualization
- prediction_ProstateX-XXXX.json  # Full prediction data


Option 3: Process Specific Patient
-----------------------------------
python scripts/demo_complete_pipeline.py --csv merged_data.csv --uid ProstateX-0000


Option 4: Python API (Programmatic)
------------------------------------

from pathlib import Path
from PIL import Image
from src.infer_with_bbox import TumorInferencePipeline

# Initialize
pipeline = TumorInferencePipeline(
    model_path='models/tumor_size_predictor_best.pth',
    device='cuda'  # or 'cpu'
)

# Load image (or create from DICOM)
image = Image.open('mri_image.png')

# Predict
prediction = pipeline.predict_from_image(image)

print(f"Tumor Size: {prediction.width_mm:.1f} x {prediction.height_mm:.1f} x {prediction.depth_mm:.1f} mm")
print(f"Severity: {prediction.severity}")
print(f"Confidence: {prediction.confidence:.2%}")

=============================================================================
USAGE - REST API
=============================================================================

1. Start the server:
   python -m webapp.fastapi_server --port 8000

   Output: 
   INFO:     Uvicorn running on http://127.0.0.1:8000
   INFO:     Press CTRL+C to quit


2. Test with curl:
   curl -X POST "http://localhost:8000/predict-size" \
     -F "file=@test_image.png" \
     -F "bbox_type=circle" \
     -F "return_image=true"


3. Python client:
   import requests
   
   with open('mri_image.png', 'rb') as f:
       files = {'file': f}
       params = {
           'bbox_type': 'circle',
           'return_image': True,
       }
       
       response = requests.post(
           'http://localhost:8000/predict-size',
           files=files,
           params=params
       )
       
       result = response.json()
       print(f"Severity: {result['severity']}")
       print(f"Size: {result['width_mm']:.1f}mm x {result['height_mm']:.1f}mm")


4. Test the API:
   python scripts/test_api_complete.py --server-url http://localhost:8000
   python scripts/test_api_complete.py --csv merged_data.csv --concurrent 5


5. Interactive API docs:
   Open browser to: http://localhost:8000/docs

=============================================================================
API ENDPOINTS
=============================================================================

POST /predict-size
  Description: Predict tumor size and severity with bounding box
  
  Request:
    - file: MRI image (PNG/JPG)
    - bbox_type: 'circle' or 'rect' (default: 'circle')
    - return_image: Include base64 visualization (default: false)
    - pixel_spacing: JSON array [row_mm, col_mm]
    - model_path: Path to model weights (or set PROSTATE_SIZE_MODEL env var)
  
  Response:
    {
      "severity": "T2",
      "width_mm": 18.5,
      "height_mm": 22.1,
      "depth_mm": 19.3,
      "max_dimension_mm": 22.1,
      "confidence": 0.92,
      "severity_probabilities": {
        "T1": 0.05,
        "T2": 0.70,
        "T3": 0.20,
        "T4": 0.05
      },
      "bbox": {
        "type": "circle",
        "center_x": 112,
        "center_y": 112,
        "radius_px": 25,
        "radius_mm": 11.1,
        "diameter_mm": 22.1
      },
      "visualization_base64": "iVBORw0KGg..."  # if return_image=true
    }

GET /health
  Description: Server health check
  Response: {"ok": true}

=============================================================================
SEVERITY GRADES & THRESHOLDS
=============================================================================

T1: ≤ 20mm   (Green 🟢)    - Small tumor, low risk
T2: 20-40mm  (Yellow 🟡)   - Intermediate tumor, moderate risk  
T3: 40-60mm  (Orange 🟠)   - Large tumor, high risk
T4: > 60mm   (Red 🔴)      - Very large tumor, very high risk

Classification is automatic based on maximum dimension.

=============================================================================
OUTPUT FILES
=============================================================================

Inference produces:
- PNG image with bounding boxes (rectangular + circular)
- JSON file with complete prediction data
- Server returns base64-encoded image via API

PNG annotations include:
✓ Patient ID
✓ Severity grade + description
✓ Tumor dimensions in mm
✓ Confidence score
✓ All severity probabilities
✓ Circular bounding box (maximum dimension)
✓ Rectangular bounding box (W x H)

=============================================================================
ADVANCED USAGE
=============================================================================

1. Custom Model Path:
   python scripts/demo_complete_pipeline.py --toy --model-path custom_model.pth

2. Specific Device:
   python scripts/demo_complete_pipeline.py --toy --device cpu

3. Batch Processing:
   Create a script to loop through dataset:
   
   from src.prostate_dataset import ProstateLesionDataset
   
   dataset = ProstateLesionDataset('merged_data.csv', sequences=['t2','adc','dwi'])
   
   for sample in dataset:
       prediction = pipeline.predict_from_image(sample['img'])
       # Process prediction...

4. DICOM Integration:
   # Load DICOM files and stack multi-sequence images
   from src.utils_dicom import load_dicom_sequence
   
   t2_img = load_dicom_sequence('patient_001_t2.dcm')
   adc_img = load_dicom_sequence('patient_001_adc.dcm')
   dwi_img = load_dicom_sequence('patient_001_dwi.dcm')
   
   # Stack and resize to 224x224
   stacked = np.stack([t2_img, adc_img, dwi_img], axis=0)
   # ... resize and predict

=============================================================================
TROUBLESHOOTING
=============================================================================

Q: Model weights not found
A: Set environment variable: 
   export PROSTATE_SIZE_MODEL=models/tumor_size_predictor_best.pth

Q: API server not starting
A: Check port is available:
   netstat -an | grep 8000
   
Q: CUDA out of memory
A: Use CPU inference:
   python scripts/demo_complete_pipeline.py --toy --device cpu

Q: Low prediction confidence
A: - Model may need retraining on your data
   - Ensure images are properly preprocessed (224x224)
   - Check pixel spacing if using DICOM

=============================================================================
MODEL ARCHITECTURE
=============================================================================

Input:  (B, 3, 224, 224) - Multi-sequence MRI stack (T2/ADC/DWI)
Backbone: ResNet18 (ImageNet pretrained)

Outputs:
  1. Size Head: (B, 3) - [width_mm, height_mm, depth_mm]
  2. Severity Head: (B, 4) - Softmax logits for [T1, T2, T3, T4]
  3. Confidence Head: (B, 1) - Confidence score [0, 1]

Training Loss:
  L_total = α * MSE(size) + β * CrossEntropy(severity) + γ * BCE(confidence)
  
  α=1.0 (size loss weight)
  β=0.5 (severity loss weight)
  γ=0.1 (confidence loss weight)

=============================================================================
FILES & STRUCTURE
=============================================================================

Key Files:
  src/size_predictor_model.py        - TumorSizePredictor model class
  src/bbox_utils.py                  - Bounding box generation & visualization
  src/infer_with_bbox.py             - End-to-end inference pipeline
  webapp/fastapi_server.py           - REST API server
  scripts/demo_complete_pipeline.py  - Demo script
  scripts/test_api_complete.py       - API testing suite
  models/tumor_size_predictor_best.pth - Pre-trained model

Related Files:
  src/prostate_dataset.py            - Dataset loading
  src/train_size_model.py            - Training script
  src/utils_image.py                 - Image preprocessing
  src/utils_dicom.py                 - DICOM file handling

=============================================================================
NEXT STEPS
=============================================================================

1. ✓ Run the demo:
   python scripts/demo_complete_pipeline.py --toy

2. ✓ Start the API server:
   python -m webapp.fastapi_server

3. ✓ Test the API:
   python scripts/test_api_complete.py

4. ✓ Integrate into your application:
   # Use REST API or Python API as shown above

5. Optional: Retrain on your specific data:
   python scripts/auto_train_unet.py --epochs 100

=============================================================================
"""

if __name__ == '__main__':
    print(__doc__)
