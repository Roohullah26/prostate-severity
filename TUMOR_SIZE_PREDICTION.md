# Tumor Size Prediction & Severity Classification System

## Overview

This system provides an end-to-end solution for predicting prostate tumor dimensions and severity classification from multi-sequence MRI data (T2, ADC, DWI). It uses deep learning to:

1. **Predict tumor dimensions** in millimeters (width, height, depth)
2. **Classify severity** based on size (T1, T2, T3, T4)
3. **Generate bounding boxes** (circular or rectangular) for visualization
4. **Provide confidence scores** for each prediction

## Quick Start

### 1. Try the Demo (No Model Training Required)

```bash
# Toy dataset demo with untrained model
python scripts/demo_tumor_size_prediction.py --toy --output demo_results/
```

This will demonstrate all features without requiring a pretrained model.

### 2. Setup FastAPI Server

```bash
# Install dependencies (if not already done)
pip install -r requirements.txt

# Start the server
python -m uvicorn webapp.fastapi_server:app --reload --port 8000

# Or use the provided script
python scripts/run_streamlit.ps1  # Windows
bash scripts/run_streamlit.sh     # Linux/Mac
```

### 3. Use the API

```python
import requests
from PIL import Image

# Prepare image
img = Image.open("mri_image.png").resize((224, 224))
img.save("temp.png", "PNG")

# Make prediction
with open("temp.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict-size",
        files={"file": f},
        data={
            "model_path": "models/tumor_size_predictor_best.pth",
            "bbox_type": "circle",
            "return_image": True
        }
    )

result = response.json()
print(f"Severity: {result['severity']}")
print(f"Size: {result['width_mm']:.1f} x {result['height_mm']:.1f} x {result['depth_mm']:.1f} mm")
print(f"Confidence: {result['confidence']:.3f}")
```

## Model Architecture

### Overview

```
Input (B, 3, 224, 224)
    │ Multi-sequence MRI stack (T2/ADC/DWI)
    ↓
ResNet18 Backbone (pretrained on ImageNet)
    │ Feature extraction: (B, 512)
    ├─────────┬──────────┬─────────┐
    ↓         ↓          ↓         ↓
  Size      Severity  Confidence  (other heads)
  Head      Head      Head
    │         │          │
    ↓         ↓          ↓
 (B,3)     (B,4)      (B,1)
 [mm]      [logits]   [0,1]
    │         │          │
    └─────────┴──────────┘
         Output
```

### Model Details

- **Backbone**: ResNet18 (ImageNet-pretrained)
- **Input Channels**: 3 (stacked T2, ADC, DWI)
- **Input Size**: 224×224 pixels
- **Feature Dimension**: 512
- **Output Heads**:
  - **Size Head**: Predicts 3D dimensions [width_mm, height_mm, depth_mm]
  - **Severity Head**: Predicts logits for [T1, T2, T3, T4]
  - **Confidence Head**: Predicts prediction confidence [0, 1]

### Loss Function

Combined multi-task loss:
```
Loss = α * MSE(size) + β * CrossEntropy(severity) + γ * BCE(confidence)
```

Default weights: α=1.0, β=0.5, γ=0.1

## Severity Classification

Severity is classified based on the maximum tumor dimension:

| Grade | Size Range | Color | Clinical Description |
|-------|-----------|-------|----------------------|
| T1    | ≤ 20 mm   | 🟢 Green  | Small, clinically insignificant |
| T2    | 20-40 mm  | 🟡 Yellow | Medium, locally confined |
| T3    | 40-60 mm  | 🟠 Orange | Large, may extend beyond organ |
| T4    | > 60 mm   | 🔴 Red    | Very large, extensive invasion |

## API Endpoints

### `/predict-size` (POST)

Predict tumor size with bounding box visualization.

**Parameters:**
- `file` (required): MRI image file (PNG, JPG, etc.)
- `model_path` (optional): Path to model weights (or set env var `PROSTATE_SIZE_MODEL`)
- `bbox_type` (optional): `"circle"` or `"rect"` (default: `"circle"`)
- `pixel_spacing` (optional): JSON string `"[row_mm, col_mm]"` for DICOM pixel spacing
- `return_image` (optional): If `true`, return base64-encoded visualization (default: `false`)

**Response:**
```json
{
  "severity": "T2",
  "width_mm": 18.5,
  "height_mm": 22.3,
  "depth_mm": 20.1,
  "max_dimension_mm": 22.3,
  "confidence": 0.94,
  "severity_probabilities": {
    "T1": 0.03,
    "T2": 0.82,
    "T3": 0.12,
    "T4": 0.03
  },
  "bbox": {
    "type": "circle",
    "center_x": 112,
    "center_y": 112,
    "radius_px": 22,
    "radius_mm": 11.15,
    "diameter_mm": 22.3
  },
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### Other Endpoints

- **`/health`** (GET): Health check
- **`/model_status`** (GET): Model and system status
- **`/series_status`** (GET): DICOM series information
- **`/quick_eval`** (GET): Quick model evaluation
- **`/predict`** (POST): Legacy binary classification endpoint

## Python Usage

### Basic Inference

```python
from src.infer_with_bbox import TumorInferencePipeline
from PIL import Image

# Initialize pipeline
pipeline = TumorInferencePipeline(
    model_path="models/tumor_size_predictor_best.pth",
    device="cuda"
)

# Load image
image = Image.open("mri_slice.png").resize((224, 224))

# Predict with visualization
result = pipeline.predict_with_visualization(
    image, 
    bbox_type="circle"
)

prediction = result['prediction']
print(f"Severity: {prediction.severity}")
print(f"Size: {prediction.width_mm:.1f} x {prediction.height_mm:.1f} mm")
print(f"Confidence: {prediction.confidence:.3f}")

# Save visualization
vis_image = result['image']
vis_image.save("prediction_visualization.png")
```

### Batch Inference

```python
from src.infer_with_bbox import TumorInferencePipeline
from PIL import Image
import glob

pipeline = TumorInferencePipeline("models/tumor_size_predictor_best.pth")

# Load multiple images
images = [Image.open(f).resize((224, 224)) for f in glob.glob("mri_data/*.png")]

# Batch predict
results = pipeline.predict_batch(images, bbox_type="circle")

# Process results
for i, result in enumerate(results):
    print(f"Image {i}: {result['prediction'].severity} - "
          f"{result['prediction'].confidence:.3f}")
```

### Bounding Box Generation

```python
from src.bbox_utils import BoundingBoxGenerator, VisualizationHelper, TumorPrediction
import numpy as np

# Create prediction object
pred = TumorPrediction(
    width_mm=18.5,
    height_mm=22.3,
    depth_mm=20.1,
    severity='T2',
    severity_logits=np.array([0.03, 0.82, 0.12, 0.03]),
    confidence=0.94,
    image_size=(224, 224),
    pixel_spacing_mm=(1.0, 1.0)
)

# Generate bounding boxes
bbox_gen = BoundingBoxGenerator()

# Circular bbox
circ_bbox = bbox_gen.get_circular_bbox(pred)
print(f"Circle radius: {circ_bbox['radius_px']} px")

# Rectangular bbox
rect_bbox = bbox_gen.get_rectangular_bbox(pred)
print(f"Rectangle: ({rect_bbox['x1']}, {rect_bbox['y1']}) to "
      f"({rect_bbox['x2']}, {rect_bbox['y2']})")

# Visualize
from PIL import Image
vis_helper = VisualizationHelper()
img = Image.open("mri_image.png")

# Draw circular bbox
vis_circ = vis_helper.draw_circular_bbox_pil(img, circ_bbox, pred)
vis_circ.save("prediction_circle.png")

# Draw rectangular bbox
vis_rect = vis_helper.draw_rectangular_bbox_pil(img, rect_bbox, pred)
vis_rect.save("prediction_rect.png")
```

## Training

### Dataset Preparation

Your data should have tumor dimensions labeled:

```csv
SeriesInstanceUID,PatientID,tumor_width_mm,tumor_height_mm,tumor_depth_mm,severity_grade,ClinSig
ProstateX-0000,0,15.2,18.5,14.1,1,1
ProstateX-0001,1,22.3,25.0,20.1,2,0
...
```

Where `severity_grade`: 0=T1, 1=T2, 2=T3, 3=T4

### Training Command

```bash
# Train on real data
python -m src.train_size_model \
    --csv merged_data.csv \
    --size-csv tumor_sizes.csv \
    --sequences t2,adc,dwi \
    --epochs 20 \
    --bs 8 \
    --lr 0.001

# Toy training (for testing)
python -m src.train_size_model \
    --toy \
    --epochs 5 \
    --bs 16
```

**Expected Output**: `models/tumor_size_predictor_best.pth`

### Training Options

```
--csv                  Path to merged_data.csv
--size-csv            Path to CSV with size annotations
--sequences           Sequences to use (t2,adc,dwi)
--epochs              Number of training epochs
--bs                  Batch size
--lr                  Learning rate
--wd                  Weight decay
--num-slices          Number of slices per sample
--output-dir          Directory to save checkpoints
--toy                 Use toy dataset
--toy-len             Toy dataset size
```

## Inference

### Command Line

```bash
# Toy inference
python -m src.infer_with_bbox \
    --model-path models/tumor_size_predictor_best.pth \
    --toy \
    --output results/

# Real data inference
python -m src.infer_with_bbox \
    --model-path models/tumor_size_predictor_best.pth \
    --csv merged_data.csv \
    --sequences t2,adc,dwi \
    --output results/ \
    --max-samples 100
```

### Script

```bash
# Run the comprehensive demo
python scripts/demo_tumor_size_prediction.py \
    --model-path models/tumor_size_predictor_best.pth \
    --csv merged_data.csv \
    --output demo_results/
```

## Output Files

### Predictions

Results are saved as JSON:

```json
[
  {
    "sample_id": 0,
    "width_mm": 18.5,
    "height_mm": 22.3,
    "depth_mm": 20.1,
    "severity": "T2",
    "confidence": 0.94,
    "severity_probs": {
      "T1": 0.03,
      "T2": 0.82,
      "T3": 0.12,
      "T4": 0.03
    }
  },
  ...
]
```

### Visualizations

- Circular bounding boxes with color-coded severity
- Predictions overlaid on original images
- Center point and radius/dimensions labeled
- PNG format for easy viewing

## Configuration

### Model Config (`src/config.py`)

```python
IMG_SIZE = (224, 224)           # Input image size
DEVICE = "cuda"                 # "cuda" or "cpu"
DICOM_ROOT = "data/PROSTATEx"   # DICOM data directory
MODELS_DIR = "models"           # Model checkpoint directory
```

### Environment Variables

```bash
# API key protection (optional)
export PROSTATE_API_KEY="your-secret-key"

# Model paths for server startup
export PROSTATE_SIZE_MODEL="models/tumor_size_predictor_best.pth"
export PROSTATE_MODEL_SCRIPTED="models/model_scripted.pt"
export PROSTATE_MODEL_STATE="models/model_state.pth"
```

## Troubleshooting

### Model Not Found
```
Error: failed to load model
Solution: Ensure model_path is correct and file exists
```

### Out of Memory
```
Error: CUDA out of memory
Solutions:
  1. Use --device cpu instead of cuda
  2. Reduce batch size (--bs)
  3. Use smaller image size in config
```

### Poor Predictions
```
Solutions:
  1. Check if using correct image sequences
  2. Verify pixel spacing (DICOM PixelSpacing)
  3. Ensure images are preprocessed correctly
  4. Verify model was trained on similar data
```

## File Structure

```
prostate-severity/
├── src/
│   ├── size_predictor_model.py      # Main model architecture
│   ├── train_size_model.py          # Training script
│   ├── infer_with_bbox.py           # Inference pipeline
│   ├── bbox_utils.py                # Bounding box utilities
│   ├── prostate_dataset.py          # Dataset loading
│   └── utils_image.py               # Image utilities
├── scripts/
│   ├── demo_tumor_size_prediction.py # Comprehensive demo
│   ├── train_unet.py                # Segmentation training
│   └── prepare_seg.py               # Data preparation
├── webapp/
│   ├── fastapi_server.py            # API server
│   └── streamlit_demo.py            # Web interface
├── models/
│   └── tumor_size_predictor_best.pth # Saved model (after training)
└── requirements.txt
```

## Performance Metrics

### Expected Performance (on test set)

- **Size Prediction**: MAE ≈ 2-3 mm
- **Severity Classification**: Accuracy ≈ 85-90%
- **Inference Speed**: ≈ 50-100 ms per image (GPU)

### Factors Affecting Performance

- Quality of training data
- Number of samples
- Data augmentation
- Model architecture (can be customized)
- Input image resolution

## References

- ResNet18: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- Multi-task Learning: Caruana, "Multitask Learning", MLJ 1997
- TNM Staging: Union for International Cancer Control (UICC)

## Citation

If you use this system in your research, please cite:

```bibtex
@software{tumor_size_predictor_2024,
  title={Tumor Size Prediction and Severity Classification System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## License

[Specify your license here]

## Contact

For questions or support, please contact: [your-email@example.com]
