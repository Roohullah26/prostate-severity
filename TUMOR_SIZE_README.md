## Tumor Size Prediction & Severity Classification (NEW FEATURE)

**Now includes a complete system for predicting tumor dimensions and severity classification!**

### 🎯 Quick Start (No Training Required)

Try the comprehensive demo with an untrained model:

```powershell
python scripts/demo_tumor_size_prediction.py --toy --output demo_results/
```

This will demonstrate all features without requiring a pretrained model or DICOM data.

### ✨ Key Capabilities

- 📏 **3D Tumor Dimension Prediction**: Predicts width, height, depth in millimeters
- 🎯 **Severity Classification**: T1 (≤20mm) / T2 (20-40mm) / T3 (40-60mm) / T4 (>60mm)
- 🔲 **Bounding Box Generation**: Circular or rectangular boxes with size labels
- 📊 **Confidence Scoring**: Model prediction confidence for each sample
- 🖼️ **Visualization**: Color-coded by severity (🟢Green/🟡Yellow/🟠Orange/🔴Red)

### 🏗️ System Architecture

```
Input: T2/ADC/DWI Multi-sequence MRI (3×224×224)
           ↓
    ResNet18 Backbone (ImageNet-pretrained)
           ↓
      ┌────┬────┬────┐
      ↓    ↓    ↓    ↓
    Size Severity Confidence
     Head Head    Head
      ↓    ↓      ↓
  [W,H,D][T1-T4][0-1]
     mm   logits  conf
```

**Components:**
- **Backbone**: ResNet18 (pretrained on ImageNet)
- **Input Channels**: 3 (stacked T2, ADC, DWI sequences)
- **Input Size**: 224×224 pixels
- **Loss**: Combined MSE (size) + CrossEntropy (severity) + BCE (confidence)

### 🌐 REST API

#### `/predict-size` Endpoint

```powershell
# Predict tumor size and severity from MRI image
curl -X POST http://localhost:8000/predict-size `
  -F "file=@mri_image.png" `
  -F "model_path=models/tumor_size_predictor_best.pth" `
  -F "bbox_type=circle" `
  -F "return_image=true"
```

**Request Parameters:**
- `file` (required): MRI image file (PNG, JPG, etc.)
- `model_path`: Path to model weights (or set `PROSTATE_SIZE_MODEL` env var)
- `bbox_type`: `"circle"` or `"rect"` (default: circle)
- `pixel_spacing`: JSON `"[row_mm, col_mm]"` for DICOM pixel spacing
- `return_image`: Return base64-encoded visualization (default: false)

**Response Example:**
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

### 🐍 Python Usage

#### Basic Inference

```python
from src.infer_with_bbox import TumorInferencePipeline
from PIL import Image

# Initialize pipeline
pipeline = TumorInferencePipeline(
    model_path="models/tumor_size_predictor_best.pth",
    device="cuda"
)

# Load and predict
image = Image.open("mri.png").resize((224, 224))
result = pipeline.predict_with_visualization(image, bbox_type="circle")

# Extract results
pred = result['prediction']
print(f"Severity: {pred.severity}")
print(f"Dimensions: {pred.width_mm:.1f}×{pred.height_mm:.1f}×{pred.depth_mm:.1f} mm")
print(f"Confidence: {pred.confidence:.3f}")

# Save visualization
result['image'].save("prediction.png")
```

#### Batch Processing

```python
from src.infer_with_bbox import TumorInferencePipeline
from PIL import Image
import glob

pipeline = TumorInferencePipeline("models/tumor_size_predictor_best.pth")

# Load images
images = [Image.open(f).resize((224, 224)) for f in glob.glob("mri_data/*.png")]

# Batch predict
results = pipeline.predict_batch(images, bbox_type="circle")

# Process results
for i, result in enumerate(results):
    pred = result['prediction']
    print(f"Image {i}: {pred.severity} ({pred.confidence:.3f})")
```

#### Custom Bounding Boxes

```python
from src.bbox_utils import BoundingBoxGenerator, VisualizationHelper, TumorPrediction
import numpy as np

# Create prediction
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
circ_bbox = bbox_gen.get_circular_bbox(pred)
rect_bbox = bbox_gen.get_rectangular_bbox(pred)

# Visualize
vis_helper = VisualizationHelper()
img = Image.open("mri.png")

vis_circ = vis_helper.draw_circular_bbox_pil(img, circ_bbox, pred)
vis_circ.save("prediction_circle.png")

vis_rect = vis_helper.draw_rectangular_bbox_pil(img, rect_bbox, pred)
vis_rect.save("prediction_rect.png")
```

### 🎓 Training

#### Toy Training (Quick Test)

```powershell
python -m src.train_size_model --toy --epochs 5 --bs 16
```

#### Real Data Training

```powershell
python -m src.train_size_model `
  --csv merged_data.csv `
  --size-csv tumor_sizes.csv `
  --sequences t2,adc,dwi `
  --epochs 20 `
  --bs 8 `
  --lr 0.001
```

**Expected Output**: `models/tumor_size_predictor_best.pth`

**Training Parameters:**
- `--csv`: Path to merged_data.csv
- `--size-csv`: CSV with tumor dimension annotations
- `--sequences`: Sequences to use (t2,adc,dwi)
- `--epochs`: Number of training epochs
- `--bs`: Batch size
- `--lr`: Learning rate (default: 0.001)
- `--wd`: Weight decay
- `--toy`: Use toy dataset for quick testing

### 🔍 Inference

#### Command Line Inference

```powershell
# Batch inference on dataset
python -m src.infer_with_bbox `
  --model-path models/tumor_size_predictor_best.pth `
  --csv merged_data.csv `
  --sequences t2,adc,dwi `
  --output results/ `
  --max-samples 100

# Toy inference (no model training needed)
python -m src.infer_with_bbox --toy --output results/
```

#### Comprehensive Demo

```powershell
# Run full feature demo
python scripts/demo_tumor_size_prediction.py `
  --model-path models/tumor_size_predictor_best.pth `
  --csv merged_data.csv `
  --output demo_results/
```

This demo includes:
- Model architecture overview
- Severity classification scheme
- Bounding box generation examples
- Visualization capabilities
- Inference on toy or real data
- Summary statistics

### 📊 Severity Classification Reference

| Grade | Range | Color | Clinical Significance |
|-------|-------|-------|----------------------|
| **T1** | ≤ 20 mm | 🟢 Green | Small, clinically insignificant |
| **T2** | 20-40 mm | 🟡 Yellow | Medium, locally confined |
| **T3** | 40-60 mm | 🟠 Orange | Large, may extend beyond organ |
| **T4** | > 60 mm | 🔴 Red | Very large, extensive local invasion |

### 📁 Output Files

**Predictions JSON:**
```json
[
  {
    "sample_id": 0,
    "width_mm": 18.5,
    "height_mm": 22.3,
    "depth_mm": 20.1,
    "severity": "T2",
    "confidence": 0.94,
    "severity_probs": {"T1": 0.03, "T2": 0.82, "T3": 0.12, "T4": 0.03}
  },
  ...
]
```

**Visualizations:**
- PNG images with color-coded bounding boxes
- Severity-based coloring for quick assessment
- Dimensions and confidence labeled

### 🔧 Configuration

**Model Config** (`src/config.py`):
```python
IMG_SIZE = (224, 224)              # Input image size
DEVICE = "cuda"                    # "cuda" or "cpu"
```

**Environment Variables:**
```bash
export PROSTATE_SIZE_MODEL="models/tumor_size_predictor_best.pth"
export PROSTATE_API_KEY="your-api-key"
```

### 📚 Related Files

- **Main Demo**: `scripts/demo_tumor_size_prediction.py`
- **Model**: `src/size_predictor_model.py`
- **Training**: `src/train_size_model.py`
- **Inference**: `src/infer_with_bbox.py`
- **Utilities**: `src/bbox_utils.py`
- **API**: `webapp/fastapi_server.py` (endpoint: `/predict-size`)
- **Documentation**: `TUMOR_SIZE_PREDICTION.md`

For detailed documentation, see: **[TUMOR_SIZE_PREDICTION.md](TUMOR_SIZE_PREDICTION.md)**
