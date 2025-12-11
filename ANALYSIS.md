# Prostate Severity Detection - Current State & Enhancement Plan

## Current System Overview

### What's Working ✅
1. **Binary Severity Classification**: ResNet18 model trained on ProstateX dataset for ClinSig (clinical significance) prediction
2. **Multi-sequence Support**: Can stack T2, ADC sequences during training
3. **DICOM Processing**: Utilities for loading and extracting lesions from DICOM series
4. **YOLO Detection**: YOLOv8 setup for basic lesion detection with approximate bounding boxes
5. **FastAPI Server**: Inference server for REST API calls
6. **Streamlit UI**: Demo interface for interactive testing

### Current Data Flow
```
DICOM Series (T2, ADC, DWI, etc.)
    ↓
Extract lesion patches (224x224)
    ↓
Binary classifier (ClinSig: 0=benign, 1=malignant)
    ↓
Inference output: Probability score
```

### Gaps & Limitations ❌
1. **No Tumor Size Prediction**: Model only outputs binary classification, doesn't measure tumor dimensions
2. **No Multi-dimensional Output**: Can't predict width, height, depth separately
3. **Approximate Bounding Boxes**: YOLO bboxes are synthetically generated (~20mm fixed size), not learned from real tumor shapes
4. **No Circular/Rounded Boxes**: Uses standard rectangular YOLO boxes
5. **No Severity Grading**: No T2/T3/T4 classification based on actual tumor size
6. **Single Sequence Inference**: Current inference mostly uses single modality, not leveraging multi-sequence power

---

## Required Enhancements

### 1. Multi-Sequence Tumor Size Predictor
**Goal**: Predict precise tumor dimensions using T2, ADC, and DWI sequences

**Approach**:
- Input: 3-channel stacked image (T2 + ADC + DWI)
- Architecture: ResNet18 → Multi-output head
  - Output 1: `tumor_width_mm` (float)
  - Output 2: `tumor_height_mm` (float)  
  - Output 3: `tumor_depth_mm` (optional, from multi-slice stack)
- Loss: MSE or L1 loss for regression
- Training data: DICOM series with manual/reference tumor measurements

### 2. Circular Bounding Box Detection
**Goal**: Detect and draw rounded/circular boxes around tumors

**Approach**:
- Regression outputs: center_x, center_y, radius_mm
- Convert predicted dimensions to circular representation
- Visualization: cv2.circle() for round boxes
- Advantage: Better approximation of tumor morphology (more tumors are roughly spherical)

### 3. Severity Grading (TNM-like Classification)
**Goal**: Map tumor size to severity/staging

**Rules** (can be adjusted based on clinical guidelines):
```
T1 (Small):     diameter ≤ 20mm  (confined to prostate)
T2 (Medium):    20-40mm (confined but larger)
T3 (Large):     40-60mm (extends beyond prostate)
T4 (Very Large): >60mm (invades adjacent structures)
```

### 4. Multi-Sequence Integration
**Enhancement**: Proper integration of T2, ADC, DWI
- T2: Shows anatomical detail (high intensity in fluid)
- ADC: Shows restricted diffusion (low = more aggressive)
- DWI: Shows diffusion restriction (high intensity = abnormal)
- Combined: Stacked or concatenated for better discrimination

---

## Implementation Plan

### Phase 1: Tumor Size Regression Model
Create new model architecture for size prediction:

```python
# Model output structure:
predictions = {
    'tumor_width_mm': 15.5,
    'tumor_height_mm': 18.2,
    'tumor_depth_mm': 14.1,  # from multi-slice
    'severity': 'T2',
    'confidence': 0.92
}
```

### Phase 2: Bounding Box Generation
Convert size predictions to visual bounding boxes:
- Compute bounding circle/ellipse from predicted dimensions
- Draw on original image
- Output: Image with overlay + metadata

### Phase 3: End-to-End Pipeline
Create unified inference that:
1. Takes DICOM series (T2, ADC, DWI) as input
2. Extracts lesion region
3. Predicts size dimensions
4. Generates rounded bounding box
5. Classifies severity
6. Returns visualized image + structured data

### Phase 4: Integration with Existing System
- Add to FastAPI server as new `/predict-size` endpoint
- Update Streamlit UI to show:
  - Predicted dimensions
  - Visualized bounding box
  - Severity grade
  - Confidence scores

---

## Data Requirements

### Training Data
- **Lesion patches**: 224×224 from T2, ADC, DWI sequences
- **Annotations needed**:
  - Ground truth tumor width (mm)
  - Ground truth tumor height (mm)
  - Ground truth tumor depth (mm)
  - DICOM PixelSpacing for mm conversion

### Current Dataset: ProstateX
- **Samples**: 10,000+ lesions
- **Available sequences**: T2, ADC, DWI
- **Current annotations**: ClinSig (0/1 only), biopsy location, zone

### Creating Ground Truth
Options:
1. **Manual annotation**: Clinician draws bounding box per lesion
2. **Semi-automatic**: ML-assisted (mask R-CNN) then manual refinement
3. **Synthetic augmentation**: Scale existing boxes by clinical knowledge

---

## Technical Implementation Details

### Key Files to Create/Modify

1. **`src/train_size_model.py`** - New training script for size regression
2. **`src/size_predictor_model.py`** - Model definition with multi-output head
3. **`src/infer_with_bbox.py`** - Inference with bounding box generation
4. **`src/severity_classifier.py`** - Logic for T-stage classification
5. **`webapp/fastapi_server.py`** - Add `/predict-size` endpoint
6. **`webapp/streamlit_demo.py`** - Update UI for visualization

### Model Architecture (Recommended)
```
Input: (3, 224, 224)  # T2 + ADC + DWI stacked
  ↓
ResNet18 backbone (pretrained on ImageNet)
  ↓
Shared features: (512, 7, 7)
  ↓
Head 1 (Size): FC layers → [width_mm, height_mm, depth_mm]
Head 2 (Severity): FC layers → [t1_score, t2_score, t3_score, t4_score]
Head 3 (Confidence): FC layer → [confidence]
```

### Loss Function
```
Total Loss = α * MSE(size_pred, size_gt) + β * CrossEntropy(severity_logits, severity_gt)
```

---

## Performance Metrics

1. **Size Prediction**:
   - MAE (Mean Absolute Error) in mm for width, height, depth
   - RMSE (Root Mean Squared Error)
   - Correlation with ground truth

2. **Bounding Box Quality**:
   - IoU (Intersection over Union) between predicted and ground truth boxes
   - Target: IoU > 0.7

3. **Severity Classification**:
   - Accuracy for T1/T2/T3/T4 classification
   - Confusion matrix
   - Target: Accuracy > 85%

---

## Next Steps

1. ✅ Create analysis document (this file)
2. 🔲 Create training script for size regression model
3. 🔲 Build severity classifier logic
4. 🔲 Implement bounding box visualization
5. 🔲 Integrate into inference pipeline
6. 🔲 Update API and UI
7. 🔲 Create evaluation metrics script
8. 🔲 Generate test results

---

## Questions to Answer

1. **Ground Truth Data**: Do you have manual tumor size annotations?
2. **Severity Criteria**: What size ranges define T2, ADC severity in your clinic?
3. **Model Priority**: Focus on accuracy vs. inference speed?
4. **Multi-slice vs Single**: How many slices should we stack for depth estimation?
5. **Clinical Integration**: Will this feed into a larger PACS/EMR system?
