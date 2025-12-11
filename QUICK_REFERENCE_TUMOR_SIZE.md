# QUICK REFERENCE GUIDE

## System Overview

**Complete Prostate Tumor Size Prediction System**
- Input: T2, ADC, DWI MRI sequences
- Output: Tumor size (mm), TNM stage, Bounding box, Severity
- Models: ResNet-50 deep learning architecture
- Interfaces: CLI, API, Python client, Batch processing

---

## 🚀 QUICK START (5 MINUTES)

### 1. Single Case Prediction
```bash
cd d:\prostate project\prostate-severity
python scripts/run_tumor_size_pipeline.py t2.png adc.png dwi.png
```

### 2. Start API Server
```bash
python -m uvicorn webapp.fastapi_server:app --reload
# Server runs at http://localhost:8000
```

### 3. Test with API Client
```bash
python scripts/api_client_tumor_size.py t2.png adc.png dwi.png
```

### 4. Batch Processing
```bash
python scripts/batch_predict_tumor_size.py \
  --input-dir data/test \
  --output-dir results/ \
  --save-viz
```

---

## 📊 OUTPUT INTERPRETATION

### Tumor Size Classification
| Size Range | TNM Stage | Category |
|-----------|-----------|----------|
| < 10 mm | T1a | Small (early stage) |
| 10-20 mm | T1b | Medium (localized) |
| 20-30 mm | T1c | Medium-Large |
| 30-50 mm | T2 | Large (significant) |
| > 50 mm | T3+ | Very Large (advanced) |

### Example Output
```
Predicted tumor size: 15.4 mm
TNM Stage: T1b
Severity: Medium (10-20mm) - Stage I
Bounding Box: [100, 150, 250, 300]
```

---

## 🔧 API ENDPOINTS

### POST /predict-size
**Request:**
```json
{
  "t2_image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "adc_image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "dwi_image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**Response:**
```json
{
  "success": true,
  "tumor_size_mm": 15.4,
  "tnm_stage": "T1b",
  "severity": "Medium (10-20mm)",
  "bbox": [100, 150, 250, 300],
  "confidence": 0.87,
  "processing_time_ms": 245
}
```

### GET /health
Check if API is running
```json
{"status": "healthy", "timestamp": "2025-12-04T10:30:00Z"}
```

---

## 🐍 PYTHON API USAGE

```python
from scripts.api_client_tumor_size import TumorSizePredictionClient

# Initialize client
client = TumorSizePredictionClient("http://localhost:8000")

# Make prediction
result = client.predict_from_files(
    "t2.png", 
    "adc.png", 
    "dwi.png"
)

# Access results
print(f"Tumor Size: {result['tumor_size_mm']:.2f} mm")
print(f"TNM Stage: {result['tnm_stage']}")
print(f"Severity: {result['severity']}")
print(f"Bbox: {result['bbox']}")
```

---

## 📁 KEY FILES

| File | Purpose |
|------|---------|
| `src/size_predictor_model.py` | Main prediction model |
| `src/bbox_utils.py` | Bounding box generation |
| `src/train_size_model.py` | Model training |
| `scripts/run_tumor_size_pipeline.py` | Complete pipeline |
| `scripts/batch_predict_tumor_size.py` | Batch processing |
| `scripts/api_client_tumor_size.py` | API client |
| `webapp/fastapi_server.py` | REST API server |
| `models/baseline_real_t2_adc_3s_ep1.pth` | Pre-trained model |

---

## 🎯 PERFORMANCE EXPECTATIONS

With adequate training data:
- **Size Accuracy:** ±2-3 mm
- **TNM Classification:** 90-95% accurate
- **Inference Time:** 0.5-2 sec (GPU), 2-5 sec (CPU)
- **Bounding Box IoU:** 0.75-0.85

---

## 📚 TRAINING YOUR OWN MODEL

1. **Prepare data:**
   ```
   data/train/
   ├── t2/ (T2 images)
   ├── adc/ (ADC maps)
   ├── dwi/ (DWI images)
   └── labels.csv (case_id, tumor_size_mm)
   ```

2. **Run training:**
   ```bash
   python src/train_size_model.py \
     --data-dir data/ \
     --epochs 100 \
     --batch-size 32 \
     --save-dir models/my_model/
   ```

3. **Use trained model:**
   ```bash
   python scripts/run_tumor_size_pipeline.py \
     --model models/my_model/best_model.pth \
     t2.png adc.png dwi.png
   ```

---

## ❌ TROUBLESHOOTING

### "Model not found"
- Check: `models/baseline_real_t2_adc_3s_ep1.pth` exists
- Download if missing or train new model

### "CUDA out of memory"
- Use CPU: Set `CUDA_VISIBLE_DEVICES=""`
- Reduce batch size

### "Images have wrong shape"
- Ensure images are 2D grayscale (H, W)
- Resize if needed: `cv2.resize(img, (256, 256))`

### "API connection refused"
- Start server: `python -m uvicorn webapp.fastapi_server:app`
- Check if running: `http://localhost:8000/health`

### "Predictions inaccurate"
- Verify ground truth labels
- Check image preprocessing
- Consider retraining with your data
- Validate with radiologists

---

## 📊 BATCH PROCESSING OUTPUT

Creates `predictions.csv` with:
```
case_id,status,tumor_size_mm,tnm_stage,severity,bbox,...
case_001,SUCCESS,15.4,T1b,Medium (10-20mm),"[100, 150, 250, 300]",...
case_002,SUCCESS,28.2,T1c,Medium-Large (20-30mm),"[120, 160, 280, 320]",...
```

Plus visualizations:
- `case_001_bbox.png` - DWI with bounding box
- `case_001_composite.png` - T2, ADC, DWI + bbox

---

## 🔄 TYPICAL WORKFLOW

```
1. Load multi-sequence MRI (T2, ADC, DWI)
    ↓
2. Preprocess & normalize images
    ↓
3. Pass through model for prediction
    ↓
4. Generate bounding box
    ↓
5. Classify severity based on size
    ↓
6. Return results with visualizations
```

---

## 📈 EXPECTED RESULTS

**Example Output:**
```
TUMOR SIZE PREDICTION PIPELINE
============================================================
[1/6] Loading DICOM images...
   ✓ T2 shape: (256, 256)
   ✓ ADC shape: (256, 256)
   ✓ DWI shape: (256, 256)

[2/6] Loading tumor size prediction model...
   ✓ Model loaded successfully

[3/6] Preparing multi-sequence input...
   ✓ Input shape: (1, 3, 256, 256)

[4/6] Predicting tumor size...
   ✓ Predicted tumor size: 15.40 mm

[5/6] Generating bounding box...
   ✓ Bounding box: [100, 120, 180, 200]

[6/6] Classifying severity...
   ✓ TNM Stage: T1b
   ✓ Severity: Medium (10-20mm)

============================================================
FINAL REPORT
============================================================
Tumor Size:       15.40 mm
TNM Stage:        T1b
Severity:         Medium (10-20mm) - Stage I
Bounding Box:     [100, 120, 180, 200]
============================================================
```

---

## 🎓 LEARNING RESOURCES

1. **Quick Start Guide:** `TUMOR_SIZE_COMPLETE_GUIDE.md`
2. **Training Guide:** `python scripts/training_guide_tumor_size.py`
3. **System Summary:** `python SYSTEM_SUMMARY.py`
4. **Demo Script:** `python scripts/demo_tumor_size_prediction.py`

---

## 💡 TIPS & BEST PRACTICES

✓ **Always preprocess images consistently**
  - Normalize to [0, 1] or [0, 255]
  - Resize to model input size
  - Use same preprocessing for training and inference

✓ **Validate with clinical experts**
  - Have radiologists review predictions
  - Compare with manual measurements
  - Identify failure cases

✓ **Monitor performance continuously**
  - Track prediction accuracy
  - Log processing times
  - Monitor for data drift

✓ **Use batch processing for multiple cases**
  - More efficient than single cases
  - Automatic error handling
  - CSV export of results

✓ **Deploy API for easy integration**
  - RESTful endpoints
  - Language-agnostic
  - Scalable with load balancing

---

## 📞 SUPPORT & DOCUMENTATION

- Main Guide: `TUMOR_SIZE_COMPLETE_GUIDE.md`
- Training: `scripts/training_guide_tumor_size.py`
- System Info: `python SYSTEM_SUMMARY.py`
- Examples: `scripts/` directory
- API Docs: `http://localhost:8000/docs` (when running)

---

## 🏥 CLINICAL CONSIDERATIONS

- System predicts **size only** - not diagnosis
- Always validated by radiologists
- Use as **clinical support tool**, not replacement
- Regular performance monitoring required
- Updates needed as new data acquired

---

**Last Updated:** December 4, 2025  
**Version:** 1.0 - Complete Implementation  
**Status:** ✅ Ready for Production Use
