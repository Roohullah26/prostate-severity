# Prostate Tumor Size Analyzer - Deployment Package

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Inference Server
```bash
python -m uvicorn webapp.fastapi_server:app --host 0.0.0.0 --port 8000
```

### 3. Run Interactive UI
```bash
python -m streamlit run webapp/streamlit_app.py
```

## Model
- **File:** `models/baseline_real_t2_adc_3s_ep1.pth`
- **Architecture:** ResNet18-based TumorSizePredictor
- **Input:** 3-channel MRI stack (T2, ADC, DWI) @ 224x224
- **Output:** Tumor size (mm), severity grade, confidence score

## API Endpoints
- `POST /predict` - Single image inference
- `POST /batch` - Batch prediction
- `GET /health` - Health check

## Dataset
Not included in this deployment. Provide your own DICOM/image data.

## Questions?
Refer to the full project README.md in the root directory.
