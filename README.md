# Prostate severity detector (prototype)

This repository contains a small prototype to train a lesion-level binary severity detector (ClinSig) on the ProstateX dataset.

Quickstart (toy/smoke test):

Install (recommended to use a venv) and then run a toy training session that doesn't require DICOM files or large GPU:

```powershell
python -m pip install -r requirements.txt
python -m src.train --toy
```

To run with real data:

1. Ensure the ProstateX DICOM files are downloaded and `src/config.py` DICOM_ROOT points to the top-level dataset directory.
2. Use the provided script to merge labels and metadata:

```powershell
python scripts/00_merge_csvs.py
```

3. Run training (this will attempt to use pretrained weights and GPU when available):

```powershell
python -m src.train --csv merged_data.csv --epochs 5 --bs 16

To run multi-sequence training (e.g. T2 + ADC), pass the sequence keywords that appear in your series descriptions:

```powershell
# try stacking T2 & ADC channels (dataset attempts to find matching series for the same subject)
python -m src.train --csv merged_data.csv --sequences t2,adc --epochs 5 --bs 8

Training tips to improve model quality:

- Stack multiple neighboring slices per lesion using `--num-slices` (must be odd). Example: `--num-slices 3` uses the center slice plus one above and one below.
- Enable data augmentation during training with `--augment` to apply flips and brightness jitter.
- Use class balancing with `--balance weighted` (uses a WeightedRandomSampler) or add automatic class weights for loss with `--loss-weight auto`.

Example: combine options:
```powershell
python -m src.train --csv merged_data.csv --sequences t2,adc --num-slices 3 --augment --balance weighted --loss-weight auto --epochs 10 --bs 8 --export
```
```
```

Next steps and improvements are listed in the code comments and the project issues.

Evaluation / inference (toy examples):

```powershell
# Evaluate a saved toy model on a small toy dataset
python -m src.eval --toy --model-path models/prototype_toy.pth

# Run inference on a few toy samples
python -m src.infer --toy --model-path models/prototype_toy.pth

# Evaluate or infer with multi-sequence model
```powershell
# Evaluate model that was trained with sequences: (example)
python -m src.eval --csv merged_data.csv --sequences t2,adc --model-path models/my_multiseq.pth

# Infer using a multi-sequence model on merged csv entries
python -m src.infer --csv merged_data.csv --sequences t2,adc --model-path models/my_multiseq.pth
```

Streamlit demo (quick UI)
-------------------------

There's a small Streamlit demo under `webapp/streamlit_demo.py` that lets you upload an image or point to a local image and call the running inference server.

Start the inference server (recommended using the launcher so PYTHONPATH is set correctly):

```powershell
& 'D:\prostate project\prostate-severity\run_server.ps1' -StatePath '.\models\prototype_toy.pth' -InCh 3 -Port 8000
```

Then run the Streamlit app (from anywhere):

```powershell
& 'D:\prostate project\prostate-severity\scripts\run_streamlit.ps1' -Port 8501 -ServerUrl 'http://127.0.0.1:8000/predict'
```

The UI allows you to check server health, upload an image, and call `/predict`. If your server requires an API key, set `-ApiKey` in the script or set the `DEMO_API_KEY` environment variable before running Streamlit.

---

## Tumor Size Prediction & Severity Classification

For **precise tumor size prediction** with **bounding box detection** and **clinical severity classification**, see the complete guide:

📖 **[TUMOR SIZE PREDICTION COMPLETE GUIDE](TUMOR_SIZE_COMPLETE_GUIDE.md)**

### Quick Start - Tumor Analysis

```powershell
# 1. Run complete end-to-end analysis
python scripts/demo_tumor_complete.py

# 2. Start API server for predictions
python webapp/fastapi_server.py --port 8000

# 3. Test the API
python scripts/test_tumor_api.py

# 4. Train custom size prediction model
python scripts/train_size_model.py --epochs 100
```

### Key Features

- 🎯 Multi-sequence input (T2, ADC, DWI)
- 📏 Precise tumor size prediction (width × height × depth in mm)
- 📦 Circular and rectangular bounding box generation
- ⚕️ Clinical severity classification (T1, T2, T3, T4)
- 🖼️ Annotated visualization with predictions
- 🚀 RESTful API endpoint: `POST /predict-size`

### Example API Usage

```bash
curl -X POST http://localhost:8000/predict-size \
  -F "file=@tumor_image.png" \
  -d "bbox_type=circle" \
  -d "return_image=true"
```

Response includes severity stage, dimensions in mm, confidence score, and optional visualization.

For full documentation, examples, and troubleshooting, see [TUMOR_SIZE_COMPLETE_GUIDE.md](TUMOR_SIZE_COMPLETE_GUIDE.md).

```
