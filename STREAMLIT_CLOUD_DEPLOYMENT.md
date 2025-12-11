# Streamlit Cloud Deployment Guide

## Deploy to Streamlit Cloud

### Step 1: Create GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Create repository name: `prostate-severity`
3. Add description: "Interactive Prostate Tumor Size Prediction & Analysis with FastAPI & Streamlit"
4. Choose **Public** (required for free Streamlit Cloud)
5. Click **Create Repository**

### Step 2: Initialize Git & Push Code

Run these commands in your project directory:

```bash
cd "D:\prostate project\prostate-severity"
git init
git add .
git commit -m "Initial commit: Prostate severity analysis system"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/prostate-severity.git
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Click **New App**
3. Authenticate with GitHub
4. Select repository: `prostate-severity`
5. Select branch: `main`
6. Set file path: `webapp/streamlit_app.py`
7. Click **Deploy**

### Step 4: (Optional) Add FastAPI Backend

For full functionality with image uploads, deploy FastAPI separately:

**Option A: Heroku/Railway (Free)**
```bash
cd deploy_clean
git init
git add .
git commit -m "FastAPI backend"
```

**Option B: Use environment variables in Streamlit to point to remote API**

Edit `webapp/streamlit_app.py` and add:
```python
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")
```

Then in Streamlit Cloud settings, add secret:
- **Key**: `API_URL`
- **Value**: Your deployed FastAPI URL

## Project Structure for Streamlit Cloud

```
prostate-severity/
├── webapp/
│   ├── streamlit_app.py       # Main Streamlit app
│   └── fastapi_server.py       # Optional: FastAPI
├── src/
│   ├── size_predictor_model.py
│   ├── visualization_enhanced.py
│   └── bbox_utils.py
├── models/
│   └── baseline_real_t2_adc_3s_ep1.pth  # Trained model
├── .streamlit/
│   └── config.toml             # Streamlit config
├── .gitignore                  # Git ignore rules
└── requirements.txt            # Python dependencies
```

## Important Notes

✅ **What's Included:**
- Streamlit UI with NAVIGATOR sidebar
- Severity color legend
- Demo Mode (no image upload needed)
- Synthetic data generation
- Beautiful gradient UI

⚠️ **Limitations on Streamlit Cloud:**
- File upload limited to ~200MB
- No GPU (demo & synthetic modes work fine)
- Image upload mode requires connected FastAPI backend

## Troubleshooting

**App won't load?**
- Check `requirements.txt` has all dependencies
- Verify model file path is relative: `models/baseline_real_t2_adc_3s_ep1.pth`
- Check console for errors at Streamlit Cloud logs

**Demo Mode shows errors?**
- This is the safest mode; doesn't require external files
- Check internet connection

**Need help?**
- Documentation: [streamlit.io/docs](https://streamlit.io/docs)
- Community: [discuss.streamlit.io](https://discuss.streamlit.io)
