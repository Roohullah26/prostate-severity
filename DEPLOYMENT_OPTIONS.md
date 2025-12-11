# Deployment Guide: FastAPI Backend + Streamlit Frontend

## Overview

This guide covers two deployment scenarios:
1. **Local Deployment** (Option 3) - Full functionality with real model
2. **Cloud Deployment** (Option 2) - FastAPI on Railway + Streamlit on Streamlit Cloud

---

## OPTION 3: Local Deployment (Recommended for Testing)

### Prerequisites
- Python 3.8+
- GPU (optional, but recommended for faster predictions)
- FastAPI server running on port 8000
- Streamlit running on port 8501

### Step 1: Start FastAPI Backend

```powershell
cd "D:\prostate project\prostate-severity\deploy_clean"
$env:PROSTATE_MODEL_STATE = "D:/prostate project/prostate-severity/deploy_clean/models/baseline_real_t2_adc_3s_ep1.pth"
python -m uvicorn webapp.fastapi_server:app --host 0.0.0.0 --port 8000 --reload
```

### Step 2: Start Streamlit Frontend

```powershell
cd "D:\prostate project\prostate-severity"
python -m streamlit run webapp/streamlit_app.py --server.port 8501
```

### Step 3: Access the App

- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Features Available Locally
✅ Demo Mode (instant predictions)
✅ Synthetic Data (generated test images)
✅ **Upload Mode with REAL predictions** (requires connected FastAPI)
✅ Full model inference using GPU
✅ No file size limits

---

## OPTION 2: Cloud Deployment

### Architecture
```
GitHub (Code Repository)
    ↓
Streamlit Cloud (Frontend)
    ↓ (API calls)
Railway (FastAPI Backend)
```

### Part A: Deploy FastAPI to Railway

#### Step 1: Create Railway Account
1. Go to https://railway.app
2. Sign up (can use GitHub login)

#### Step 2: Create new project
1. Click **New Project**
2. Click **Deploy from GitHub repo**
3. Select your `Roohullah26/prostate-severity` repository

#### Step 3: Configure Railway
1. Click **Add Service**
2. Select **GitHub**
3. Set these environment variables:
   - **PYTHON_VERSION**: `3.11`
   - **PROSTATE_MODEL_STATE**: `/app/models/baseline_real_t2_adc_3s_ep1.pth`

#### Step 4: Add Procfile
Create a file named `Procfile` in your repo root:
```
web: cd deploy_clean && python -m uvicorn webapp.fastapi_server:app --host 0.0.0.0 --port $PORT
```

Then commit:
```powershell
cd "D:\prostate project\prostate-severity"
git add Procfile
git commit -m "Add Railway Procfile for FastAPI deployment"
git push origin main
```

#### Step 5: Deploy
1. Railway will auto-deploy when you push
2. Once deployed, copy your Railway URL (e.g., `https://your-project.railway.app`)

### Part B: Connect Streamlit to Railway API

#### Step 1: Update Streamlit secrets
1. Go to https://share.streamlit.io
2. Open your app settings
3. Go to **Secrets** tab
4. Add this secret:
```
API_URL = "https://your-railway-url.railway.app"
```

#### Step 2: Your Streamlit app will automatically:
- Detect the API connection
- Show ✅ "Connected to FastAPI backend"
- Use real predictions for uploads

---

## Troubleshooting

### Local Issues

**FastAPI won't start:**
```powershell
# Check if port 8000 is in use
netstat -ano | findstr :8000
# Kill the process if needed
taskkill /PID <PID> /F
```

**Streamlit can't find the model:**
- Ensure `PROSTATE_MODEL_STATE` env var points to correct path
- Check that `models/baseline_real_t2_adc_3s_ep1.pth` exists

**Upload takes too long:**
- Make sure images are < 5MB
- Use PNG format instead of NIFTI

### Cloud Issues

**Railway deployment fails:**
- Check build logs in Railway dashboard
- Ensure `Procfile` is in repo root
- Verify `requirements.txt` has all dependencies

**Streamlit can't reach Railway API:**
- Check Railway URL in Secrets is correct
- Verify Railway is running (check logs)
- Test API manually: `https://your-railway-url.railway.app/health`

**Upload size limits:**
- Railway free tier: files < 100MB
- Streamlit Cloud: files < 200MB

---

## Cost Comparison

| Option | Cost | Speed | Maintenance |
|--------|------|-------|-------------|
| **Local (Option 3)** | Free | ⚡ Fast (GPU) | Manual start |
| **Railway (Option 2)** | $5/month+ | ⚡ Moderate | Auto-scaling |
| **Heroku** | Free tier ended | - | - |

---

## Summary

- **Testing locally?** → Use Option 3
- **Want cloud without manual startup?** → Use Option 2 with Railway
- **Both?** → Keep local running + deploy to Railway for others to access

Choose your path and let me know if you need help! 🚀
