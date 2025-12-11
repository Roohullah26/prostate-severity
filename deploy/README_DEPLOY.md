Minimal deployment folder for Prostate Severity service.

Contents:
- models/: trained model files (place `baseline_real_t2_adc_3s_ep1.pth` here)
- webapp/: minimal webapp entrypoint (FastAPI)
- src/: optional copy of source code if you want a standalone package

Quick start (local):

1. Activate the project's venv:

   PowerShell:

       & "venv/Scripts/Activate.ps1"

2. Install requirements (if not installed):

       python -m pip install -r requirements.txt

3. Run the API:

       python deploy/webapp/run_uvicorn.py

4. Health check:

   Visit http://localhost:8000/health

Docker build:

    docker build -t prostate-severity:latest deploy/
    docker run -p 8000:8000 prostate-severity:latest

Notes:
- This folder is intentionally minimal. For a fully standalone deployable image, copy `src/` into `deploy/src` and ensure `models/` contains the model.
- The `prepare_deploy.py` script automates creating this structure and copying the model file.
