#!/usr/bin/env python3
"""
Prepare a clean deployment directory `deploy/` with minimal files and instructions.
This script will:
 - create `deploy/` with `models/`, `webapp/`, and `src/` placeholders
 - copy the chosen model filename into `deploy/models` (by file copy)
 - write a minimal `deploy/README_DEPLOY.md` with run instructions
 - create a `deploy/Dockerfile` for containerized deployment

Run:
    python scripts/prepare_deploy.py --model models/baseline_real_t2_adc_3s_ep1.pth

Note: model file will be copied; if large, ensure sufficient disk space.
"""

import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/baseline_real_t2_adc_3s_ep1.pth')
    args = parser.parse_args()

    model_path = ROOT / args.model
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    deploy_dir = ROOT / 'deploy'
    models_dir = deploy_dir / 'models'
    webapp_dir = deploy_dir / 'webapp'
    src_dir = deploy_dir / 'src'

    deploy_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    webapp_dir.mkdir(exist_ok=True)
    src_dir.mkdir(exist_ok=True)

    # Copy model
    dest_model = models_dir / model_path.name
    print(f"Copying model to {dest_model}")
    shutil.copyfile(model_path, dest_model)

    # Create minimal run script
    run_py = webapp_dir / 'run_uvicorn.py'
    run_py.write_text("""from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get('/health')
def health():
    return {'ok': True}

if __name__ == '__main__':
    uvicorn.run('webapp.run_uvicorn:app', host='0.0.0.0', port=8000, reload=False)
""")

    # README
    readme = deploy_dir / 'README_DEPLOY.md'
    readme.write_text(f"""Deploy folder (minimal)

Contents:
 - models/: copy of trained model ({model_path.name})
 - webapp/: minimal webapp entrypoint
 - src/: placeholder for application source (use project's `src/` or copy files here)

To run locally:

1) Activate your venv (the project venv recommended)

On Windows PowerShell:

    & "{ROOT / 'venv' / 'Scripts' / 'Activate.ps1'}"

2) Install requirements (if needed):

    python -m pip install -r requirements.txt

3) Start the API:

    python deploy/webapp/run_uvicorn.py

Or using uvicorn directly:

    & "{ROOT / 'venv' / 'Scripts' / 'python.exe'}" -m uvicorn webapp.run_uvicorn:app --host 0.0.0.0 --port 8000

For containerized deployment, see Dockerfile in this folder.
""")

    # Dockerfile
    dockerfile = deploy_dir / 'Dockerfile'
    dockerfile.write_text("""FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["python", "webapp/run_uvicorn.py"]
""")

    print(f"Deploy folder prepared at: {deploy_dir}")
    print("Next: copy or review `src/` files into deploy/src if you want a standalone package.")

if __name__ == '__main__':
    main()
