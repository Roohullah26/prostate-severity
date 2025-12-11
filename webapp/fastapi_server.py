from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.responses import JSONResponse, FileResponse
import argparse
import uvicorn
from pathlib import Path
import os
import logging
import torch
from typing import Optional
import base64
import io
from PIL import Image
import cv2
import numpy as np

from src.utils_image import pil_to_tensor, get_eval_tensor_transform
from src.train import make_model
from src.size_predictor_model import TumorSizePredictor
from src.bbox_utils import BoundingBoxGenerator, VisualizationHelper, TumorPrediction
from src import config


app = FastAPI(title="Prostate Severity Inference")

# global model cache (optionally preloaded on startup)
GLOBAL_MODEL = None
GLOBAL_DEVICE = 'cpu'
GLOBAL_MODEL_PATH = None
GLOBAL_IN_CH = None

# configure simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fastapi_server')


@app.get("/health")
def health():
    return {"ok": True}


def _check_api_key(x_api_key: str | None = Header(None), authorization: str | None = Header(None)):
    """Dependency to verify API key. Looks for X-API-Key header or Bearer token in Authorization.
    If config.API_KEY is None, auth is disabled and this dependency returns True.
    """
    if config.API_KEY is None:
        return True

    # prefer explicit X-API-Key
    key = x_api_key
    if not key and authorization:
        if authorization.lower().startswith("bearer "):
            key = authorization.split(None, 1)[1]

    if not key or key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True


@app.get('/authorize', response_class=HTMLResponse)
def get_authorize():
    """Simple HTML page which accepts an API key and returns a token (for demo).
    This is intentionally very small and not secure — it's only a demonstration.
    """
    html = """
    <html>
      <body>
        <h2>Authorize (demo)</h2>
        <p>Enter API key to receive the token (the API key itself is returned as token in this demo).</p>
        <form method="post" action="/authorize">
          <label>API Key: <input type="text" name="api_key" /></label>
          <input type="submit" value="Authorize" />
        </form>
      </body>
    </html>
    """
    return html


@app.get('/')
def root_redirect():
    """Redirect the web root to the authorize page for easy browser access."""
    return RedirectResponse(url='/authorize')


@app.post('/authorize')
def post_authorize(api_key: str = Form(...)):
    """Accepts an api_key form parameter and returns a token (demo).
    If the server has no config.API_KEY set, we'll return a message indicating auth is disabled.
    """
    if config.API_KEY is None:
        return {"ok": False, "detail": "Server running with no API key configured — auth disabled."}
    if api_key == config.API_KEY:
        # In a real app you'd return a signed JWT. Here we echo back the API key as a demo token.
        return {"token": api_key, "type": "bearer"}
    else:
        raise HTTPException(status_code=401, detail="invalid api_key")


def _load_model(scripted_path: Optional[Path], state_path: Optional[Path], in_ch: int, device: str):
    device = device if device in ("cpu", "cuda") else "cpu"
    if scripted_path and scripted_path.exists():
        model = torch.jit.load(str(scripted_path), map_location=device)
        return model, device

    if state_path and state_path.exists():
        model = make_model(num_classes=2, pretrained=False, in_channels=in_ch)
        sd = torch.load(str(state_path), map_location=device)
        model.load_state_dict(sd)
        model = model.to(device)
        model.eval()
        return model, device

    raise FileNotFoundError("No valid model path provided")


@app.on_event('startup')
def _startup_load_model():
    """On startup, attempt to preload a model if environment variables are set.
    This helps catch model loading errors early and keeps the server running even
    if model load fails (we log the error but don't stop the process).
    Environment variables:
      PROSTATE_MODEL_SCRIPTED -> path to TorchScript model
      PROSTATE_MODEL_STATE -> path to state_dict
      PROSTATE_MODEL_INCH -> integer in_channels
    """
    global GLOBAL_MODEL, GLOBAL_DEVICE, GLOBAL_MODEL_PATH
    script_path = os.environ.get('PROSTATE_MODEL_SCRIPTED')
    state_path = os.environ.get('PROSTATE_MODEL_STATE')
    in_ch = int(os.environ.get('PROSTATE_MODEL_INCH', '3'))
    device = ('cuda' if torch.cuda.is_available() and config.DEVICE == 'cuda' else 'cpu')

    if not script_path and not state_path:
        logger.info('No model env var set (PROSTATE_MODEL_SCRIPTED/PROSTATE_MODEL_STATE) - skipping preload')
        return

    try:
        sp = Path(script_path) if script_path else None
        st = Path(state_path) if state_path else None
        model, dev = _load_model(sp, st, in_ch, device)
        # store globals
        global GLOBAL_MODEL, GLOBAL_DEVICE, GLOBAL_MODEL_PATH, GLOBAL_IN_CH
        GLOBAL_MODEL = model
        GLOBAL_DEVICE = dev
        GLOBAL_IN_CH = in_ch
        GLOBAL_MODEL_PATH = script_path or state_path
        logger.info(f'Preloaded model from {GLOBAL_MODEL_PATH} on device {GLOBAL_DEVICE}')
    except Exception as e:
        logger.exception('Model preload failed: %s', e)


@app.get('/model_status')
def model_status():
    """Return runtime information about the loaded model and quick diagnostics."""
    from pathlib import Path
    info = {
        'loaded': GLOBAL_MODEL is not None,
        'model_path': GLOBAL_MODEL_PATH,
        'device': GLOBAL_DEVICE,
        'in_channels': GLOBAL_IN_CH,
    }
    try:
        logfp = Path(__file__).resolve().parents[1] / 'train_log.txt'
        if logfp.exists():
            with open(logfp, 'r', encoding='utf-8', errors='ignore') as fh:
                txt = fh.read().strip().splitlines()[-20:]
            info['train_log_tail'] = '\n'.join(txt)
    except Exception:
        pass
    return info


@app.get('/series_status')
def series_status(series_uid: str = None):
    """Return basic information about a DICOM series UID (exists, path, count files).
       If series_uid is not provided, returns a small sample of available series keys.
    """
    try:
        from src.utils_dicom import build_uid_map
        uid_map = build_uid_map(config.DICOM_ROOT)
    except Exception as e:
        return {'error': f'failed to build uid_map: {e}'}

    if series_uid is None:
        keys = list(uid_map.keys())[:20]
        return {'available_series_count': len(uid_map), 'sample_keys': keys}

    path = uid_map.get(series_uid)
    if not path:
        return {'found': False, 'series_uid': series_uid}

    from glob import glob
    files = sorted(glob(os.path.join(path, '*.dcm')))
    count = len(files)
    sample = files[:5]
    zvals = []
    try:
        import pydicom
        for f in files[:20]:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            ipp = getattr(ds, 'ImagePositionPatient', None)
            if ipp and len(ipp) >= 3:
                zvals.append(float(ipp[2]))
    except Exception:
        pass

    return {'found': True, 'series_uid': series_uid, 'path': path, 'dcm_count': count, 'sample_files': sample, 'z_values_sample': zvals[:10]}


@app.get('/quick_eval')
def quick_eval(model_path: str = None, toy: bool = True, toy_len: int = 64, num_slices: int = 1):
    """Quickly evaluate a model on a small toy dataset for demo purposes.
       Returns accuracy/auc/precision/recall/f1 when possible.
    """
    try:
        from src.prostate_dataset import ProstateLesionDataset
        from src.eval import evaluate
        from src.train import make_model
        from src import config
        import torch
        from torch.utils.data import DataLoader

        device = 'cpu'
        if torch.cuda.is_available() and config.DEVICE == 'cuda':
            device = 'cuda'

        # build small dataset
        if toy:
            ds = ProstateLesionDataset(toy=True, toy_len=int(toy_len), num_slices=int(num_slices))
        else:
            return {'error': 'non-toy quick_eval not yet supported'}

        # evaluation transform
        try:
            from src.utils_image import get_eval_tensor_transform
            ds.transform = get_eval_tensor_transform(img_size=config.IMG_SIZE)
        except Exception:
            ds.transform = None

        dl = DataLoader(ds, batch_size=8)

        in_ch = ds.num_channels
        model = make_model(num_classes=2, pretrained=False, in_channels=in_ch)
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            # try to use preloaded global model
            if GLOBAL_MODEL is not None:
                model = GLOBAL_MODEL

        model = model.to(device)
        out = evaluate(model, dl, device=device)
        return out
    except Exception as e:
        return {'error': str(e)}


@app.post("/predict")
async def predict(file: UploadFile = File(...), scripted_path: Optional[str] = None, state_dict: Optional[str] = None, in_channels: Optional[int] = 3, auth: bool = Depends(_check_api_key)):
    # accept an image and run model prediction
    try:
        contents = await file.read()
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image upload: {e}")

    try:
        transform = get_eval_tensor_transform(img_size=config.IMG_SIZE)
        t = transform(img)
    except Exception:
        # fallback to pil_to_tensor
        t = pil_to_tensor(img, img_size=config.IMG_SIZE)

    # load model (prefer scripted if provided)
    scripted_path_p = Path(scripted_path) if scripted_path else None
    state_path_p = Path(state_dict) if state_dict else None
    try:
        # prefer globally cached model when available
        if GLOBAL_MODEL is not None and (scripted_path is None and state_dict is None):
            model = GLOBAL_MODEL
            device = GLOBAL_DEVICE
        else:
            model, device = _load_model(scripted_path_p, state_path_p, int(in_channels), device=('cuda' if torch.cuda.is_available() and config.DEVICE == 'cuda' else 'cpu'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to load model: {e}")

    # If model is TorchScript it will accept tensors on device-compatible dtype; else it's a nn.Module
    try:
        model.eval()
        with torch.no_grad():
            xb = t.unsqueeze(0).to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            pred_class = int((prob > 0.5))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")

    return JSONResponse({"probability": float(prob), "class": pred_class})


@app.post("/predict-size")
async def predict_size(
    file: UploadFile = File(...),
    model_path: Optional[str] = None,
    bbox_type: str = "circle",
    pixel_spacing: Optional[str] = None,
    return_image: bool = False,
    auth: bool = Depends(_check_api_key)
):
    """Predict tumor size with bounding box visualization.
    
    Args:
        file: MRI image upload
        model_path: Path to TumorSizePredictor weights (uses env var if not provided)
        bbox_type: 'circle' or 'rect'
        pixel_spacing: JSON string with [row_spacing, col_spacing] in mm
        return_image: If True, return base64-encoded visualization
        
    Returns:
        JSON with prediction results and optional base64 image
    """
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {e}")
    
    # Load model
    model_path = model_path or os.environ.get('PROSTATE_SIZE_MODEL')
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path required (or set PROSTATE_SIZE_MODEL env var)")
    
    device = "cuda" if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu"
    
    try:
        model = TumorSizePredictor(pretrained=False, in_channels=3)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    
    # Parse pixel spacing if provided
    ps = None
    if pixel_spacing:
        try:
            import json
            ps = tuple(json.loads(pixel_spacing))
        except:
            pass
    
    # Inference
    try:
        # Prepare input
        transform = get_eval_tensor_transform(img_size=config.IMG_SIZE)
        img_tensor = transform(img) if transform else pil_to_tensor(img, img_size=config.IMG_SIZE)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(img_tensor)
        
        # Extract predictions
        size = output['size'][0].cpu().numpy()
        severity_probs = output['severity_probs'][0].cpu().numpy()
        confidence = output['confidence'][0, 0].item()
        
        # Determine severity
        severity_idx = int(np.argmax(severity_probs))
        severity_grades = ['T1', 'T2', 'T3', 'T4']
        severity = severity_grades[severity_idx]
        
        # Create prediction object
        prediction = TumorPrediction(
            width_mm=float(size[0]),
            height_mm=float(size[1]),
            depth_mm=float(size[2]),
            severity=severity,
            severity_logits=severity_probs,
            confidence=confidence,
            image_size=(224, 224),
            pixel_spacing_mm=ps,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    
    # Generate bounding box
    try:
        bbox_gen = BoundingBoxGenerator(pixel_spacing_mm=ps)
        if bbox_type == 'circle':
            bbox = bbox_gen.get_circular_bbox(prediction)
        else:
            bbox = bbox_gen.get_rectangular_bbox(prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bbox generation failed: {e}")
    
    # Build response
    response = {
        'severity': prediction.severity,
        'width_mm': prediction.width_mm,
        'height_mm': prediction.height_mm,
        'depth_mm': prediction.depth_mm,
        'max_dimension_mm': max(prediction.width_mm, prediction.height_mm, prediction.depth_mm),
        'confidence': prediction.confidence,
        'severity_probabilities': {
            'T1': float(severity_probs[0]),
            'T2': float(severity_probs[1]),
            'T3': float(severity_probs[2]),
            'T4': float(severity_probs[3]),
        },
        'bbox': bbox,
    }
    
    # Optionally return visualization
    if return_image:
        try:
            vis_helper = VisualizationHelper()
            if bbox_type == 'circle':
                vis_image = vis_helper.draw_circular_bbox_pil(img, bbox, prediction)
            else:
                vis_image = vis_helper.draw_rectangular_bbox_pil(img, bbox, prediction)
            
            # Convert to base64
            buffer = io.BytesIO()
            vis_image.save(buffer, format="PNG")
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            response['image_base64'] = img_base64
        except Exception as e:
            logger.warning(f"Could not generate visualization: {e}")
    
    return JSONResponse(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scripted-path', default=None)
    parser.add_argument('--state-dict', default=None)
    parser.add_argument('--in-ch', type=int, default=3)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    # If the user passed a model path via CLI, make it available on the process env
    if args.scripted_path:
        os.environ['PROSTATE_MODEL_SCRIPTED'] = str(args.scripted_path)
    if args.state_dict:
        os.environ['PROSTATE_MODEL_STATE'] = str(args.state_dict)

    # Validate model availability before starting
    if not args.scripted_path and not args.state_dict and not os.environ.get('PROSTATE_MODEL_SCRIPTED') and not os.environ.get('PROSTATE_MODEL_STATE'):
        print('Warning: no model specified. /predict will fail until a model path is provided via query params, env var, or server args.')

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
