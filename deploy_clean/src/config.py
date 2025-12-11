from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# point to local repository data folder (expected to contain PROSTATEx/ subfolder)
DICOM_ROOT = Path(__file__).resolve().parents[1] / "data" / "PROSTATEx"
STACKED_OUT = ROOT / "stacked_multislice"
YOLO_DATA = ROOT / "yolo_dataset"
MODELS_DIR = ROOT / "models"
IMG_SIZE = (224, 224)
DEVICE = "cuda"
# Optional simple API key for protecting the inference server. Default None -> auth disabled
import os
API_KEY = os.environ.get("PROSTATE_API_KEY", None)
