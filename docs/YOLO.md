YOLO Detection Pipeline (prototype)
==================================

This project contains a prototype detection pipeline using Ultralytics YOLO (default: yolov8s) and a simple auto-label heuristic. Use this only as a quick way to try detection and size-based severity; for production you should label boxes manually.

Files/entrypoints added
- scripts/prepare_yolo.py — builds a YOLO-format dataset (images, labels, meta) from merged_data.csv and the DICOM folders. The script writes per-image JSON sidecars with pixel spacing so we can compute physical sizes in mm.
- scripts/train_yolo.py — trains a YOLOv8 model (yolov8s by default) on the prepared dataset using the Ultralytics API.
- src/yolo_infer.py — runs inference with YOLO weights and computes bbox sizes in mm using the JSON sidecar; maps size->severity category.
- webapp/streamlit_demo.py — the Streamlit demo now includes a YOLO detection tab where you can run detection locally and see size/severity overlays.

Quick usage (recommended for a quick smoke run):

1) Install dependencies in your python environment (use the repo requirements):

```powershell
python -m pip install -r requirements.txt
```

2) Create a small YOLO dataset for testing (limit rows to keep this quick):

```powershell
py -3 scripts/prepare_yolo.py --csv merged_data.csv --dicom data/PROSTATEx --out yolo_dataset --split 0.9 --fixed-mm 20 --limit 128
```

3) Run a short training job (yolov8s):

```powershell
py -3 scripts/train_yolo.py --data yolo_dataset --epochs 10 --img 640 --batch 8
```

4) Quick detection example (use trained weights or the default path):

```powershell
py -3 src/yolo_infer.py --weights models/yolov8s_prostate/weights/best.pt --image yolo_dataset/val/images/img_val_1.png --meta yolo_dataset/val/meta/img_val_1.json
```

Notes and limitations
- The prepare script produces heuristic centered boxes; if you have real ground-truth boxes or an annotation tool, please re-run prepare_yolo.py or replace labels with real ones.
- For accurate physical size estimations, the DICOM PixelSpacing must be present and correct; we store pixel spacing in the per-image meta files.

Next steps you might want me to do for you
- add a full manual annotation workflow (convert DICOM slices into an annotation-friendly format), or
- implement training hyperparameter tuning and multi-GPU usage, or
- incorporate model quantization & export for fast inference.
