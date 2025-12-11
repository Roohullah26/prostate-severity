#!/usr/bin/env python
"""
Evaluate clinical significance (ClinSig) confusion matrix.
Maps model max tumor dimension > 20 mm -> ClinSig=True, compares with ground-truth ClinSig
from ProstateX-Findings-Test.csv (or merged_data.csv). Saves confusion matrix PNG and CSV.

Usage:
    python scripts/eval_clinsig_confusion.py --csv data/ProstateX-Findings-Test.csv --model models/baseline_real_t2_adc_3s_ep1.pth --limit 200

Notes:
- Script searches for case directories under `data/PROSTATEx/*` matching `ProxID`.
- If images are missing for a case it is skipped.
- Outputs saved to `results/confusion_clinsig.csv` and `results/confusion_clinsig.png`.
"""

import argparse
import csv
import glob
import os
from pathlib import Path
import sys
import numpy as np
import torch
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from size_predictor_model import make_size_predictor, TumorSizePredictor

try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
except Exception:
    confusion_matrix = None


def find_case_dir(proxid):
    # search under data/PROSTATEx for folder name containing proxid
    pattern = f"data/PROSTATEx/**/{proxid}*"
    matches = glob.glob(pattern, recursive=True)
    # prefer exact folder names
    for m in matches:
        if os.path.isdir(m):
            return m
    return matches[0] if matches else None


def find_sequence_files(case_dir):
    # find candidate files for T2/ADC/DWI in a case directory
    t2 = None
    adc = None
    dwi = None

    for root, dirs, files in os.walk(case_dir):
        for f in files:
            name = f.lower()
            if 't2' in name and t2 is None:
                t2 = os.path.join(root, f)
            if 'adc' in name and adc is None:
                adc = os.path.join(root, f)
            if 'dwi' in name and dwi is None:
                dwi = os.path.join(root, f)
        if t2 and adc and dwi:
            break
    return t2, adc, dwi


def load_stack_from_paths(t2_path, adc_path, dwi_path):
    # Minimal loader: try to read image via PIL/OpenCV; if DICOM series directory given, skip
    import cv2
    def read_gray(p):
        if p is None:
            return None
        p = str(p)
        if os.path.isdir(p):
            return None
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        # resize to 224x224 if needed
        img = cv2.resize(img, (224, 224))
        return img

    t2 = read_gray(t2_path)
    adc = read_gray(adc_path)
    dwi = read_gray(dwi_path)
    if t2 is None or adc is None or dwi is None:
        return None
    # Stack as (3,224,224) and normalize to float32
    stack = np.stack([t2, adc, dwi], axis=0).astype(np.float32) / 255.0
    # Convert to torch tensor (1,3,224,224)
    tensor = torch.from_numpy(stack).unsqueeze(0)
    return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='ProstateX-Findings-Test.csv')
    parser.add_argument('--model', type=str, default='models/baseline_real_t2_adc_3s_ep1.pth')
    parser.add_argument('--limit', type=int, default=200, help='Max cases to evaluate (0=all)')
    parser.add_argument('--out-dir', type=str, default='results')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.model} on {device}")
    model = make_size_predictor(pretrained=False, in_channels=3, checkpoint_path=args.model)
    model = model.to(device)
    model.eval()

    # Read CSV
    rows = []
    with open(args.csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if args.limit > 0:
        rows = rows[:args.limit]

    y_true = []
    y_pred = []
    skipped = 0
    processed = 0

    for r in rows:
        proxid = r.get('ProxID') or r.get('ProxID')
        clin = r.get('ClinSig') or r.get('ClinSig')
        if clin is None:
            continue
        clin = str(clin).strip().upper()
        true_label = 1 if clin in ('TRUE','True','true','1','T','Y','YES') else 0

        case_dir = find_case_dir(proxid)
        if case_dir is None:
            skipped += 1
            continue
        t2, adc, dwi = find_sequence_files(case_dir)
        if not (t2 and adc and dwi):
            skipped += 1
            continue

        tensor = load_stack_from_paths(t2, adc, dwi)
        if tensor is None:
            skipped += 1
            continue

        tensor = tensor.to(device)
        with torch.no_grad():
            out = model(tensor)
        size = out['size'][0].cpu().numpy()
        max_dim = float(np.max(size))
        # Predict ClinSig: max_dim > 20 mm -> clinically significant
        pred_label = 1 if max_dim > 20.0 else 0

        y_true.append(true_label)
        y_pred.append(pred_label)
        processed += 1

    print(f"Processed: {processed}, Skipped: {skipped}")

    if len(y_true) == 0:
        print("No labeled cases processed; cannot compute confusion matrix.")
        return 1

    # Compute confusion matrix
    try:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
    except Exception as e:
        print("sklearn or matplotlib not available; install scikit-learn and matplotlib to compute confusion matrix")
        # Save raw CSV
        csv_out = out_dir / 'confusion_raw.csv'
        with open(csv_out, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['y_true','y_pred'])
            writer.writerows(zip(y_true, y_pred))
        print(f"Saved raw predictions to {csv_out}")
        return 1

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['ClinNeg','ClinPos'])
    disp.plot(cmap='Blues')
    plt.title('ClinSig Confusion Matrix (pred: max_dim>20mm)')
    png_out = out_dir / 'confusion_clinsig.png'
    plt.savefig(png_out, bbox_inches='tight', dpi=200)
    csv_out = out_dir / 'confusion_clinsig.csv'
    np.savetxt(csv_out, cm, fmt='%d', delimiter=',')

    print(f"Saved confusion matrix PNG: {png_out}")
    print(f"Saved confusion matrix CSV: {csv_out}")
    # Print summary metrics
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    acc = (np.sum(np.diag(cm)) / np.sum(cm))
    print(f"Accuracy: {acc:.3f}")
    if cm.size == 4:
        print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
