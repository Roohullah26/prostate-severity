#!/usr/bin/env python
"""
Fast ClinSig confusion evaluator: loads model, processes images, computes confusion metrics,
saves PNG + CSV.
"""
import argparse
import csv
import glob
import os
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from size_predictor_model import make_size_predictor

def find_case_dir(proxid):
    pattern = f"data/PROSTATEx/**/{proxid}*"
    matches = glob.glob(pattern, recursive=True)
    for m in matches:
        if os.path.isdir(m):
            return m
    return matches[0] if matches else None

def find_sequence_files(case_dir):
    t2 = adc = dwi = None
    for root, dirs, files in os.walk(case_dir):
        for f in files:
            fname = f.lower()
            if 't2' in fname and t2 is None:
                t2 = os.path.join(root, f)
            if 'adc' in fname and adc is None:
                adc = os.path.join(root, f)
            if 'dwi' in fname and dwi is None:
                dwi = os.path.join(root, f)
        if t2 and adc and dwi:
            break
    return t2, adc, dwi

def load_images(t2, adc, dwi):
    import cv2
    images = []
    for path in [t2, adc, dwi]:
        if path is None or not os.path.exists(path):
            return None
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        images.append(img)
    # Stack twice to create 6 channels (3x2 for each sequence)
    stack = np.concatenate([np.stack(images, axis=0)] * 2, axis=0)
    return torch.from_numpy(stack).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='ProstateX-Findings-Test.csv')
    parser.add_argument('--model', type=str, default='models/baseline_real_t2_adc_3s_ep1.pth')
    parser.add_argument('--limit', type=int, default=150)
    parser.add_argument('--out', type=str, default='results')
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Loading model from {args.model}')
    
    model = make_size_predictor(pretrained=False, in_channels=6, checkpoint_path=None)
    model = model.to(device)
    
    # Load checkpoint with fallback strategies
    ckpt = torch.load(args.model, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt
    
    try:
        model.load_state_dict(sd)
        print('[OK] Checkpoint loaded into full model')
    except:
        try:
            model.backbone.load_state_dict(sd, strict=False)
            print('[OK] Checkpoint loaded into model.backbone')
        except:
            try:
                mapped = {'backbone.' + k: v for k, v in sd.items()}
                model.load_state_dict(mapped, strict=False)
                print('[OK] Checkpoint loaded with prefix mapping')
            except:
                print('[WARN] Partial or no checkpoint loaded')
    
    model.eval()

    # Read CSV
    rows = []
    with open(args.csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    
    if args.limit > 0:
        rows = rows[:args.limit]
    
    print(f'Processing {len(rows)} rows...')
    
    y_true = []
    y_pred = []
    raw_data = []
    skipped = 0
    
    for i, row in enumerate(rows):
        if (i + 1) % 20 == 0:
            print(f'  [{i+1}/{len(rows)}]')
        
        proxid = row.get('ProxID', '').strip()
        clin_str = row.get('ClinSig', '').strip().upper()
        
        if not proxid or not clin_str:
            skipped += 1
            continue
        
        true_label = 1 if clin_str in ('TRUE', '1', 'T', 'Y', 'YES') else 0
        
        case_dir = find_case_dir(proxid)
        if not case_dir:
            skipped += 1
            continue
        
        t2, adc, dwi = find_sequence_files(case_dir)
        if not (t2 and adc and dwi):
            skipped += 1
            continue
        
        tensor = load_images(t2, adc, dwi)
        if tensor is None:
            skipped += 1
            continue
        
        tensor = tensor.to(device)
        with torch.no_grad():
            out = model(tensor)
        
        size = out['size'][0].cpu().numpy()
        max_dim = float(np.max(size))
        pred_label = 1 if max_dim > 20.0 else 0
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        raw_data.append({
            'ProxID': proxid,
            'ground_truth': true_label,
            'prediction': pred_label,
            'max_size_mm': f'{max_dim:.2f}'
        })
    
    total = len(y_true)
    print(f'\nProcessed: {total}, Skipped: {skipped}')
    
    if total == 0:
        print('Error: no samples processed')
        return 1
    
    # Compute confusion matrix
    tn = fp = fn = tp = 0
    for t, p in zip(y_true, y_pred):
        if t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tp += 1
    
    acc = (tp + tn) / total if total > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Save raw predictions
    raw_csv = outdir / 'confusion_clinsig_raw.csv'
    with open(raw_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['ProxID', 'ground_truth', 'prediction', 'max_size_mm'])
        w.writeheader()
        w.writerows(raw_data)
    
    # Save summary metrics
    summary_csv = outdir / 'confusion_clinsig_summary.csv'
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Metric', 'Value'])
        w.writerow(['TN', tn])
        w.writerow(['FP', fp])
        w.writerow(['FN', fn])
        w.writerow(['TP', tp])
        w.writerow(['Total', total])
        w.writerow(['Accuracy', f'{acc:.4f}'])
        w.writerow(['Sensitivity (Recall)', f'{sensitivity:.4f}'])
        w.writerow(['Specificity', f'{specificity:.4f}'])
        w.writerow(['Precision', f'{precision:.4f}'])
    
    # Plot confusion matrix
    try:
        from sklearn.metrics import confusion_matrix as sk_cm, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
        
        cm = sk_cm(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not ClinSig', 'ClinSig'])
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap='Blues')
        plt.title(f'Clinical Significance Confusion Matrix (n={total})\nAccuracy={acc:.3f}')
        png_path = outdir / 'confusion_clinsig.png'
        plt.savefig(png_path, dpi=100, bbox_inches='tight')
        print(f'[OK] Confusion matrix PNG saved: {png_path}')
    except Exception as e:
        print(f'[WARN] PNG plotting failed: {e}')
    
    print(f'\n=== CONFUSION MATRIX ===')
    print(f'TN (true negatives):  {tn}')
    print(f'FP (false positives): {fp}')
    print(f'FN (false negatives): {fn}')
    print(f'TP (true positives):  {tp}')
    print(f'\n=== METRICS ===')
    print(f'Accuracy:    {acc:.4f}')
    print(f'Sensitivity: {sensitivity:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'Precision:   {precision:.4f}')
    print(f'\n[OK] Raw predictions: {raw_csv}')
    print(f'[OK] Summary metrics: {summary_csv}')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
