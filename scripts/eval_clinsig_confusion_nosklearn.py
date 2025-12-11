#!/usr/bin/env python3
"""
Lightweight ClinSig confusion evaluator without sklearn to avoid heavy deps.
Saves CSV of predictions and a simple summary CSV with TN/FP/FN/TP.
"""
import argparse
import csv
import glob
import os
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

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


def read_gray_resize(path):
    import cv2
    if path is None:
        return None
    p = str(path)
    if os.path.isdir(p):
        return None
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (224,224))
    return img


def load_stack(t2,adc,dwi):
    t2i = read_gray_resize(t2)
    adci = read_gray_resize(adc)
    dwii = read_gray_resize(dwi)
    if t2i is None or adci is None or dwii is None:
        return None
    stack = np.stack([t2i, adci, dwii], axis=0).astype(np.float32)/255.0
    tensor = torch.from_numpy(stack).unsqueeze(0)
    return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='ProstateX-Findings-Test.csv')
    parser.add_argument('--model', type=str, default='models/baseline_real_t2_adc_3s_ep1.pth')
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--out', type=str, default='results')
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Creating model')
    model = make_size_predictor(pretrained=False, in_channels=3, checkpoint_path=None)
    model = model.to(device)

    # Robust checkpoint loading: try multiple strategies to accommodate different saved formats
    if args.model:
        print('Loading checkpoint from', args.model)
        ckpt = torch.load(args.model, map_location='cpu')
        # if ckpt is a dict with nested state, extract
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt

        loaded = False
        # 1) try direct load into full model
        try:
            model.load_state_dict(sd)
            loaded = True
            print('Loaded checkpoint directly into model')
        except Exception:
            pass

        # 2) try loading into backbone (backbone-only checkpoint)
        if not loaded:
            try:
                model.backbone.load_state_dict(sd, strict=False)
                loaded = True
                print('Loaded checkpoint into model.backbone (strict=False)')
            except Exception:
                pass

        # 3) try prefixing keys with 'backbone.' then load
        if not loaded and isinstance(sd, dict):
            try:
                mapped = {'backbone.' + k: v for k, v in sd.items()}
                model.load_state_dict(mapped, strict=False)
                loaded = True
                print("Loaded checkpoint by prefixing keys with 'backbone.' (strict=False)")
            except Exception:
                pass

        if not loaded:
            print('Warning: failed to fully load checkpoint into model; proceeding with partial weights')

    model.eval()

    rows = []
    with open(args.csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if args.limit>0:
        rows = rows[:args.limit]

    y_true = []
    y_pred = []
    raw = []
    skipped = 0

    for r in rows:
        proxid = r.get('ProxID') or ''
        clin = r.get('ClinSig') or r.get('ClinSig')
        if clin is None:
            skipped += 1
            continue
        true_label = 1 if str(clin).strip().lower() in ('true','1','t','y','yes') else 0
        case_dir = find_case_dir(proxid)
        if not case_dir:
            skipped += 1
            continue
        t2,adc,dwi = find_sequence_files(case_dir)
        if not (t2 and adc and dwi):
            skipped += 1
            continue
        tensor = load_stack(t2,adc,dwi)
        if tensor is None:
            skipped += 1
            continue
        tensor = tensor.to(device)
        with torch.no_grad():
            out = model(tensor)
        size = out['size'][0].cpu().numpy()
        max_dim = float(np.max(size))
        pred_label = 1 if max_dim>20.0 else 0
        y_true.append(true_label)
        y_pred.append(pred_label)
        raw.append({'proxid':proxid,'true':true_label,'pred':pred_label,'max_dim':max_dim})

    # compute counts
    tn = fp = fn = tp = 0
    for t,p in zip(y_true,y_pred):
        if t==0 and p==0:
            tn+=1
        elif t==0 and p==1:
            fp+=1
        elif t==1 and p==0:
            fn+=1
        elif t==1 and p==1:
            tp+=1
    total = tn+fp+fn+tp
    acc = (tn+tp)/total if total>0 else 0.0

    # save raw
    raw_csv = outdir / 'confusion_clinsig_raw.csv'
    with open(raw_csv,'w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f,fieldnames=['proxid','true','pred','max_dim'])
        w.writeheader()
        w.writerows(raw)

    summary_csv = outdir / 'confusion_clinsig_summary.csv'
    with open(summary_csv,'w',newline='',encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['TN','FP','FN','TP','Total','Accuracy'])
        w.writerow([tn,fp,fn,tp,total,f'{acc:.4f}'])

    print(f'Processed: {total}, Skipped: {len(rows)-total}')
    print(f'TN={tn}, FP={fp}, FN={fn}, TP={tp}, Acc={acc:.4f}')
    print('Raw predictions saved to', raw_csv)
    print('Summary saved to', summary_csv)
    return 0

if __name__=='__main__':
    sys.exit(main())
