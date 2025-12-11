import argparse
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score

from src.prostate_dataset import ProstateLesionDataset
from src.utils_image import pil_to_tensor
from src import config


def evaluate(model, dataloader, device="cpu"):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds.extend(prob.tolist())
            trues.extend(yb.numpy().tolist())

    preds = np.array(preds)
    trues = np.array(trues)

    acc = accuracy_score(trues, (preds > 0.5).astype(int))
    auc = roc_auc_score(trues, preds) if len(np.unique(trues)) > 1 else float("nan")
    cm = confusion_matrix(trues, (preds > 0.5).astype(int))
    prec = precision_score(trues, (preds > 0.5).astype(int), zero_division=0)
    rec = recall_score(trues, (preds > 0.5).astype(int), zero_division=0)
    f1 = f1_score(trues, (preds > 0.5).astype(int), zero_division=0)
    return {"accuracy": acc, "auc": auc, "confusion_matrix": cm.tolist(), "precision": float(prec), "recall": float(rec), "f1": float(f1)}


def run(args):
    device = "cuda" if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu"

    if args.toy:
        ds = ProstateLesionDataset(toy=True, toy_len=args.toy_len, num_slices=args.num_slices)
    else:
        seqs = None if args.sequences is None else [s.strip().lower() for s in args.sequences.split(",") if s.strip()]
        ds = ProstateLesionDataset(csv_path=args.csv, sequences=seqs, num_slices=args.num_slices)
        # ensure evaluation transform
    ds.transform = None
    try:
        from src.utils_image import get_eval_tensor_transform
        ds.transform = get_eval_tensor_transform(img_size=config.IMG_SIZE)
    except Exception:
        ds.transform = None

    from torch.utils.data import DataLoader
    from src.train import make_model, collate_fn

    dl = DataLoader(ds, batch_size=args.bs, collate_fn=collate_fn)
    in_ch = ds.num_channels
    model = make_model(num_classes=2, pretrained=False, in_channels=in_ch)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))

    model = model.to(device)
    out = evaluate(model, dl, device=device)
    print("Eval results:")
    print(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--sequences", default=None, help="comma separated sequence keywords (e.g. t2,adc)")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--num-slices", type=int, default=1, help="odd number of slices to stack (e.g. 3)")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--toy-len", type=int, default=64)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
