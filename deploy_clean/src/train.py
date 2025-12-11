import argparse
import os
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import models

from src.prostate_dataset import ProstateLesionDataset
from src.utils_image import pil_to_tensor, get_train_tensor_transform, get_eval_tensor_transform
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler
from src import config


def collate_fn(batch):
    # batch: list of (image, label) where image can be PIL.Image or torch.Tensor
    imgs, labels = zip(*batch)
    if isinstance(imgs[0], torch.Tensor):
        # assume tensors are already shaped (C,H,W) and float32
        imgs = torch.stack(imgs)
    else:
        imgs = [pil_to_tensor(img, img_size=config.IMG_SIZE) for img in imgs]
        imgs = torch.stack(imgs)

    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels


def make_model(num_classes=2, pretrained=True, in_channels=3):
    # If the model needs non-3 channel inputs, construct a conv1 that accepts in_channels.
    # When using pretrained weights, only allow pretrained if in_channels == 3.
    use_pretrained = pretrained and in_channels == 3
    model = models.resnet18(pretrained=use_pretrained)
    in_feat = model.fc.in_features
    # adapt conv1 to accept in_channels
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model.fc = nn.Linear(in_feat, num_classes)
    return model


def train(args):
    device = "cuda" if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu"

    # support sequences: comma-separated list of sequence keywords (e.g. t2,adc)
    seqs = None if args.sequences is None else [s.strip().lower() for s in args.sequences.split(",") if s.strip()]

    if args.toy:
        ds = ProstateLesionDataset(toy=True, toy_len=args.toy_len, num_slices=args.num_slices)
    else:
        ds = ProstateLesionDataset(csv_path=args.csv, img_size=config.IMG_SIZE, sequences=seqs, num_slices=args.num_slices)

    # assign transforms (augmentations) for stacked/multi-channel or single-image cases
    if args.augment:
        ds.transform = get_train_tensor_transform(img_size=config.IMG_SIZE, p_hflip=0.5, p_vflip=0.25, bright_jitter=0.08)
    else:
        ds.transform = get_eval_tensor_transform(img_size=config.IMG_SIZE)

    # class balancing
    sampler = None
    if not args.toy and args.balance in ("weighted", "oversample"):
        # compute sample weights inverse to class frequency
        labels = ds.df["label"].tolist()
        counts = Counter(labels)
        total = len(labels)
        class_weights = {c: total / (counts[c] * len(counts)) for c in counts}
        sample_weights = [class_weights[int(l)] for l in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    if sampler is not None:
        dl = DataLoader(ds, batch_size=args.bs, sampler=sampler, collate_fn=collate_fn, num_workers=0)
    else:
        dl = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=collate_fn, num_workers=0)

    in_ch = ds.num_channels
    # prefer pretrained only when not toy and in_ch == 3
    pretrained_flag = (not args.toy) and in_ch == 3
    model = make_model(num_classes=2, pretrained=pretrained_flag, in_channels=in_ch)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optionally set class weights for loss
    crit = None
    if not args.toy and args.loss_weight == "auto":
        labels = ds.df["label"].tolist()
        counts = Counter(labels)
        weights = [0.0] * (max(counts.keys()) + 1)
        total = len(labels)
        for k, v in counts.items():
            weights[int(k)] = total / (v + 1e-12)
        w = torch.tensor(weights, dtype=torch.float32).to(device)
        crit = nn.CrossEntropyLoss(weight=w)
    else:
        crit = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        count = 0
        t0 = time.time()
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            running += loss.item() * xb.size(0)
            count += xb.size(0)

        epoch_loss = running / max(1, count)
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {epoch_loss:.4f}  time:{time.time()-t0:.1f}s")

    # save model
    out_dir = Path(config.MODELS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (args.save_name or "prototype_toy.pth")
    torch.save(model.state_dict(), out_file)
    print("Saved model ->", out_file)

    # export options
    if args.export:
        try:
            ts_out = out_dir / (args.save_name or "prototype_toy")
            # create scripted module
            model.eval()
            example = torch.randn(1, in_ch, config.IMG_SIZE[0], config.IMG_SIZE[1]).to(device)
            scripted = torch.jit.trace(model, example)
            scripted_path = ts_out.with_suffix('.pt')
            scripted.save(scripted_path)
            print('Saved TorchScript ->', scripted_path)
            # export ONNX
            onnx_path = ts_out.with_suffix('.onnx')
            torch.onnx.export(model, example, onnx_path, opset_version=12)
            print('Saved ONNX ->', onnx_path)
        except Exception as e:
            print('Export failed:', e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy", action="store_true", help="run a tiny smoke-training job (no DICOM required)")
    parser.add_argument("--toy-len", type=int, default=64)
    parser.add_argument("--csv", default=None, help="merged csv path")
    parser.add_argument("--sequences", default=None, help="optional comma-separated sequence keywords (e.g. t2,adc). If provided, dataset will attempt multi-sequence stacking")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--num-slices", type=int, default=1, help="odd number of axial slices to stack around lesion (e.g. 3)")
    parser.add_argument("--augment", action="store_true", help="enable data augmentation transforms")
    parser.add_argument("--balance", choices=["none", "weighted", "oversample"], default="none", help="class balancing strategy")
    parser.add_argument("--loss-weight", choices=["none", "auto"], default="none", help="auto-compute class weights for loss")
    parser.add_argument("--export", action="store_true", help="export final model to TorchScript and ONNX")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-name", default=None)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
