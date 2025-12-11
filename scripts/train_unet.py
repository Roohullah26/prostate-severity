"""Train a U-Net on a segmentation dataset created by scripts/prepare_seg.py

Usage (full training):
  py -3 scripts/train_unet.py --data seg_dataset --epochs 100 --batch 8 --aug aggressive

Includes strong augmentation, learning rate scheduling, early stopping, and multi-scale architecture.
For high accuracy.
"""

import argparse
from pathlib import Path
import random
import json
from PIL import Image, ImageOps
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SegDataset(Dataset):
    def __init__(self, root: str, split: str, aug_level='none'):
        self.root = Path(root)
        self.img_dir = self.root / split / 'images'
        self.mask_dir = self.root / split / 'masks'
        self.samples = sorted([p.stem for p in self.img_dir.glob('*.png')])
        self.aug_level = aug_level  # 'none', 'mild', 'aggressive'

    def __len__(self):
        return len(self.samples)

    def _augment(self, img, mask):
        """Apply augmentation to both image and mask consistently."""
        if self.aug_level == 'none':
            return img, mask
        
        # Random flip + rotation + elastic
        if np.random.rand() < 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        if np.random.rand() < 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        
        # Small rotations
        if np.random.rand() < 0.4:
            angle = np.random.uniform(-15, 15)
            img = img.rotate(angle, fillcolor=128)
            mask = mask.rotate(angle, fillcolor=0)
        
        # Intensity augmentation on image only
        img_arr = np.array(img).astype(np.float32) / 255.0
        if self.aug_level == 'aggressive':
            # brightness, contrast, gamma
            if np.random.rand() < 0.5:
                img_arr = np.power(img_arr, np.random.uniform(0.8, 1.2))
            if np.random.rand() < 0.5:
                img_arr = img_arr * np.random.uniform(0.8, 1.2)
            img_arr = np.clip(img_arr, 0, 1)
        
        img = Image.fromarray((img_arr * 255).astype(np.uint8))
        return img, mask

    def __getitem__(self, idx):
        name = self.samples[idx]
        img = Image.open(self.img_dir / (name + '.png')).convert('RGB')
        mask = Image.open(self.mask_dir / (name + '.png')).convert('L')
        
        # Apply augmentation
        img, mask = self._augment(img, mask)
        
        img = np.array(img).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32) / 255.0
        # HWC -> CHW
        img = np.transpose(img, (2,0,1)).astype(np.float32)
        mask = np.expand_dims(mask, 0).astype(np.float32)
        # to tensors
        img_t = torch.tensor(img, dtype=torch.float32)
        mask_t = torch.tensor(mask, dtype=torch.float32)
        return img_t, mask_t


class UNet(nn.Module):
    """Simple working U-Net architecture."""
    def __init__(self, in_ch=3, base=16):
        super().__init__()
        # Encoder (downsampling)
        self.enc1 = self._conv_block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._conv_block(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base*2, base*4)
        
        # Decoder (upsampling)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base*4, base*2)  # after concat: base*2 + base*2 = base*4
        
        self.up1 = nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base*2, base)  # after concat: base + base = base*2
        
        self.out = nn.Conv2d(base, 1, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2)
        
        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)
        
        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)
        
        out = self.out(d1)
        return torch.sigmoid(out)


def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for segmentation (better for imbalanced masks)."""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1.0 - dice


def combined_loss(pred, target):
    """Combine BCE and Dice loss."""
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dice


def train_one_epoch(net, loader, opt, device):
    net.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        preds = net(imgs)
        loss = combined_loss(preds, masks)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset) if len(loader.dataset) else 0.0


def eval_one_epoch(net, loader, device):
    net.eval()
    total_loss = 0.0
    total_dice = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = net(imgs)
            loss = combined_loss(preds, masks)
            d = 1.0 - dice_loss(preds, masks)
            total_loss += loss.item() * imgs.size(0)
            total_dice += d.item() * imgs.size(0)
    n = len(loader.dataset) if len(loader.dataset) else 1
    return total_loss / n, total_dice / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='seg_dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--aug', default='aggressive', choices=['none', 'mild', 'aggressive'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    print(f'training with {args.epochs} epochs, batch={args.batch}, aug={args.aug}')

    root = Path(args.data)
    if not root.exists():
        raise FileNotFoundError('dataset not found: ' + str(root))

    train_ds = SegDataset(args.data, 'train', aug_level=args.aug)
    val_ds = SegDataset(args.data, 'val', aug_level='none')
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=max(1,args.batch), shuffle=False, num_workers=0)
    
    print(f'train samples: {len(train_ds)}, val samples: {len(val_ds)}')

    net = UNet(in_ch=3, base=16).to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_val_dice = 0.0
    patience_counter = 0
    max_patience = 15
    
    out_dir = root / 'train_unet_out'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / 'train_log.txt'

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(net, train_loader, opt, device)
        val_loss, val_dice = eval_one_epoch(net, val_loader, device)
        
        msg = f'Epoch {epoch:03d}: train_loss={tr:.4f}, val_loss={val_loss:.4f}, val_dice={val_dice:.4f}'
        print(msg)
        with open(log_file, 'a') as fh:
            fh.write(msg + '\n')
        
        scheduler.step(val_loss)
        
        # save best based on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_dice = val_dice
            torch.save(net.state_dict(), out_dir / 'best.pt')
            patience_counter = 0
            print(f'  -> saved best checkpoint (loss={val_loss:.4f}, dice={val_dice:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f'Early stopping at epoch {epoch} (no improvement for {max_patience} epochs)')
                break

    print(f'training finished; best_val_loss={best_val_loss:.4f}, best_val_dice={best_val_dice:.4f}')
    with open(log_file, 'a') as fh:
        fh.write(f'\nFinal: best_val_loss={best_val_loss:.4f}, best_val_dice={best_val_dice:.4f}\n')


if __name__ == '__main__':
    main()
