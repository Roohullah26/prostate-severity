"""
Training script for multi-sequence tumor size predictor.

This script trains the TumorSizePredictor model on the ProstateX dataset.
It requires a CSV with ground truth tumor dimensions or creates synthetic targets
from existing bounding box annotations.

Usage (toy):
    python -m src.train_size_model --toy --epochs 5 --bs 16

Usage (real data with size annotations):
    python -m src.train_size_model \\
        --csv merged_data.csv \\
        --size-csv tumor_sizes.csv \\
        --sequences t2,adc,dwi \\
        --epochs 20 \\
        --bs 8

The training will save the best model to models/tumor_size_predictor.pth
"""

import argparse
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from collections import Counter

from src.prostate_dataset import ProstateLesionDataset
from src.size_predictor_model import TumorSizePredictor, SizeRegressionLoss
from src.utils_image import pil_to_tensor, get_train_tensor_transform, get_eval_tensor_transform
from src import config


class SizeDataset(ProstateLesionDataset):
    """Extension of ProstateLesionDataset that includes size and severity labels.
    
    Expects the CSV to have additional columns:
    - tumor_width_mm: Ground truth width in mm
    - tumor_height_mm: Ground truth height in mm
    - tumor_depth_mm: Ground truth depth in mm (optional)
    - severity_grade: T-stage (0=T1, 1=T2, 2=T3, 3=T4)
    
    If these columns don't exist, they're generated synthetically for testing.
    """
    
    def __init__(self, csv_path=None, img_size=config.IMG_SIZE, toy=False, 
                 toy_len=64, sequences=None, num_slices=1, transform=None):
        super().__init__(
            csv_path=csv_path,
            img_size=img_size,
            toy=toy,
            toy_len=toy_len,
            sequences=sequences,
            num_slices=num_slices,
            transform=transform
        )
        
        if toy:
            # Generate random size targets for toy data
            self.df['tumor_width_mm'] = np.random.uniform(5, 40, len(self.df))
            self.df['tumor_height_mm'] = np.random.uniform(5, 40, len(self.df))
            self.df['tumor_depth_mm'] = np.random.uniform(5, 40, len(self.df))
            # Compute severity from max dimension
            max_size = self.df[['tumor_width_mm', 'tumor_height_mm', 'tumor_depth_mm']].max(axis=1)
            self.df['severity_grade'] = pd.cut(max_size, bins=[0, 20, 40, 60, 100], 
                                                labels=[0, 1, 2, 3]).astype(int)
        else:
            # For real data: create synthetic sizes if not present
            # In practice, you'd provide these via the CSV
            if 'tumor_width_mm' not in self.df.columns:
                print("Warning: tumor_width_mm not in CSV, generating synthetic sizes for demo")
                # Synthetic: sizes based on ClinSig status
                self.df['tumor_width_mm'] = self.df.get('ClinSig', 0).astype(int) * 15 + np.random.uniform(5, 20, len(self.df))
                self.df['tumor_height_mm'] = self.df.get('ClinSig', 0).astype(int) * 15 + np.random.uniform(5, 20, len(self.df))
                self.df['tumor_depth_mm'] = self.df.get('ClinSig', 0).astype(int) * 10 + np.random.uniform(5, 15, len(self.df))
            
            if 'severity_grade' not in self.df.columns:
                print("Warning: severity_grade not in CSV, generating from size")
                max_size = self.df[['tumor_width_mm', 'tumor_height_mm', 'tumor_depth_mm']].max(axis=1)
                self.df['severity_grade'] = pd.cut(max_size, bins=[0, 20, 40, 60, 100],
                                                    labels=[0, 1, 2, 3]).astype(int).fillna(0)
    
    def __getitem__(self, idx):
        """Return (image, target_dict) where target_dict has size and severity."""
        # Get image from parent class
        if self.toy:
            from PIL import Image
            img = Image.new('RGB', self.img_size, color=(128, 128, 128))
        else:
            # Use parent's image loading logic
            img = super().__getitem__(idx)
            if isinstance(img, tuple):
                img, _ = img
        
        # Get size and severity targets
        row = self.df.iloc[idx]
        size_target = np.array([
            float(row['tumor_width_mm']),
            float(row['tumor_height_mm']),
            float(row['tumor_depth_mm']),
        ], dtype=np.float32)
        
        severity_target = int(row['severity_grade'])
        
        return img, {
            'size': size_target,
            'severity': severity_target,
        }


def collate_fn_size(batch):
    """Collate function for size dataset."""
    from src.utils_image import pil_to_tensor
    
    imgs = []
    sizes = []
    severities = []
    
    for img, targets in batch:
        if isinstance(img, torch.Tensor):
            imgs.append(img)
        else:
            imgs.append(pil_to_tensor(img, img_size=config.IMG_SIZE))
        sizes.append(targets['size'])
        severities.append(targets['severity'])
    
    imgs = torch.stack(imgs)
    sizes = torch.tensor(np.array(sizes), dtype=torch.float32)
    severities = torch.tensor(severities, dtype=torch.long)
    
    return imgs, {
        'size': sizes,
        'severity': severities,
    }


def train_size_model(args):
    """Main training loop."""
    device = "cuda" if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu"
    print(f"Using device: {device}")
    
    # Parse sequences
    seqs = None if args.sequences is None else [s.strip().lower() for s in args.sequences.split(",") if s.strip()]
    
    # Load dataset
    if args.toy:
        ds = SizeDataset(toy=True, toy_len=args.toy_len, num_slices=args.num_slices)
    else:
        ds = SizeDataset(
            csv_path=args.csv,
            img_size=config.IMG_SIZE,
            sequences=seqs,
            num_slices=args.num_slices
        )
    
    # Apply transforms
    if args.augment:
        ds.transform = get_train_tensor_transform(
            img_size=config.IMG_SIZE, p_hflip=0.5, p_vflip=0.25, bright_jitter=0.08
        )
    else:
        ds.transform = get_eval_tensor_transform(img_size=config.IMG_SIZE)
    
    # DataLoader
    dl = DataLoader(
        ds,
        batch_size=args.bs,
        shuffle=True,
        collate_fn=collate_fn_size,
        num_workers=0
    )
    
    print(f"Dataset size: {len(ds)}")
    print(f"Number of channels: {ds.num_channels}")
    print(f"Batch size: {args.bs}")
    
    # Model
    in_ch = ds.num_channels
    pretrained_flag = (not args.toy) and in_ch == 3
    model = TumorSizePredictor(pretrained=pretrained_flag, in_channels=in_ch)
    model = model.to(device)
    
    print(f"Model created. Pretrained: {pretrained_flag}, Input channels: {in_ch}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    loss_fn = SizeRegressionLoss(alpha=args.size_weight, beta=args.severity_weight)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_loss = float('inf')
    best_model_path = config.MODELS_DIR / 'tumor_size_predictor_best.pth'
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_losses = {'size': 0.0, 'severity': 0.0, 'confidence': 0.0}
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (imgs, targets) in enumerate(dl):
            imgs = imgs.to(device)
            targets = {
                'size': targets['size'].to(device),
                'severity': targets['severity'].to(device),
            }
            
            # Forward pass
            opt.zero_grad()
            predictions = model(imgs)
            
            # Loss
            loss, loss_dict = loss_fn(predictions, targets)
            
            # Backward
            loss.backward()
            opt.step()
            
            # Accumulate
            epoch_loss += loss.item()
            for key in epoch_losses:
                epoch_losses[key] += loss_dict.get(key, 0.0)
            num_batches += 1
            
            if (batch_idx + 1) % args.log_interval == 0:
                print(f"  Batch {batch_idx + 1}/{len(dl)}: "
                      f"loss={loss.item():.4f}, "
                      f"size={loss_dict['size']:.4f}, "
                      f"severity={loss_dict['severity']:.4f}")
        
        # Epoch summary
        epoch_loss /= num_batches
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1}/{args.epochs} - "
              f"Loss: {epoch_loss:.4f}, "
              f"Time: {elapsed:.2f}s, "
              f"LR: {opt.param_groups[0]['lr']:.2e}")
        print(f"  Size loss: {epoch_losses['size']:.4f}, "
              f"Severity loss: {epoch_losses['severity']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(epoch_loss)
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"  -> Saving best model (loss: {epoch_loss:.4f})")
            torch.save(model.state_dict(), best_model_path)
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = config.MODELS_DIR / f'tumor_size_predictor_ep{epoch + 1}.pth'
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved checkpoint: {ckpt_path}")
    
    print(f"\nTraining complete. Best model saved to {best_model_path}")
    print(f"Best loss: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train tumor size predictor")
    
    parser.add_argument("--csv", default=None, help="Path to merged_data.csv")
    parser.add_argument("--size-csv", default=None, help="CSV with tumor size annotations")
    parser.add_argument("--toy", action="store_true", help="Use toy dataset")
    parser.add_argument("--toy-len", type=int, default=64, help="Toy dataset size")
    parser.add_argument("--sequences", default="t2,adc", help="Comma-separated sequence keywords")
    parser.add_argument("--num-slices", type=int, default=1, help="Number of slices to stack (odd)")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--size-weight", type=float, default=1.0, help="Weight for size loss")
    parser.add_argument("--severity-weight", type=float, default=0.5, help="Weight for severity loss")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval (batches)")
    parser.add_argument("--save-interval", type=int, default=5, help="Checkpoint save interval (epochs)")
    
    args = parser.parse_args()
    
    train_size_model(args)


if __name__ == "__main__":
    main()
