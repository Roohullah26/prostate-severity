"""Monitor dataset generation and training progress, report when complete."""
import time
import os
from pathlib import Path

DATA_DIR = Path('seg_dataset_full')
TRAIN_IMG_DIR = DATA_DIR / 'train' / 'images'
VAL_IMG_DIR = DATA_DIR / 'val' / 'images'
DATA_LOG = DATA_DIR / 'prepare_seg.log'
TRAIN_LOG = DATA_DIR / 'train_unet_out' / 'train_log.txt'

last_data_count = 0
last_train_epoch = 0
dataset_done = False
training_done = False

print('=' * 70)
print('MONITORING: Dataset generation + U-Net training')
print('=' * 70)

while not (dataset_done and training_done):
    # Check dataset
    if TRAIN_IMG_DIR.exists():
        train_count = len(list(TRAIN_IMG_DIR.glob('*.png')))
        val_count = len(list(VAL_IMG_DIR.glob('*.png'))) if VAL_IMG_DIR.exists() else 0
        
        if train_count > last_data_count or val_count > 0:
            print(f'[DATASET] train: {train_count} images, val: {val_count} images')
            last_data_count = train_count
        
        # Check if dataset generation finished
        if DATA_LOG.exists():
            log_text = DATA_LOG.read_text(encoding='utf-8', errors='ignore')
            if 'Dataset generation complete' in log_text or 'saved' in log_text:
                if not dataset_done:
                    print('[DATASET] ✓ Dataset generation COMPLETE')
                    dataset_done = True
    
    # Check training
    if TRAIN_LOG.exists():
        log_lines = TRAIN_LOG.read_text(encoding='utf-8', errors='ignore').strip().split('\n')
        
        # Extract last epoch number
        for line in reversed(log_lines[-20:]):
            if 'Epoch' in line:
                try:
                    epoch_num = int(line.split('Epoch')[1].split(':')[0])
                    if epoch_num > last_train_epoch:
                        print(f'[TRAINING] {line}')
                        last_train_epoch = epoch_num
                except:
                    pass
                break
        
        # Check if training finished
        if any(k in log_lines[-1] for k in ['Final:', 'finished', 'complete']):
            if not training_done:
                print('[TRAINING] ✓ Training COMPLETE')
                print(f'[TRAINING] {log_lines[-1]}')
                training_done = True
    
    if dataset_done and training_done:
        break
    
    time.sleep(30)

print('=' * 70)
print('✓✓✓ BOTH DATASET GENERATION AND TRAINING COMPLETE ✓✓✓')
print('=' * 70)
print('')
print('Next steps:')
print(f'  - Dataset: {DATA_DIR}')
print(f'  - Training outputs: {DATA_DIR}/train_unet_out/')
print(f'  - Best model: {DATA_DIR}/train_unet_out/best.pt')
print(f'  - Training log: {TRAIN_LOG}')
print('')
