"""Wait for segmentation dataset to be ready, then start long U-Net training and log output.

Usage: run this in the repo root. It will block until `seg_dataset_full/train/images` contains files.
"""
import time
import subprocess
from pathlib import Path

DATA_DIR = Path('seg_dataset_full')
TRAIN_IMG_DIR = DATA_DIR / 'train' / 'images'
LOG = Path('train_unet_long.log')

print('auto-trainer started; waiting for dataset...')
while True:
    if TRAIN_IMG_DIR.exists() and any(TRAIN_IMG_DIR.glob('*.png')):
        print('dataset detected; starting training')
        break
    print('dataset not ready yet; sleeping 10s')
    time.sleep(10)

cmd = ['py', '-3', 'scripts/train_unet.py', '--data', str(DATA_DIR), '--epochs', '200', '--batch', '4', '--aug', 'aggressive']
print('running:', ' '.join(cmd))
with open(LOG, 'a', encoding='utf-8') as fh:
    fh.write('Starting training\n')
    proc = subprocess.Popen(cmd, stdout=fh, stderr=fh)
    proc.wait()
    fh.write('Training finished with code ' + str(proc.returncode) + '\n')
print('training subprocess finished; check', LOG)
