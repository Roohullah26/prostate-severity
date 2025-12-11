"""Monitor dataset generation and long training run, print status and final summary.

This script polls for:
 - 'seg_dataset_full/prepare_seg.log' containing 'Dataset generation complete'
 - 'train_unet_long.log' containing 'Training finished with code'

It prints periodic updates to stdout. Run in background.
"""
import time
from pathlib import Path

DATA_LOG = Path('seg_dataset_full') / 'prepare_seg.log'
TRAIN_LOG = Path('train_unet_long.log')
STATUS = Path('run_status.txt')

print('monitor_run started; polling every 30s')

def contains(path, substr):
    try:
        if not path.exists():
            return False
        text = path.read_text(errors='ignore')
        return substr in text
    except Exception:
        return False

last_dataset_status = False
last_train_started = False
last_train_finished = False

while True:
    ds_done = contains(DATA_LOG, 'Dataset generation complete')
    if ds_done and not last_dataset_status:
        print('Dataset generation complete detected at', time.ctime())
        last_dataset_status = True
    elif not ds_done:
        print('Waiting for dataset generation to complete... (now', time.ctime(),')')

    # check train_unet_long.log; if exists and contains 'Starting training', mark started
    if TRAIN_LOG.exists():
        ttxt = TRAIN_LOG.read_text(errors='ignore')
        if 'Starting training' in ttxt and not last_train_started:
            print('Long training started (train_unet_long.log found) at', time.ctime())
            last_train_started = True
        if 'Training finished with code' in ttxt:
            print('Long training finished at', time.ctime())
            print('--- Last 400 chars of train_unet_long.log ---')
            print(ttxt[-400:])
            last_train_finished = True
    else:
        pass

    if last_dataset_status and last_train_finished:
        print('Both dataset generation and training finished. Exiting monitor.')
        STATUS.write_text(f'finished at {time.ctime()}')
        break

    time.sleep(30)
