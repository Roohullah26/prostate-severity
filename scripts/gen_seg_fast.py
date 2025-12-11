"""Standalone fast segmentation dataset generator (no external imports for path logic)."""
import os
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import pydicom
from glob import glob

def _ensure(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def build_uid_map_local(dicom_root):
    uid_map = {}
    for series in glob(os.path.join(dicom_root, "*", "*", "*")) + glob(os.path.join(dicom_root, "*", "*")):
        dcm_list = glob(os.path.join(series, "*.dcm"))
        if not dcm_list: 
            continue
        try:
            ds = pydicom.dcmread(dcm_list[0], stop_before_pixels=True)
            uid_map[str(ds.SeriesInstanceUID)] = os.path.dirname(dcm_list[0])
        except:
            continue
    return uid_map

def parse_pos(pos):
    if pos is None: 
        return None
    if isinstance(pos, str):
        parts = pos.strip().split()
        if len(parts) != 3: 
            return None
        try:
            return [float(x) for x in parts]
        except:
            return None
    if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) == 3:
        return [float(x) for x in pos]
    return None

def world_to_pixel(ipp, iop, spacing, world_point):
    ipp = np.array(ipp, dtype=float)
    iop = np.array(iop, dtype=float)
    row_cos = iop[:3]
    col_cos = iop[3:]
    vec = np.array(world_point, dtype=float) - ipp
    row_spacing, col_spacing = float(spacing[0]), float(spacing[1])
    r = float(np.dot(vec, row_cos) / row_spacing)
    c = float(np.dot(vec, col_cos) / col_spacing)
    return r, c

def make_circle_mask(shape, center, radius_px):
    h, w = shape
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    cx = float(center[1])
    cy = float(center[0])
    bbox = [cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px]
    draw.ellipse(bbox, fill=255)
    return np.array(mask, dtype=np.uint8)

def get_series_slices(series_path):
    dcm_files = sorted(glob(os.path.join(series_path, "*.dcm")))
    items = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None and len(ipp) >= 3:
                z = float(ipp[2])
            else:
                z = float(getattr(ds, "SliceLocation", 0.0))
            items.append((z, f))
        except:
            continue
    items.sort(key=lambda x: x[0])
    return items

# MAIN EXECUTION
print('loading CSV...')
df = pd.read_csv('merged_data.csv')
print(f'CSV rows: {len(df)}')

print('building UID map...')
uid_map = build_uid_map_local('data/PROSTATEx')
print(f'UID map entries: {len(uid_map)}')

out_dir = Path('seg_dataset_full')
_ensure(out_dir)
log_fp = out_dir / 'prepare_seg.log'

def log(m):
    try:
        with open(log_fp, 'a', encoding='utf-8') as fh:
            fh.write(m + '\n')
    except:
        pass

log(f'Starting dataset build: {len(df)} rows')

df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
split = 0.85
n_train = int(len(df) * split)
parts = [('train', df.iloc[:n_train]), ('val', df.iloc[n_train:])]

saved_count = {'train': 0, 'val': 0}
skip_count = {'train': 0, 'val': 0}

for split_name, subset in parts:
    img_dir = out_dir / split_name / 'images'
    mask_dir = out_dir / split_name / 'masks'
    _ensure(img_dir)
    _ensure(mask_dir)
    
    print(f'{split_name}: processing {len(subset)} rows...')

    for i, row in subset.iterrows():
        series_uid = str(row.get('Series UID', ''))
        series_path = uid_map.get(series_uid)
        if not series_path:
            skip_count[split_name] += 1
            continue

        lesion_pos = parse_pos(row.get('pos'))

        try:
            slices = get_series_slices(series_path)
            if len(slices) == 0:
                skip_count[split_name] += 1
                continue
            
            if lesion_pos is None:
                pick_idx = len(slices) // 2
            else:
                zs = [z for z, _ in slices]
                pick_idx = int(np.argmin(np.abs(np.array(zs) - float(lesion_pos[2]))))

            dcm_fp = slices[pick_idx][1]
            dsc = pydicom.dcmread(dcm_fp)
            arr = dsc.pixel_array
            rows, cols = int(getattr(dsc, 'Rows', arr.shape[0])), int(getattr(dsc, 'Columns', arr.shape[1]))
            ipp = getattr(dsc, 'ImagePositionPatient', None)
            iop = getattr(dsc, 'ImageOrientationPatient', None)
            spacing = getattr(dsc, 'PixelSpacing', None)

            if ipp is None or iop is None or spacing is None:
                skip_count[split_name] += 1
                continue

            if lesion_pos is None:
                center = (rows/2.0, cols/2.0)
            else:
                center = world_to_pixel(ipp, iop, spacing, lesion_pos)

            avg_mm_per_px = (float(spacing[0]) + float(spacing[1])) / 2.0
            radius_px = max(1, int(10.0 / avg_mm_per_px))

            if arr.ndim == 3:
                arr0 = arr[..., 0]
            else:
                arr0 = arr

            mask = make_circle_mask(arr0.shape, center, radius_px)

            out_name = f'img_{split_name}_{i}'
            out_img = img_dir / f'{out_name}.png'
            out_mask = mask_dir / f'{out_name}.png'

            darr = arr0.astype(np.float32)
            darr -= darr.min()
            if darr.max() > 0:
                darr = darr / darr.max() * 255.0
            darr = darr.astype(np.uint8)

            im = Image.fromarray(darr).convert('RGB').resize((256, 256))
            msk = Image.fromarray(mask).resize((256, 256))

            im.save(out_img)
            msk.save(out_mask)

            meta = {
                'series_uid': series_uid,
                'pixel_spacing': [float(spacing[0]), float(spacing[1])],
                'lesion_world_pos': lesion_pos,
                'lesion_pixel': [float(center[0]), float(center[1])],
                'radius_mm': 10.0,
            }
            meta_fp = img_dir.parent / 'meta' / f'{out_name}.json'
            _ensure(meta_fp.parent)
            with open(meta_fp, 'w', encoding='utf-8') as fh:
                json.dump(meta, fh)
                fh.flush()
                os.fsync(fh.fileno())

            saved_count[split_name] += 1
            if saved_count[split_name] % 100 == 0:
                print(f'  {split_name}: saved {saved_count[split_name]} samples')
        except Exception as e:
            skip_count[split_name] += 1
            continue

print(f'\nDataset generation complete:')
print(f'  train: saved {saved_count["train"]}, skipped {skip_count["train"]}')
print(f'  val: saved {saved_count["val"]}, skipped {skip_count["val"]}')
log('Dataset generation complete')
