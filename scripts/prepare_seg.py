"""Prepare a segmentation dataset (images + binary masks) from merged_data.csv + DICOM series.

This produces a folder structure under <out>/<train|val> with:
  images/  (rgb PNG resized to target size)
  masks/   (binary PNG masks matching resized images)

Currently this script creates circular masks centered at the lesion world coordinates
(column `pos`) using a fixed radius in mm (default 10mm). This is a prototype — for
high accuracy you should replace circles with manual per-slice masks.

Usage (quick):
  py -3 scripts/prepare_seg.py --csv merged_data.csv --dicom data/PROSTATEx --out seg_dataset --limit 256

The script writes a prepare_seg.log file into the output folder to ease debugging.
"""

import argparse
import os
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd

from src.utils_dicom import build_uid_map, parse_pos


def _ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def world_to_pixel(ipp, iop, spacing, world_point):
    """Convert world (patient) coordinates to pixel (row, col).

    ipp: 3-float list ImagePositionPatient
    iop: 6-float list ImageOrientationPatient (row_cosine[3], col_cosine[3])
    spacing: list/tuple [row_spacing, col_spacing]
    world_point: 3-float target world coordinate
    returns (row_float, col_float)
    """
    ipp = np.array(ipp, dtype=float)
    iop = np.array(iop, dtype=float)
    row_cos = iop[:3]
    col_cos = iop[3:]
    vec = np.array(world_point, dtype=float) - ipp
    row_spacing, col_spacing = float(spacing[0]), float(spacing[1])
    # projection lengths
    r = float(np.dot(vec, row_cos) / row_spacing)
    c = float(np.dot(vec, col_cos) / col_spacing)
    return r, c


def make_circle_mask(shape, center, radius_px):
    # shape: (H,W), center: (row,col) floats
    h, w = shape
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    cx = float(center[1])
    cy = float(center[0])
    bbox = [cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px]
    draw.ellipse(bbox, fill=255)
    return np.array(mask, dtype=np.uint8)


def build_dataset(csv_fp, dicom_root, out_dir, split=0.9, radius_mm=10.0, img_size=(256,256), limit=0):
    import pandas as pd
    csv_fp = Path(csv_fp)
    out_dir = Path(out_dir)
    df = pd.read_csv(csv_fp)

    uid_map = build_uid_map(dicom_root)

    _ensure(out_dir)
    log_fp = out_dir / 'prepare_seg.log'

    def log(m):
        try:
            with open(log_fp, 'a', encoding='utf-8') as fh:
                fh.write(m + '\n')
        except Exception:
            pass

    log(f'Building segmentation dataset from {csv_fp} using dicom root {dicom_root} (uid_map_len={len(uid_map)})')

    if limit and limit > 0:
        df = df.iloc[:limit]

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_train = int(len(df) * split)
    parts = [('train', df.iloc[:n_train]), ('val', df.iloc[n_train:])]

    for split_name, subset in parts:
        img_dir = out_dir / split_name / 'images'
        mask_dir = out_dir / split_name / 'masks'
        _ensure(img_dir)
        _ensure(mask_dir)

        for i, row in subset.iterrows():
            series_uid = str(row.get('Series UID', ''))
            series_path = uid_map.get(series_uid)
            if not series_path:
                log(f'skipping {i}: series uid {series_uid} not found')
                continue

            lesion_pos = parse_pos(row.get('pos'))

            # select slice closest to lesion z
            try:
                from src.prostate_dataset import ProstateLesionDataset
                ds_helper = ProstateLesionDataset(csv_path=csv_fp)
                slices = ds_helper._get_series_slices(series_path)
                if len(slices) == 0:
                    log(f'no DICOM slices in series_path {series_path} (idx {i})')
                    continue
                if lesion_pos is None:
                    pick_idx = len(slices) // 2
                else:
                    zs = [z for z, _ in slices]
                    pick_idx = int(np.argmin(np.abs(np.array(zs) - float(lesion_pos[2]))))

                dcm_fp = slices[pick_idx][1]
                import pydicom
                dsc = pydicom.dcmread(dcm_fp)
                arr = dsc.pixel_array
                rows, cols = int(getattr(dsc, 'Rows', arr.shape[0])), int(getattr(dsc, 'Columns', arr.shape[1]))
                ipp = getattr(dsc, 'ImagePositionPatient', None)
                iop = getattr(dsc, 'ImageOrientationPatient', None)
                spacing = getattr(dsc, 'PixelSpacing', None)

                if ipp is None or iop is None or spacing is None:
                    log(f'missing geometry in {dcm_fp} — ipp/iop/spacing required; skipping idx {i}')
                    continue

                # compute pixel coordinate center
                if lesion_pos is None:
                    # fallback to image center
                    center = (rows/2.0, cols/2.0)
                else:
                    try:
                        center = world_to_pixel(ipp, iop, spacing, lesion_pos)
                    except Exception as e:
                        log(f'failed to compute world->pixel for {dcm_fp} idx {i}: {e}')
                        continue

                # radius in px
                avg_mm_per_px = (float(spacing[0]) + float(spacing[1])) / 2.0
                radius_px = max(1, int(radius_mm / avg_mm_per_px))

                # build mask and save
                # ensure arr is 2D
                if arr.ndim == 3:
                    arr0 = arr[..., 0]
                else:
                    arr0 = arr

                mask = make_circle_mask(arr0.shape, center, radius_px)

                # save resized images and resized masks
                out_name = f'img_{split_name}_{i}'
                out_img = img_dir / f'{out_name}.png'
                out_mask = mask_dir / f'{out_name}.png'

                # normalize and convert to RGB 0..255
                darr = arr0.astype(np.float32)
                darr -= darr.min()
                if darr.max() > 0:
                    darr = darr / darr.max() * 255.0
                darr = darr.astype(np.uint8)

                im = Image.fromarray(darr).convert('RGB').resize(img_size)
                msk = Image.fromarray(mask).resize(img_size)

                # atomic write
                tmp_img = out_img.with_suffix('.png.tmp')
                tmp_mask = out_mask.with_suffix('.png.tmp')
                im.save(tmp_img)
                os.replace(tmp_img, out_img)
                msk.save(tmp_mask)
                os.replace(tmp_mask, out_mask)

                meta = {
                    'series_uid': series_uid,
                    'series_path': series_path,
                    'image_shape': [rows, cols],
                    'pixel_spacing': [float(spacing[0]), float(spacing[1])],
                    'lesion_world_pos': lesion_pos,
                    'lesion_pixel': [float(center[0]), float(center[1])],
                    'radius_mm': float(radius_mm),
                }
                meta_fp = img_dir.parent / 'meta' / f'{out_name}.json'
                _ensure(meta_fp.parent)
                tmp_meta = meta_fp.with_suffix('.json.tmp')
                with open(tmp_meta, 'w', encoding='utf-8') as fh:
                    json.dump(meta, fh)
                    fh.flush(); os.fsync(fh.fileno())
                os.replace(tmp_meta, meta_fp)

                log(f'saved {out_img} / {out_mask} / {meta_fp}')
            except Exception as e:
                log(f'exception processing idx {i}: {e}')
                continue

    log('done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='merged_data.csv')
    parser.add_argument('--dicom', default=str(Path(__file__).resolve().parents[1] / 'data' / 'PROSTATEx'))
    parser.add_argument('--out', default='seg_dataset')
    parser.add_argument('--split', type=float, default=0.9)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--radius-mm', type=float, default=10.0)
    parser.add_argument('--img-size', type=int, nargs=2, default=(256,256))
    args = parser.parse_args()

    build_dataset(args.csv, args.dicom, args.out, split=args.split, radius_mm=args.radius_mm, img_size=tuple(args.img_size), limit=args.limit)


if __name__ == '__main__':
    main()
