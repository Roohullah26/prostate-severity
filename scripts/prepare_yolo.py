"""Prepare a YOLO-format detection dataset from merged_data.csv + DICOM series.

This script creates a minimal detection dataset with images in JPEG/PNG and
YOLO-format label txt files. Because we don't have ground-truth bboxes, we
generate approximate bboxes around the image center using a fixed physical size
(e.g. 20 mm box) converted to pixels via DICOM PixelSpacing when available.

Outputs saved under repo/yolo_dataset/<train|val>/images and /labels, and
metadata JSON sidecars containing original DICOM path and PixelSpacing so
inference can compute physical sizes.

Usage (quick):
  py -3 scripts/prepare_yolo.py --csv merged_data.csv --out yolo_dataset --split 0.9

Note: this is a heuristic approach for a prototype. Replace with real bounding-box
annotations for production-quality detection.
"""

import argparse
from pathlib import Path
import pandas as pd
import os
from PIL import Image
import json
import numpy as np

from src.utils_dicom import build_uid_map, parse_pos


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _save_image(arr, out_path: Path):
    # arr is numpy array, 2D or 3D
    im = Image.fromarray(arr.astype('uint8'))
    im.save(out_path)


def _make_label_file(out_label_fp: Path, class_id: int, bbox_norm):
    # YOLO format: <class> <x_center> <y_center> <width> <height> (normalized)
    with open(out_label_fp, 'w') as fh:
        fh.write(f"{class_id} {bbox_norm[0]:.6f} {bbox_norm[1]:.6f} {bbox_norm[2]:.6f} {bbox_norm[3]:.6f}\n")


def build_dataset(csv_fp: str, dicom_root: str, out_dir: str, split: float = 0.9, fixed_mm: float = 20.0, limit: int = 0):
    csv_fp = Path(csv_fp)
    out_dir = Path(out_dir)
    df = pd.read_csv(csv_fp)

    # build uid map for series lookup
    try:
        uid_map = build_uid_map(dicom_root)
    except ModuleNotFoundError as e:
        print('ERROR: unable to build uid map -', e)
        print("Make sure pydicom is installed in your environment (pip install pydicom)")
        return

    print(f'Built series UID map with {len(uid_map)} entries')

    # create a simple log file in out_dir so we can inspect progress later
    _ensure_dir(out_dir)
    log_fp = out_dir / 'prepare_yolo.log'
    def log(msg):
        try:
            with open(log_fp, 'a', encoding='utf-8') as L:
                L.write(msg + '\n')
        except Exception:
            pass

    log(f'Building dataset from {csv_fp} using dicom root {dicom_root} (uid_map_len={len(uid_map)})')

    if len(uid_map) == 0:
        print('WARNING: series UID map is empty; no DICOM series will be found. Double-check the dicom path and nested folder structure.')
        log('uid_map empty; aborting')
        return

    # optionally limit rows for faster testing
    if limit and limit > 0:
        df = df.iloc[:limit]

    # we'll create a small shuffled split
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_train = int(len(df) * split)
    parts = [('train', df.iloc[:n_train]), ('val', df.iloc[n_train:])]

    for split_name, subset in parts:
        img_dir = out_dir / split_name / 'images'
        label_dir = out_dir / split_name / 'labels'
        meta_dir = out_dir / split_name / 'meta'
        _ensure_dir(img_dir)
        _ensure_dir(label_dir)
        _ensure_dir(meta_dir)

        for i, row in subset.iterrows():
            # resolve series path
            series_uid = str(row.get('Series UID', ''))
            series_path = uid_map.get(series_uid)
            if not series_path:
                log(f'skipping index {i}: series uid {series_uid} not found in uid_map')
                continue

            # use singleton: load closest slice using prostate_dataset helper logic
            try:
                from src.prostate_dataset import ProstateLesionDataset
                ds = ProstateLesionDataset(csv_path=csv_fp, sequences=None, num_slices=1)
                arr = ds._load_series_closest_slice(series_path, lesion_pos=parse_pos(row.get('pos')))
            except Exception:
                # fallback: try reading first dcm
                import pydicom
                from glob import glob
                dcm_files = sorted(glob(os.path.join(series_path, '*.dcm')))
                if not dcm_files:
                    log(f'skipping index {i}: no dcm files found in {series_path}')
                    continue
                ds0 = pydicom.dcmread(dcm_files[0])
                arr = ds0.pixel_array

            # normalize to 0..255
            darr = arr.astype(np.float32)
            darr -= darr.min()
            if darr.max() > 0:
                darr = darr / darr.max() * 255.0
            darr = darr.astype(np.uint8)

            # determine PixelSpacing (mm per pixel)
            ps = None
            try:
                import pydicom
                from glob import glob as _glob
                # look across all dicom files in the series for a PixelSpacing value (take the first non-empty)
                dcm_list2 = sorted(_glob(os.path.join(series_path, '*.dcm')))
                ps = None
                for dcmfp in dcm_list2:
                    try:
                        dsc = pydicom.dcmread(dcmfp, stop_before_pixels=True)
                        candidate = getattr(dsc, 'PixelSpacing', None)
                        if candidate:
                            ps = candidate
                            # log which file supplied spacing
                            log(f'pixel_spacing found for series {series_uid} in {dcmfp}: {ps}')
                            break
                    except Exception:
                        continue
                # if no files are present or none contained PixelSpacing, ps remains None
            except Exception:
                ps = None

            # compute box half-size in pixels based on physical mm if possible
            h, w = darr.shape[0], darr.shape[1]
            if ps and isinstance(ps, (list, tuple)) and len(ps) >= 2:
                # average spacing
                avg_mm_per_px = (float(ps[0]) + float(ps[1])) / 2.0
                half_px = max(1, int((fixed_mm / 2.0) / avg_mm_per_px))
            else:
                half_px = int(min(h, w) * 0.05)  # 5% of dimension

            cx = w // 2
            cy = h // 2
            x1 = max(0, cx - half_px)
            y1 = max(0, cy - half_px)
            x2 = min(w - 1, cx + half_px)
            y2 = min(h - 1, cy + half_px)

            # normalize bbox
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            bx = (x1 + x2) / 2.0 / w
            by = (y1 + y2) / 2.0 / h

            # prepare filenames
            out_name = f"img_{split_name}_{i}"
            out_img = img_dir / f"{out_name}.png"
            out_label = label_dir / f"{out_name}.txt"
            out_meta = meta_dir / f"{out_name}.json"

            try:
                _save_image(darr, out_img)
                _make_label_file(out_label, 0, (bx, by, bw, bh))
            except Exception as e:
                msg = f'Failed to save image/label for row {i} series {series_uid}: {e}'
                print(msg)
                log(msg)
                continue

            log(f'saved {out_img} / {out_label} / {out_meta}')

            # normalize pixel spacing to plain list of floats for JSON safety
            def _normalize_ps(pval):
                if pval is None:
                    return None
                # handle pydicom MultiValue, lists/tuples or single values
                try:
                    if isinstance(pval, (list, tuple)):
                        return [float(x) for x in pval]
                    return [float(pval)]
                except Exception:
                    return None

            pixel_spacing_json = _normalize_ps(ps)

            meta = {
                'series_uid': series_uid,
                'series_path': series_path,
                'pixel_spacing': pixel_spacing_json,
                'image_shape': [h, w],
                'bbox_px': [x1, y1, x2, y2],
            }

            # write meta atomically to avoid partial/truncated files on interruption
            tmp_meta = out_meta.with_suffix('.json.tmp')
            try:
                with open(tmp_meta, 'w', encoding='utf-8') as fh:
                    json.dump(meta, fh)
                    fh.flush()
                    os.fsync(fh.fileno())
                # replace is atomic on most OSes
                os.replace(tmp_meta, out_meta)
            except Exception as e:
                # record failure but continue processing other entries
                log(f'Failed to write meta for {out_meta}: {e}')
                if tmp_meta.exists():
                    try:
                        tmp_meta.unlink()
                    except Exception:
                        pass

    print('done')
    log('done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='merged_data.csv')
    parser.add_argument('--dicom', default=str(Path(__file__).resolve().parents[1] / 'data' / 'PROSTATEx'))
    parser.add_argument('--out', default='yolo_dataset')
    parser.add_argument('--split', type=float, default=0.9)
    parser.add_argument('--limit', type=int, default=0, help='limit number of rows processed (0 = all)')
    parser.add_argument('--fixed-mm', type=float, default=20.0, help='box physical size in mm')
    args = parser.parse_args()

    build_dataset(args.csv, args.dicom, args.out, split=args.split, fixed_mm=args.fixed_mm, limit=args.limit)


if __name__ == '__main__':
    main()
