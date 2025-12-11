import argparse
from pathlib import Path
from ultralytics import YOLO
import json
import numpy as np
from PIL import Image


def bbox_to_mm(bbox_pixels, meta):
    # bbox_pixels: x1,y1,x2,y2 in pixel units
    # meta should contain pixel_spacing [row, col]
    ps = meta.get('pixel_spacing')
    if not ps or len(ps) < 2:
        return None
    avg = (float(ps[0]) + float(ps[1])) / 2.0
    w_px = bbox_pixels[2] - bbox_pixels[0]
    h_px = bbox_pixels[3] - bbox_pixels[1]
    w_mm = w_px * avg
    h_mm = h_px * avg
    return (w_mm, h_mm)


def map_severity(size_mm, thresholds=(10.0, 20.0)):
    # thresholds: (small->medium, medium->large) in mm
    if size_mm is None:
        return 'unknown'
    major = float(size_mm)
    if major < thresholds[0]:
        return 'small'
    if major < thresholds[1]:
        return 'medium'
    return 'large'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--meta', required=False, help='optional json metadata sidecar for pixel spacing')
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()

    model = YOLO(args.weights)

    img = Image.open(args.image).convert('RGB')
    results = model.predict(source=str(args.image), conf=args.conf)
    # pick first result
    r = results[0]
    dets = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else []
    scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else []
    labels = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, 'cls') else []

    meta = {}
    if args.meta:
        try:
            with open(args.meta, 'r') as fh:
                meta = json.load(fh)
        except Exception:
            meta = {}

    # If pixel spacing isn't present in the meta sidecar, try to read it from
    # the original DICOM series (meta should have series_path when prepared)
    if not meta.get('pixel_spacing'):
        series_path = meta.get('series_path')
        if series_path:
            try:
                import pydicom
                from glob import glob as _glob
                # attempt to find any .dcm in the series and read PixelSpacing
                dcm_list = sorted(_glob(str(Path(series_path) / '*.dcm')))
                for dcmfp in dcm_list:
                    try:
                        dsc = pydicom.dcmread(dcmfp, stop_before_pixels=True)
                        ps = getattr(dsc, 'PixelSpacing', None)
                        if ps:
                            meta['pixel_spacing'] = ps
                            break
                    except Exception:
                        continue
            except Exception:
                # no pydicom available or other io error; leave meta as-is
                pass

    out = []
    for i, b in enumerate(dets):
        x1, y1, x2, y2 = [float(x) for x in b]
        bbox_px = [x1, y1, x2, y2]
        mm = bbox_to_mm(bbox_px, meta)
        # severity from the larger dimension
        size_mm = None
        if mm:
            size_mm = max(mm)
        severity = map_severity(size_mm)
        out.append({'bbox_px': bbox_px, 'score': float(scores[i]) if len(scores) > i else None, 'label': int(labels[i]) if len(labels) > i else None, 'size_mm': size_mm, 'severity': severity})

    print({'detections': out})


if __name__ == '__main__':
    main()
