"""Train a YOLOv8 model on the prepared dataset using ultralytics.

Requires `ultralytics` in your environment (already listed in requirements.txt).

Usage (quick):
  py -3 scripts/train_yolo.py --data yolo_dataset --epochs 5 --img 640

This will create a small data YAML and call the ultralytics API.
"""

import argparse
from pathlib import Path
import yaml


def build_yaml(data_root: Path, out_yaml: Path):
    # data_root expected to contain train/val with images/labels
    # use relative paths inside the data root so ultralytics resolves them correctly
    d = {
        'names': {0: 'lesion'},
        'nc': 1,
        'train': str(Path('train') / 'images').replace('\\', '/'),
        'val': str(Path('val') / 'images').replace('\\', '/')
    }
    with open(out_yaml, 'w') as fh:
        yaml.safe_dump(d, fh)
    return out_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='yolo_dataset')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--img', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--weights', default='yolov8s.pt')
    parser.add_argument('--out', default='models/yolov8s_prostate')
    args = parser.parse_args()

    data_root = Path(args.data)
    if not data_root.exists():
        raise FileNotFoundError(f'{data_root} not found — run scripts/prepare_yolo.py first')

    out_yaml = data_root / 'data.yaml'
    build_yaml(data_root, out_yaml)

    # import ultralytics lazily so the script can still be parsed without package
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError('ultralytics not installed; install via requirements.txt') from e

    model = YOLO(args.weights)
    model.train(data=str(out_yaml), epochs=args.epochs, imgsz=args.img, batch=args.batch, project=args.out)


if __name__ == '__main__':
    main()
