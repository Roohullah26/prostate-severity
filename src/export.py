import argparse
from pathlib import Path
import torch

from src.train import make_model
from src import config


def export_model(state_dict_path, out_name=None, in_channels=3, device='cpu'):
    p = Path(state_dict_path)
    if not p.exists():
        raise FileNotFoundError(p)

    out_base = Path(config.MODELS_DIR) / (out_name or p.stem)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    # create model and load state
    model = make_model(num_classes=2, pretrained=False, in_channels=in_channels)
    sd = torch.load(p, map_location=device)
    try:
        model.load_state_dict(sd)
    except Exception:
        # sd may already be a scripted module or saved differently
        # try direct torch.load and assume it's a full Module
        obj = torch.load(p, map_location=device)
        if isinstance(obj, dict):
            raise
        else:
            model = obj

    model = model.to(device)
    model.eval()

    # TorchScript
    ts_path = out_base.with_suffix('.pt')
    try:
        example = torch.randn(1, in_channels, config.IMG_SIZE[0], config.IMG_SIZE[1]).to(device)
        scripted = torch.jit.trace(model, example)
        scripted.save(ts_path)
        print('Saved TorchScript ->', ts_path)
    except Exception as e:
        print('TorchScript export failed:', e)

    # ONNX
    onnx_path = out_base.with_suffix('.onnx')
    try:
        example = torch.randn(1, in_channels, config.IMG_SIZE[0], config.IMG_SIZE[1]).to(device)
        torch.onnx.export(model, example, onnx_path, opset_version=12)
        print('Saved ONNX ->', onnx_path)
    except Exception as e:
        print('ONNX export failed:', e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--out-name', default=None)
    parser.add_argument('--in-ch', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    export_model(args.model_path, out_name=args.out_name, in_channels=args.in_ch, device=args.device)


if __name__ == '__main__':
    main()
