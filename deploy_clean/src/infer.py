import argparse
from pathlib import Path
import torch

from src.prostate_dataset import ProstateLesionDataset
from src.train import make_model
from src import config


def infer_image(model, image_tensor, device="cpu"):
    model.eval()
    with torch.no_grad():
        xb = image_tensor.unsqueeze(0).to(device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
    return prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--sequences", default=None, help="comma separated sequence keywords (e.g. t2,adc)")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--toy-len", type=int, default=10)
    parser.add_argument("--num-slices", type=int, default=1, help="odd number of slices to stack (e.g. 3)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu"

    

    if args.toy:
        ds = ProstateLesionDataset(toy=True, toy_len=args.toy_len)
        from torch.utils.data import DataLoader
        # use same collate_fn as training so PIL images are converted to tensors
        from src.train import make_model, collate_fn
        dl = DataLoader(ds, batch_size=1, collate_fn=collate_fn)
        in_ch = ds.num_channels
        model = make_model(num_classes=2, pretrained=False, in_channels=in_ch)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        for i, (img, label) in enumerate(dl):
            # convert to tensor (if PIL) in train collate style
            if isinstance(img[0], torch.Tensor):
                t = img[0]
            else:
                from src.utils_image import pil_to_tensor
                t = pil_to_tensor(img[0], img_size=config.IMG_SIZE)
            prob = infer_image(model, t, device=device)
            print(f"example {i} label={label.item()} prob={prob:.3f}")
    else:
        seqs = None if args.sequences is None else [s.strip().lower() for s in args.sequences.split(",") if s.strip()]
        ds = ProstateLesionDataset(csv_path=args.csv, sequences=seqs, num_slices=args.num_slices)
        # ensure evaluation transform
        try:
            from src.utils_image import get_eval_tensor_transform
            ds.transform = get_eval_tensor_transform(img_size=config.IMG_SIZE)
        except Exception:
            ds.transform = None
        from torch.utils.data import DataLoader
        from src.train import collate_fn
        dl = DataLoader(ds, batch_size=1, collate_fn=collate_fn)
        in_ch = ds.num_channels
        model = make_model(num_classes=2, pretrained=False, in_channels=in_ch)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        for i, (img, label) in enumerate(dl):
            if isinstance(img, torch.Tensor):
                t = img[0]
            else:
                from src.utils_image import pil_to_tensor
                t = pil_to_tensor(img, img_size=config.IMG_SIZE)
            prob = infer_image(model, t, device=device)
            print(f"example {i} label={label} prob={prob:.3f}")


if __name__ == "__main__":
    main()
