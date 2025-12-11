from torchvision import transforms
import torch
import random
from PIL import Image


def get_default_transforms(img_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def pil_to_tensor(img, img_size=(224, 224)):
    t = get_default_transforms(img_size)
    return t(img)


def get_train_tensor_transform(img_size=(224, 224), p_hflip=0.5, p_vflip=0.3, bright_jitter=0.1):
    """Return a callable that applies lightweight tensor-based augmentations to a CxHxW tensor.

    The function expects a float32 tensor in range [0,1]."""

    def fn(t):
        # t: torch.Tensor CxHxW, values 0..1
        if not isinstance(t, torch.Tensor):
            # convert from PIL/np to tensor
            t = pil_to_tensor(t, img_size=img_size)

        # random horizontal flip
        if random.random() < p_hflip:
            t = torch.flip(t, dims=[2])

        # random vertical flip
        if random.random() < p_vflip:
            t = torch.flip(t, dims=[1])

        # brightness jitter
        if bright_jitter and bright_jitter > 0:
            factor = 1.0 + (random.random() * 2 - 1) * bright_jitter
            t = t * float(factor)
            t = torch.clamp(t, 0.0, 1.0)

        # normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        # expand mean/std to match number of channels
        c = t.shape[0]
        if c != 3:
            # repeat first mean/std across channels
            mean = mean[0].repeat(c)
            std = std[0].repeat(c)

        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        t = (t - mean) / std
        return t

    return fn


def get_eval_tensor_transform(img_size=(224, 224)):
    def fn(t):
        if not isinstance(t, torch.Tensor):
            t = pil_to_tensor(t, img_size=img_size)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        c = t.shape[0]
        if c != 3:
            mean = mean[0].repeat(c)
            std = std[0].repeat(c)
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        t = (t - mean) / std
        return t

    return fn
