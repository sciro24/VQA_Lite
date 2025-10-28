import random
import numpy as np
import torch
from torchvision import transforms


def get_device(cfg) -> str:
    if cfg.get('device', 'auto') == 'auto':
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() and torch.backends.mps.is_available()
        return 'mps' if has_mps else 'cpu'
    return cfg['device']


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.mps.manual_seed non sempre presente; manual_seed basta per determinismo di base


def get_image_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


