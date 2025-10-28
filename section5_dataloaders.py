from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import torch

from torchvision.datasets import CIFAR10
from common.utils import get_image_transform
from common.config import cfg, DEVICE


class VQADatasetNPZ(Dataset):
    def __init__(self, npz_path: str, cifar_dataset: CIFAR10, is_training: bool):
        data = np.load(npz_path)
        self.indices = data['indices']
        self.emb = data['emb']
        self.y = data['y']
        self.cifar = cifar_dataset
        self.transform = get_image_transform(is_training=is_training)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img_idx = int(self.indices[i])
        img, _ = self.cifar[img_idx]
        image = self.transform(img).float()
        q = torch.from_numpy(self.emb[i].astype(np.float32))
        y = int(self.y[i])
        return image, q, y


def main():
    # Import CIFAR datasets produced in previous section
    from section4_data import train_set_cifar_g, test_set_cifar_g

    train_ds_full = VQADatasetNPZ(cfg['paths']['train_npz'], cifar_dataset=train_set_cifar_g, is_training=True)
    test_ds = VQADatasetNPZ(cfg['paths']['test_npz'], cifar_dataset=test_set_cifar_g, is_training=False)

    val_split = cfg['training']['val_split']
    val_size = max(1, int(len(train_ds_full) * val_split))
    train_size = len(train_ds_full) - val_size
    train_ds, val_ds = random_split(train_ds_full, [train_size, val_size])
    val_ds.dataset.transform = get_image_transform(is_training=False)

    bs = cfg['training']['batch_size']
    nw = cfg['training']['num_workers']
    pin = DEVICE == 'cuda'

    global train_loader, val_loader, test_loader
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin, persistent_workers=True if nw > 0 else False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=True if nw > 0 else False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=True if nw > 0 else False)

    print(f"Dataset: Train={train_size}, Val={val_size}, Test={len(test_ds)}")
    print(f"Dataloader: Batch Size={bs}, Num Workers={nw}")

    # auto-run next
    import section6_train as next_section
    next_section.main()


if __name__ == '__main__':
    main()
