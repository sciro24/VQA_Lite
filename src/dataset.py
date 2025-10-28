# src/dataset.py
from torch.utils.data import Dataset
import pickle
import torchvision.transforms as transforms
import torch
from .utils import get_image_transform

class VQADataset(Dataset):
    def __init__(self, dataset_path):
        """Carica un dataset VQA pre-processato da un file pickle."""
        with open(dataset_path, "rb") as f:
            self.data = pickle.load(f)
        
        # Normalizzazione coerente con ResNet
        self.transform = get_image_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = self.transform(item['image']).float()
        question_emb = item['question_emb']
        if isinstance(question_emb, torch.Tensor):
            # Assicura CPU e dtype coerente anche se il pickle conteneva tensor su MPS/CUDA
            if question_emb.device.type != 'cpu':
                question_emb = question_emb.to('cpu')
            question_emb = question_emb.float()
        return image, question_emb, item['answer']