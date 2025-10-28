import sys
import os
from pathlib import Path
import torch

IN_COLAB = 'google.colab' in sys.modules

# Local save directory
BASE_SAVE_DIR = Path('.')

CIFAR10_CLASSES = [
    "aeroplano", "automobile", "uccello", "gatto", "cervo",
    "cane", "rana", "cavallo", "nave", "camion"
]

cfg = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model': {
        'question_dim': 384,
        'image_feature_dim': 256,
        'attention_hidden_dim': 128,
        'dropout': 0.3,
    },
    'answers': CIFAR10_CLASSES,
    'categories': {'animale':[2,3,4,5,6,7],'veicolo':[0,1,8,9]},
    'training': {
        'batch_size': 256,
        'epochs': 10,
        'learning_rate': 5e-3,
        'weight_decay': 1e-4,
        'val_split': 0.1,
        'seed': 42,
        'num_workers': 4,
    },
    'paths': {
        'model_save_path': str(BASE_SAVE_DIR / 'vqa_model_best.pth'),
        'data_dir': 'data',
        'train_npz': 'data/train_dataset_full.npz',
        'test_npz': 'data/test_dataset_full.npz'
    },
    'embedding_model': 'all-MiniLM-L6-v2'
}

DEVICE = cfg['device']
DATA_DIR = cfg['paths']['data_dir']
Path(DATA_DIR).mkdir(exist_ok=True)
