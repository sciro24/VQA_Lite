# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import os
from tqdm import tqdm
from .utils import get_device, set_seed

from vqa_model import VQANet
from dataset import VQADataset

# ================= CONFIGURAZIONE =================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = get_device(config)

# Carica parametri dal config
params = config['training']
model_params = config['model']
paths = config['paths']

set_seed(params.get('seed', 42))

# ================= CARICA DATASET =================
print("🔄 Caricamento del dataset...")
if not os.path.exists(paths['train_dataset_path']):
    print(f"❌ Errore: Dataset non trovato. Esegui prima 'src/prepare_dataset.py'.")
    exit()

train_dataset = VQADataset(paths['train_dataset_path'])

# Split train/val
val_split = float(params.get('val_split', 0.1))
val_size = int(len(train_dataset) * val_split)
train_size = len(train_dataset) - val_size
train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

num_workers = int(params.get('num_workers', 0))
dataloader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False, num_workers=num_workers)

# ================= INIZIALIZZA MODELLO =================
print(f"🚀 Inizio addestramento su {DEVICE} per {params['epochs']} epoche...")
model = VQANet(
    num_answers=len(config['answers']),
    question_dim=model_params['question_dim'],
    image_feature_dim=model_params['image_feature_dim'],
    attention_hidden_dim=model_params['attention_hidden_dim'],
    dropout=model_params.get('dropout', 0.3)
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params.get('weight_decay', 0.0))

# Scheduler
scheduler = None
if params.get('scheduler', 'none') == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'])

# ================= TRAINING LOOP =================
best_val_acc = 0.0
epochs_no_improve = 0
patience = max(3, params['epochs'] // 4)

use_autocast = (DEVICE == 'mps')

for epoch in range(params['epochs']):
    model.train()
    running_loss = 0.0
    for images, questions, answers in tqdm(dataloader, desc=f"Epoch {epoch+1}/{params['epochs']}"):
        images, questions, answers = images.to(DEVICE), questions.to(DEVICE), answers.to(DEVICE)

        optimizer.zero_grad()
        if use_autocast:
            with torch.autocast(device_type='mps', dtype=torch.float16):
                outputs = model(images, questions)
                loss = criterion(outputs, answers)
        else:
            outputs = model(images, questions)
            loss = criterion(outputs, answers)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / max(1, len(dataloader))

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, questions, answers in val_loader:
            images, questions, answers = images.to(DEVICE), questions.to(DEVICE), answers.to(DEVICE)
            if use_autocast:
                with torch.autocast(device_type='mps', dtype=torch.float16):
                    outputs = model(images, questions)
            else:
                outputs = model(images, questions)
            _, predicted = torch.max(outputs.data, 1)
            total += answers.size(0)
            correct += (predicted == answers).sum().item()
    val_acc = 100.0 * correct / max(1, total)

    if scheduler is not None:
        scheduler.step()

    print(f"Epoch {epoch+1}/{params['epochs']} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.2f}%")

    # Early stopping + checkpointing
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), paths['model_save_path'])
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"⏹️  Early stopping: nessun miglioramento per {patience} epoche.")
            break

# ================= SALVA MODELLO =================
print(f"✅ Training terminato. Best model salvato in '{paths['model_save_path']}' (Val Acc: {best_val_acc:.2f}%)")
