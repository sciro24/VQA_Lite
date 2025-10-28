# evaluate.py
import torch
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm

from vqa_model import VQANet
from dataset import VQADataset
from .utils import get_device

# ================= CONFIGURAZIONE =================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = get_device(config)

# Carica parametri
params = config['training']
model_params = config['model']
paths = config['paths']

# ================= SCRIPT DI VALUTAZIONE =================
def evaluate():
    print(f"🚀 Inizio valutazione sul dispositivo: {DEVICE}")

    # --- Controlli preliminari ---
    if not os.path.exists(paths['model_save_path']) or not os.path.exists(paths['test_dataset_path']):
        print("❌ Errore: File del modello o del dataset non trovati. Esegui prima prepare e train.")
        return

    # --- Carica Test Set ---
    test_dataset = VQADataset(paths['test_dataset_path'])
    test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    # --- Carica Modello ---
    model = VQANet(
        num_answers=len(config['answers']),
        question_dim=model_params['question_dim'],
        image_feature_dim=model_params['image_feature_dim'],
        attention_hidden_dim=model_params['attention_hidden_dim'],
        dropout=model_params.get('dropout', 0.3)
    ).to(DEVICE)
    model.load_state_dict(torch.load(paths['model_save_path'], map_location=DEVICE))
    model.eval()

    # --- Loop di Valutazione ---
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, questions, answers in tqdm(test_dataloader, desc="📊 Valutazione"):
            images, questions, answers = images.to(DEVICE), questions.to(DEVICE), answers.to(DEVICE)

            outputs = model(images, questions)
            _, predicted = torch.max(outputs.data, 1)

            total_samples += answers.size(0)
            correct_predictions += (predicted == answers).sum().item()

    accuracy = 100 * correct_predictions / total_samples
    print(f"\n--- Risultato Valutazione ---")
    print(f"🎯 Accuratezza sul test set: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate()