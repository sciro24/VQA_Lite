# inference.py
import torch
import argparse
import os
import yaml
from vqa_model import VQANet
from PIL import Image
from torchvision import transforms
from sentence_transformers import SentenceTransformer
from .utils import get_device, get_image_transform

# ================= CONFIGURAZIONE =================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = get_device(config)

# Carica parametri
model_params = config['model']
paths = config['paths']
answers_vocab = config['answers']
embedding_model_name = config['embedding_model']

# ================= UTILS PER INFERENZA =================
transform = get_image_transform()

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

def preprocess_question(question, model):
    q = model.encode(question, convert_to_tensor=True)
    if q.dim() == 1:
        q = q.unsqueeze(0)
    return q

# ================= SCRIPT PRINCIPALE =================
def main(args):
    print(f"🚀 Esecuzione inferenza su {DEVICE}")

    # --- Controlli preliminari ---
    if not os.path.exists(args.image_path):
        print(f"❌ Errore: File immagine non trovato in '{args.image_path}'")
        return
    if not os.path.exists(paths['model_save_path']):
        print(f"❌ Errore: Modello non trovato. Esegui prima 'src/train.py'.")
        return

    # --- Carica modelli ---
    print("🔄 Caricamento dei modelli...")
    embedding_model = SentenceTransformer(embedding_model_name, device="cpu")
    model = VQANet(
        num_answers=len(answers_vocab),
        question_dim=model_params['question_dim'],
        image_feature_dim=model_params['image_feature_dim'],
        attention_hidden_dim=model_params['attention_hidden_dim'],
        dropout=model_params.get('dropout', 0.3)
    ).to(DEVICE)
    model.load_state_dict(torch.load(paths['model_save_path'], map_location=DEVICE))
    model.eval()

    # --- Preprocess ---
    print("🖼️  Elaborazione dell'immagine e della domanda...")
    img_tensor = preprocess_image(args.image_path).to(DEVICE)
    question_emb = preprocess_question(args.question, embedding_model).to(DEVICE)

    # --- Inferenza ---
    with torch.no_grad():
        output = model(img_tensor, question_emb)
        predicted_idx = torch.argmax(output, dim=1).item()
        predicted_answer = answers_vocab[predicted_idx]

    # --- Stampa risultato ---
    print("\n--- Risultato ---")
    print(f"❓ Domanda: '{args.question}'")
    print(f"💡 Risposta: {predicted_answer}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegui l'inferenza VQA su un'immagine.")
    parser.add_argument("--image_path", type=str, required=True, help="Percorso del file immagine.")
    parser.add_argument("--question", type=str, required=True, help="Domanda da porre (es. 'C\'è un animale?', 'Che oggetto c\'è?').")
    args = parser.parse_args()
    main(args)