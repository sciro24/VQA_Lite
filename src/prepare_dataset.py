# prepare_dataset.py
import pickle
import os
import yaml
import argparse
from sentence_transformers import SentenceTransformer
from torchvision.datasets import CIFAR10
from tqdm import tqdm

# ================= CONFIGURAZIONE =================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Carica la configurazione
EMBEDDING_MODEL_NAME = config['embedding_model']
ANSWER_VOCAB = config['answers']
CATEGORIES = config['categories']
TRAIN_PATH = config['paths']['train_dataset_path']
TEST_PATH = config['paths']['test_dataset_path']

# Mappa inversa per le risposte (da stringa a indice)
answer_to_idx = {answer: i for i, answer in enumerate(ANSWER_VOCAB)}

# Nomi delle classi CIFAR per le domande
cifar10_classes = ["aeroplano", "automobile", "uccello", "gatto", "cervo", "cane", "rana", "cavallo", "nave", "camion"]

# ================= FUNZIONE DI CREAZIONE DATASET =================
def create_dataset(dataset, description, max_items: int = 0):
    """Genera una lista di (immagine, embedding_domanda, indice_risposta)."""
    data_list = []
    
    # Definisci le categorie inverse per un accesso rapido
    label_to_category = {}
    for cat, labels in CATEGORIES.items():
        for label in labels:
            label_to_category[label] = cat

    count = 0
    for img, label in tqdm(dataset, desc=f"Generando {description}"):
        true_class_name = cifar10_classes[label]
        true_category = label_to_category.get(label)
        
        # 1. Domanda sulla categoria corretta (es. "C'è un animale?")
        if true_category:
            q1_text = f"C'è un {true_category}?"
            a1_idx = answer_to_idx["Sì"]
            data_list.append({"image": img, "question": q1_text, "answer": a1_idx})

        # 2. Domanda sulla categoria errata
        other_category = 'veicolo' if true_category == 'animale' else 'animale'
        q2_text = f"C'è un {other_category}?"
        a2_idx = answer_to_idx["No"]
        data_list.append({"image": img, "question": q2_text, "answer": a2_idx})

        # 3. Domanda sull'identità dell'oggetto ("Che oggetto c'è?")
        q3_text = "Che oggetto c'è nell'immagine?"
        a3_idx = answer_to_idx[true_class_name]
        data_list.append({"image": img, "question": q3_text, "answer": a3_idx})

        count += 1
        if max_items and count >= max_items:
            break

    return data_list

# ================= SCRIPT PRINCIPALE =================
def main(args):
    # --- Carica dataset e modello di embedding ---
    print("🔄 Caricamento di CIFAR-10 e del modello di embedding...")
    train_set = CIFAR10(root=DATA_DIR, train=True, download=True)
    test_set = CIFAR10(root=DATA_DIR, train=False, download=True)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

    # --- Crea i dataset V2 ---
    train_data = create_dataset(train_set, "Training Set", max_items=int(args.max_train) if args.max_train else 0)
    test_data = create_dataset(test_set, "Test Set", max_items=int(args.max_test) if args.max_test else 0)

    # --- Calcola gli embedding per tutte le domande generate ---
    print("⚙️  Calcolo degli embedding per le domande (potrebbe richiedere tempo)...")
    train_questions = [item['question'] for item in train_data]
    test_questions = [item['question'] for item in test_data]
    
    train_embeddings = embedding_model.encode(
        train_questions, convert_to_tensor=True, show_progress_bar=True, batch_size=int(args.encode_batch_size)
    )
    test_embeddings = embedding_model.encode(
        test_questions, convert_to_tensor=True, show_progress_bar=True, batch_size=int(args.encode_batch_size)
    )

    # --- Finalizza e salva i dataset ---
    final_train_dataset = []
    for i, item in enumerate(train_data):
        final_train_dataset.append({
            "image": item['image'],
            "question_emb": train_embeddings[i],
            "answer": item['answer']
        })

    final_test_dataset = []
    for i, item in enumerate(test_data):
        final_test_dataset.append({
            "image": item['image'],
            "question_emb": test_embeddings[i],
            "answer": item['answer']
        })

    print(f"💾 Salvataggio dei dataset in '{TRAIN_PATH}' e '{TEST_PATH}'...")
    with open(TRAIN_PATH, 'wb') as f:
        pickle.dump(final_train_dataset, f)
    with open(TEST_PATH, 'wb') as f:
        pickle.dump(final_test_dataset, f)

    print("✅ Dataset V2 creati con successo!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset VQA da CIFAR-10 con domande generate.")
    parser.add_argument("--max_train", type=int, default=0, help="Numero massimo di immagini train da processare (0=all)")
    parser.add_argument("--max_test", type=int, default=0, help="Numero massimo di immagini test da processare (0=all)")
    parser.add_argument("--encode_batch_size", type=int, default=64, help="Batch size per SentenceTransformer.encode")
    args = parser.parse_args()
    main(args)
