from pathlib import Path
import numpy as np
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from sentence_transformers import SentenceTransformer

from common.config import cfg, DATA_DIR

ANSWER_VOCAB = cfg['answers']
answer_to_idx = {a: i for i, a in enumerate(ANSWER_VOCAB)}


def create_indexed_dataset_full(dataset, description):
    indices, questions, targets = [], [], []
    for idx, (_, label) in enumerate(tqdm(dataset, desc=f"Preparazione {description}", ncols=100)):
        true_class_name = ANSWER_VOCAB[label]
        indices.append(idx)
        questions.append(f"C'è un {true_class_name}?")
        targets.append(label)
        indices.append(idx)
        questions.append("Che oggetto c'è?")
        targets.append(label)
    return indices, questions, targets


def main():
    train_npz = cfg['paths']['train_npz']
    test_npz = cfg['paths']['test_npz']

    print("🔄 Caricamento CIFAR-10...")
    train_set_cifar = CIFAR10(root=DATA_DIR, train=True, download=True)
    test_set_cifar = CIFAR10(root=DATA_DIR, train=False, download=True)

    # Expose for next sections via module globals
    global train_set_cifar_g, test_set_cifar_g
    train_set_cifar_g, test_set_cifar_g = train_set_cifar, test_set_cifar

    embedding_model = None
    if not (Path(train_npz).exists() and Path(test_npz).exists()):
        print("⚠️ File NPZ non trovati. Rigenerazione in corso...")
        try:
            embedding_model = SentenceTransformer(cfg['embedding_model'], device='cpu')
        except Exception as e:
            print(f"🔴 Errore caricamento embedding_model: {e}")
            embedding_model = None
        if embedding_model is None:
            print("🔴 Impossibile generare NPZ: modello embedding non caricato.")
        else:
            train_indices, train_questions, train_targets = create_indexed_dataset_full(train_set_cifar, "Train FULL")
            test_indices, test_questions, test_targets = create_indexed_dataset_full(test_set_cifar, "Test FULL")
            print("🔄 Calcolo embedding domande (Train)...")
            train_emb = embedding_model.encode(train_questions, convert_to_numpy=True, show_progress_bar=True, batch_size=256, normalize_embeddings=False)
            print("🔄 Calcolo embedding domande (Test)...")
            test_emb = embedding_model.encode(test_questions, convert_to_numpy=True, show_progress_bar=True, batch_size=256, normalize_embeddings=False)
            train_emb = train_emb.astype(np.float16)
            test_emb = test_emb.astype(np.float16)
            train_indices = np.asarray(train_indices, dtype=np.int32)
            test_indices = np.asarray(test_indices, dtype=np.int32)
            train_targets = np.asarray(train_targets, dtype=np.int16)
            test_targets = np.asarray(test_targets, dtype=np.int16)
            Path(Path(train_npz).parent).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(train_npz, indices=train_indices, emb=train_emb, y=train_targets)
            np.savez_compressed(test_npz, indices=test_indices, emb=test_emb, y=test_targets)
            print(f"✅ Dataset salvati in {train_npz} e {test_npz}")
    else:
        print(f"✅ File NPZ trovati in '{DATA_DIR}'. Salto la rigenerazione.")

    # auto-run next
    import section5_dataloaders as next_section
    next_section.main()


if __name__ == '__main__':
    main()
