from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer

from common.config import cfg, DEVICE
from common.utils import get_image_transform
from models import VQANet
from section8_infer import format_answer


class VQAModelWrapper(nn.Module):
    def __init__(self, model: VQANet, question_embedding: torch.Tensor):
        super().__init__()
        self.model = model
        self.q_emb = question_embedding.clone().detach().to(DEVICE)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        q = self.q_emb.expand(b, -1)
        return self.model(x, q)


def get_vanilla_saliency(model_wrapper, input_tensor, target_class_idx):
    # BatchNorm fails with batch_size=1, so we duplicate the input
    model_wrapper.eval()
    input_tensor_dup = torch.cat([input_tensor, input_tensor], dim=0)
    input_tensor_copy = input_tensor_dup.clone().detach().requires_grad_(True)
    model_wrapper.zero_grad()
    output = model_wrapper(input_tensor_copy)
    score = output[0, target_class_idx]  # Take first output only
    score.backward()
    # Take gradient from first input only
    saliency = input_tensor_copy.grad.data[0:1].abs()
    saliency, _ = torch.max(saliency, dim=1)
    saliency = saliency.squeeze(0).cpu().numpy()
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-8)
    return saliency


def main():
    # Load model
    m_cfg = cfg['model']
    num_answers = len(cfg['answers'])
    model = VQANet(
        num_answers, m_cfg['question_dim'], m_cfg['image_feature_dim'],
        m_cfg['attention_hidden_dim'], dropout=m_cfg.get('dropout', 0.3)
    ).to(DEVICE)

    model_path = cfg['paths']['model_save_path']
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✅ Modello caricato per saliency: {model_path}")
    except Exception as e:
        print(f"🔴 Errore caricamento modello per saliency: {e}")
        return

    # Get image and question from section 8
    try:
        import section8_infer as sec8
        USER_IMAGE_PATH = sec8.USER_IMAGE_PATH
        user_question_2 = sec8.USER_QUESTION
    except (AttributeError, ModuleNotFoundError):
        # Fallback if section8 not run or variables not set
        USER_IMAGE_PATH = 'data/test_image7.jpg'
        user_question_2 = "Che oggetto c'è'?"

    if not Path(USER_IMAGE_PATH).exists():
        print("🔴 Immagine non trovata per saliency, termina.")
        return

    rgb_img = Image.open(USER_IMAGE_PATH).convert('RGB')
    transform = get_image_transform(is_training=False)
    input_tensor = transform(rgb_img).unsqueeze(0).to(DEVICE)
    vis_img = np.array(rgb_img.resize((256, 256))) / 255.0

    try:
        embedding_model = SentenceTransformer(cfg['embedding_model'], device='cpu')
    except Exception as e:
        print(f"🔴 ERRORE: Modello embedding non caricato: {e}")
        return

    q_emb = embedding_model.encode(user_question_2, convert_to_tensor=True, normalize_embeddings=False)
    if q_emb.dim() == 1:
        q_emb = q_emb.unsqueeze(0)
    q_emb = q_emb.to(DEVICE).float()

    wrapped_model = VQAModelWrapper(model, q_emb).to(DEVICE)
    wrapped_model.eval()
    wrapped_model.model.eval()

    with torch.no_grad():
        output = wrapped_model(input_tensor)
        pred_idx = output.argmax(1).item()
        formatted = format_answer(user_question_2, pred_idx)

    saliency_map = get_vanilla_saliency(wrapped_model, input_tensor, pred_idx)

    print("\n--- Analisi Vanilla Saliency Map ---")
    print(f"Immagine: {USER_IMAGE_PATH}")
    print(f"Domanda: {user_question_2}")
    print(f"Risposta Predetta: {formatted}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(vis_img)
    plt.title("Immagine (256x256)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(saliency_map, cmap='hot')
    plt.title("Vanilla Saliency")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
