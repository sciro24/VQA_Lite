from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer

from common.config import cfg, DEVICE
from common.utils import get_image_transform
from models import VQANet

ANSWERS_VOCAB = cfg['answers']


def format_answer(question: str, pred_idx: int) -> str:
    pred_class = ANSWERS_VOCAB[pred_idx]
    q = question.strip().lower()
    if q.startswith("c'è un ") or q.startswith("c'è una "):
        prefix_len = len("c'è un ") if q.startswith("c'è un ") else len("c'è una ")
        asked_class = q[prefix_len:-1].strip()
        return f"Sì, c'è un/una {pred_class}." if asked_class == pred_class else f"No, non c'è un/una {asked_class}. C'è un/una {pred_class}."
    if q.startswith("che "):
        return f"C'è un/una {pred_class}."
    return pred_class


def run_vqa_inference(image_path: str, question: str, model: VQANet, device: str):
    try:
        img = Image.open(image_path).convert('RGB')
        transform = get_image_transform(is_training=False)
        img_t = transform(img).unsqueeze(0).to(device).float()
    except FileNotFoundError:
        return f"ERRORE: Immagine non trovata: {image_path}", 0.0
    except Exception as e:
        return f"ERRORE caricamento/trasformazione immagine: {e}", 0.0

    try:
        embedding_model = SentenceTransformer(cfg['embedding_model'], device='cpu')
    except Exception as e:
        return f"ERRORE: Modello embedding non caricato: {e}", 0.0

    q_emb = embedding_model.encode(question, convert_to_tensor=True, normalize_embeddings=False)
    if q_emb.dim() == 1:
        q_emb = q_emb.unsqueeze(0)
    q_emb = q_emb.to(device).float()

    model.eval()
    with torch.no_grad():
        out = model(img_t, q_emb)
        probabilities = F.softmax(out, dim=1)
        pred_idx = out.argmax(1).item()
        confidence = probabilities[0, pred_idx].item() * 100.0

    formatted_answer = format_answer(question, pred_idx)
    return formatted_answer, confidence


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
        print(f"✅ Modello caricato per inferenza: {model_path}")
    except Exception as e:
        print(f"🔴 Errore caricamento modello per inferenza: {e}")
        # Continue to next section even if failed

    USER_IMAGE_PATH = 'data/test_image6.jpg'
    user_question_2 = "Che animale c'è'?"

    # Save image path and question for section 9
    import section8_infer as self_module
    self_module.USER_IMAGE_PATH = USER_IMAGE_PATH
    self_module.USER_QUESTION = user_question_2

    if Path(USER_IMAGE_PATH).exists():
        answer2, conf2 = run_vqa_inference(USER_IMAGE_PATH, user_question_2, model, DEVICE)
        print(f"\nDomanda 2: {user_question_2}")
        print(f"Risposta 2: {answer2} (Confidenza: {conf2:.2f}%)")
    else:
        print(f"🔴 Immagine non trovata: {USER_IMAGE_PATH}")

    # auto-run next
    import section9_saliency as next_section
    next_section.main()


if __name__ == '__main__':
    main()
