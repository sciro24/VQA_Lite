from pathlib import Path
import torch

from common.config import cfg, DEVICE
from models import VQANet
from section6_train import evaluate


def main():
    model_save_path = cfg['paths']['model_save_path']
    m_cfg = cfg['model']
    num_answers = len(cfg['answers'])
    model = VQANet(
        num_answers, m_cfg['question_dim'], m_cfg['image_feature_dim'],
        m_cfg['attention_hidden_dim'], dropout=m_cfg.get('dropout', 0.3)
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
        print(f"✅ Modello migliore caricato da: {model_save_path}")
        from section5_dataloaders import test_loader
        test_acc = evaluate(model, test_loader, DEVICE)
        print(f"\n📊 Accuracy Finale sul Test Set: {test_acc:.2f}%")
    except FileNotFoundError:
        print(f"🔴 ATTENZIONE: File modello non trovato in '{model_save_path}'.")
        print("   Eseguire il training.")
    except Exception as e:
        print(f"🔴 Errore caricamento modello: {e}")

    # expose model for next sections
    global model_g
    model_g = model

    # auto-run next
    import section8_infer as next_section
    next_section.main()


if __name__ == '__main__':
    main()
