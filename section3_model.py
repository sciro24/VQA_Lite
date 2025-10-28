from common.config import cfg, DEVICE
from models import VQANet


def build_model():
    m_cfg = cfg['model']
    num_answers = len(cfg['answers'])
    model = VQANet(
        num_answers,
        m_cfg['question_dim'],
        m_cfg['image_feature_dim'],
        m_cfg['attention_hidden_dim'],
        dropout=m_cfg.get('dropout', 0.3)
    ).to(DEVICE)
    return model


def main():
    # Build once and store in module global for following sections via import
    global model
    model = build_model()
    print("✅ Modello VQANet costruito.")

    # auto-run next
    import section4_data as next_section
    next_section.main()


if __name__ == '__main__':
    main()
