from pathlib import Path
import torch
from tqdm import tqdm

from common.config import cfg, DEVICE
from models import VQANet
from section3_model import build_model


def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, q, y in tqdm(loader, desc="🔄 Valutazione", leave=False, ncols=100):
            images, q, y = images.to(device), q.to(device), y.to(device)
            out = model(images, q)
            pred = out.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return 100.0 * correct / max(1, total)


def main():
    model = build_model()
    model_save_path = cfg['paths']['model_save_path']
    epochs = cfg['training']['epochs']

    if Path(model_save_path).exists():
        print(f"✅ Modello pre-addestrato trovato in: {model_save_path}")
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
            print("✅ Pesi caricati correttamente. Salto il training.")
            run_training = False
        except Exception as e:
            print(f"🔴 Errore nel caricamento del modello pre-addestrato: {e}")
            print("   Procedo con il riaddestramento.")
            run_training = True
    else:
        print(f"⚠️ Modello pre-addestrato non trovato in {model_save_path}. Inizio training...")
        run_training = True

    if run_training:
        from section5_dataloaders import train_loader, val_loader
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg['training']['learning_rate'],
            weight_decay=cfg['training'].get('weight_decay', 0.0)
        )
        best_acc = 0.0
        print("\n--- Inizio Training Effettivo ---")
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", ncols=100)
            for images, q, y in train_pbar:
                images, q, y = images.to(DEVICE), q.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out = model(images, q)
                loss = criterion(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.item()
                train_pbar.set_postfix(loss=loss.item())
            val_acc = evaluate(model, val_loader, DEVICE)
            avg_loss = running_loss / len(train_loader)
            print(f"\n[Epoch {epoch}/{epochs}] Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")
            if val_acc > best_acc:
                best_acc = val_acc
                try:
                    torch.save(model.state_dict(), model_save_path)
                    print(f"💾 Modello salvato! Nuova best acc: {best_acc:.2f}% -> {model_save_path}")
                except Exception as e:
                    print(f"⚠️ Errore salvataggio modello: {e}")

    # expose trained/loaded model for next sections
    global model_g
    model_g = model

    # auto-run next
    import section7_eval as next_section
    next_section.main()


if __name__ == '__main__':
    main()
