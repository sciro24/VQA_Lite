# Progetto VQA Lite

Questo è un semplice progetto di Visual Question Answering (VQA) addestrato sul dataset CIFAR-10. Il modello prende in input un'immagine e una domanda testuale e predice una risposta.

## Struttura del Progetto

```
/VQA lite/
├── src/                  # Codice sorgente
│   ├── dataset.py          # Classe PyTorch Dataset
│   ├── evaluate.py         # Script per valutare il modello
│   ├── inference.py        # Script per l'inferenza su singola immagine
│   ├── prepare_dataset.py  # Script per pre-calcolare gli embedding
│   ├── train.py            # Script di addestramento
│   └── vqa_model.py        # Architettura del modello
├── data/                   # Dati (generati automaticamente)
├── config.yaml             # File di configurazione centrale
├── requirements.txt        # Dipendenze Python
└── README.md               # Questo file
```

## Installazione

1.  Assicurati di avere Python 3.8+ installato.
2.  Crea un ambiente virtuale (consigliato):
    ```bash
    python3 -m venv .venv
    ```
3.  **Attiva l'ambiente virtuale**. Questo passo è fondamentale e va ripetuto per ogni nuova sessione del terminale.
    ```bash
    source .venv/bin/activate
    ```
    (Dovresti vedere `(.venv)` all'inizio del prompt).

4.  Installa le dipendenze usando `python -m pip` per garantire di usare il pip corretto:
    ```bash
    python -m pip install -r requirements.txt
    # (Opzionale) Verifica supporto MPS su Mac:
    python - << 'PY'
    import torch
    print('MPS available:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(), 'built:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_built())
    PY
    ```

## Esecuzione

Segui questi passi in ordine. Gli script devono essere eseguiti dalla directory principale del progetto.

1.  **Preparazione del Dataset**

    Questo script scarica CIFAR-10 e pre-calcola gli embedding delle domande. Va eseguito solo una volta.
    ```bash
    python src/prepare_dataset.py
    ```

2.  **Addestramento del Modello**

    Questo script addestra il modello VQA e salva il best checkpoint in `vqa_model_best.pth`.
    ```bash
    python src/train.py
    ```

3.  **Valutazione del Modello**

    Questo script calcola l'accuratezza del modello sul test set.
    ```bash
    python src/evaluate.py
    ```

4.  **Inferenza**

    Usa questo script per testare il modello su una tua immagine e con una tua domanda.
    ```bash
    python src/inference.py --image_path /percorso/della/tua/immagine.jpg --question "C'è un cane?"
    ```
    Nota: Il flag `--visualize` non è più supportato.
