# VQA_Lite

VQA_Lite è un progetto didattico di Visual Question Answering (VQA) costruito su CIFAR-10. Fornisce una pipeline leggera per: preparare dataset (con embedding delle domande), definire e addestrare un modello VQA semplice basato su ResNet18 + meccanismo di attenzione, eseguire inferenza su immagini locali e visualizzare mappe di salienza.

## Tecnologie utilizzate

- Python 3.x
- PyTorch (modello, allenamento, inferenza)
- torchvision (dataset CIFAR-10, trasformazioni, ResNet18)
- sentence-transformers (embedding delle domande)
- numpy, tqdm, matplotlib, PIL

## Struttura del notebook

- `CELLA 1`: Setup e configurazione (variabili globali, dipendenze, gestione Google Drive opzionale).
- `CELLA 2`: Trasformazioni immagine e utility (resize a 32x32 per inferenza; augmentazioni per training).
- `CELLA 3`: Definizione modello VQANet (ResNet18 + attent., classificatore).
- `CELLA 4`: Preparazione dataset e generazione NPZ con embedding delle domande (salvataggio/skipping se già presenti).
- `CELLA 5`: Classe Dataset e DataLoader (caricamento NPZ + CIFAR con trasformazioni corrette).
- `CELLA 6`: Training e validazione (salvataggio del migliore modello e skip se modello già presente).
- `CELLA 7`: Valutazione finale su test set.
- `CELLA 8`: Funzioni di inferenza (eseguire VQA su immagine locale + domanda).
- `CELLA 10`: Visualizzazione mappe di salienza (vanilla saliency).

> Nota: il notebook mantiene logica di skip per evitare rigenerazione dei NPZ e per caricare modelli salvati.

## Come usare

1. Installare le dipendenze (opzionale, il notebook può installare internamente):

```bash
pip install -r requirements.txt
# oppure installare i pacchetti principali:
# pip install sentence-transformers tqdm PyYAML grad-cam opencv-python-headless matplotlib
```

2. Aprire `VQA_Lite.ipynb` e seguire le celle in ordine. Le celle includono logiche per saltare operazioni costose se i file NPZ o il modello sono già presenti.

3. Per testare inferenza locale:
   - Copiare un'immagine in `./test_images/` o usare `DRIVE_SAVE_DIR / 'test_images'` se si usa Google Drive.
   - Aggiornare `USER_IMAGE_NAME` e la domanda nella cella di inferenza.
   - Eseguire la cella di inferenza.

4. Per generare saliency map, eseguire la cella dedicata dopo aver caricato il modello e fornito l'immagine.

## Suggerimenti

- Per uso interattivo, eseguire le celle in sequenza.
- In ambiente Colab, il notebook può montare Google Drive per salvataggio persistente.
- Per riproducibilità impostare il seed nella configurazione (`cfg['training']['seed']`).

## Contatti

Per domande o contributi aprire un issue o una pull request nel repository.
