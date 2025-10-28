# Localizzazione File Importanti

## File NPZ (Dataset)
**Percorso cercato:**
- `data/train_dataset_full.npz` 
- `data/test_dataset_full.npz`

**Vengono cercati in:** `section4_data.py`

Se esistono, il dataset NON viene rigenerato. Se non esistono, vengono creati automaticamente.

## File Modello (Pesi Addestrati)
**Percorso cercato:**
- `./vqa_model_best.pth` (nella root del progetto)

**Vengono cercati in:**
- `section6_train.py` - Se esiste, salta il training
- `section7_eval.py` - Carica il modello per la valutazione
- `section8_infer.py` - Carica il modello per l'inferenza
- `section9_saliency.py` - Carica il modello per la saliency map

## Struttura Directory Finale
```
VQA_Lite/
├── vqa_model_best.pth          ← Modello addestrato (qui)
├── data/
│   ├── train_dataset_full.npz  ← Dataset train
│   ├── test_dataset_full.npz   ← Dataset test
│   └── test_image7.jpg          ← Immagine di esempio per inferenza
├── common/
├── section*.py
└── run_all.py
```

## Note
- I percorsi sono definiti in `common/config.py`
- Se vuoi cambiare dove vengono salvati/cercati i file, modifica `cfg['paths']` in `common/config.py`

