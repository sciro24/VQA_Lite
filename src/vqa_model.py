# vqa_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VQANet(nn.Module):
    def __init__(self, num_answers, question_dim, image_feature_dim, attention_hidden_dim, dropout: float = 0.3):
        super().__init__()

        # 1) Backbone ResNet18 pre-addestrato su ImageNet (pesato) per feature spaziali 7x7
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Togli il classifier finale e mantieni fino a layer4
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # -> [B, 512, H/32, W/32]

        # Congela i primi layer per stabilità su piccolo dataset
        for name, param in self.backbone.named_parameters():
            if name.startswith("0") or name.startswith("1") or name.startswith("4"):  # stem e layer1
                param.requires_grad = False

        # Proiezione canali a image_feature_dim
        self.proj = nn.Conv2d(512, image_feature_dim, kernel_size=1)

        # 2) Attention spaziale condizionata dalla domanda
        self.attention_conv = nn.Conv2d(image_feature_dim + question_dim, attention_hidden_dim, 1)
        self.attention_fc = nn.Conv2d(attention_hidden_dim, 1, 1)

        # 3) Classificatore finale con dropout
        self.fc = nn.Sequential(
            nn.Linear(image_feature_dim + question_dim, attention_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(attention_hidden_dim, num_answers)
        )

    def forward(self, image, question_emb, temperature: float = 1.0):
        # --- Estrazione Feature Immagine ---
        # img_features -> [Batch, Canali, Altezza, Larghezza]
        x = self.backbone(image)
        img_features = self.proj(x)
        B, C, H, W = img_features.shape

        # --- Meccanismo di Attention ---
        # Ripeti l'embedding della domanda per ogni "pixel" della mappa di feature
        question_emb_expanded = question_emb.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
        
        # Concatena le feature dell'immagine e della domanda
        combined_features = torch.cat([img_features, question_emb_expanded], dim=1)
        
        # Calcola i pesi di attenzione
        attn_hidden = torch.tanh(self.attention_conv(combined_features))
        logits = self.attention_fc(attn_hidden).view(B, -1)
        logits = logits / max(temperature, 1e-6)
        # attn_weights -> [B, 1, H, W]
        attn_weights = F.softmax(logits, dim=1).view(B, 1, H, W)

        # Applica i pesi di attenzione per ottenere un vettore "attento" dell'immagine
        # (B, 1, H, W) * (B, C, H, W) -> (B, C, H, W)
        attended_img_features = attn_weights * img_features
        # Somma su altezza e larghezza per ottenere un singolo vettore per immagine
        # -> [B, C]
        attended_img_vector = attended_img_features.sum(dim=[2, 3])

        # --- Classificazione Finale ---
        # Combina il vettore attento e la domanda per la risposta finale
        final_combined = torch.cat([attended_img_vector, question_emb], dim=1)
        
        return self.fc(final_combined)