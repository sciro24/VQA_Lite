import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VQANet(nn.Module):
    def __init__(self, num_answers, question_dim, image_feature_dim, attention_hidden_dim, dropout: float = 0.3):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.proj = nn.Conv2d(512, image_feature_dim, kernel_size=1)
        self.attention_conv = nn.Conv2d(image_feature_dim + question_dim, attention_hidden_dim, 1)
        self.attention_fc = nn.Conv2d(attention_hidden_dim, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(image_feature_dim + question_dim, attention_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(attention_hidden_dim, num_answers)
        )

    def forward(self, image, question_emb, temperature: float = 1.0):
        x = self.backbone(image)
        img_features = self.proj(x)
        B, C, H, W = img_features.shape
        question_emb_expanded = question_emb.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
        combined_features = torch.cat([img_features, question_emb_expanded], dim=1)
        attn_hidden = torch.tanh(self.attention_conv(combined_features))
        logits = self.attention_fc(attn_hidden).view(B, -1)
        logits = logits / max(temperature, 1e-6)
        attn_weights = F.softmax(logits, dim=1).view(B, 1, H, W)
        attended_img_vector = (attn_weights * img_features).sum(dim=[2, 3])
        final_combined = torch.cat([attended_img_vector, question_emb], dim=1)
        return self.fc(final_combined)
