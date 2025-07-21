import torch
import torch.nn as nn
import os
import sys

# Thêm đường dẫn chứa models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.convnext_lite import ConvNeXtZ, GlobalAvgPooling, count_parameters


class FocalAttention(nn.Module):
    def __init__(self, in_features, reduction=8):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(in_features, in_features // reduction),
            nn.ReLU(),
            nn.Linear(in_features // reduction, in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x.shape) == 3:
            attention = self.channel_attention(x.mean(dim=1))  # (B, C)
            return x * attention.unsqueeze(1)
        else:
            attention = self.channel_attention(x)  # (B, C)
            return x * attention


class ConvNeXtZ_Sequence(nn.Module):
    """
    ConvNeXt_Z nhẹ kết hợp FocalAttention và xử lý chuỗi (sequence).
    Tổng số tham số ~2.5M
    """
    def __init__(self, num_classes=2, embed_dim=8, dropout=0.1, dropout_path=0.1):
        super().__init__()
        self.backbone = ConvNeXtZ(
            in_channels=1,
            depths=[2, 2, 4, 2],
            dims=[24, 48, 96, 192],
            dropout_path=dropout_path
        )
        self.global_pool = GlobalAvgPooling()
        self.focal_attention = FocalAttention(192)

        self.projection = nn.Sequential(
            nn.Linear(192, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.fc = nn.Linear(embed_dim, num_classes)
        self._initialize_weights()

        print(f"[ConvNeXtZ_SequenceLite] Tổng số tham số: {count_parameters(self)/1e6:.2f}M")

    def _initialize_weights(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x: (B, T, 1, 64, 684)
        if len(x.shape) == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)

            features = self.backbone(x)              # (B*T, 192, h, w)
            pooled = self.global_pool(features)      # (B*T, 192)
            attended = self.focal_attention(pooled)  # (B*T, 192)
            embedded = self.projection(attended)     # (B*T, embed_dim)
            embedded = embedded.view(B, T, -1)       # (B, T, embed_dim)

            aggregated = embedded.mean(dim=1)        # (B, embed_dim)
            output = self.fc(aggregated)             # (B, num_classes)

        else:
            # (B, 1, H, W)
            features = self.backbone(x)
            pooled = self.global_pool(features)
            attended = self.focal_attention(pooled)
            embedded = self.projection(attended)
            output = self.fc(embedded)

        return output
