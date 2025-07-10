# models/convnext_lstm_sequence.py
import torch
import torch.nn as nn
from .convnext_lite import ConvNeXtBackboneLite, GlobalAvgPooling, count_parameters

class ConvNeXtLSTMLiteSequence(nn.Module):
    """
    ConvNeXt-LSTM xử lý chuỗi đầu vào (sequence)
    Input: (B, T, 1, 64, 684)
    Output: (B, T, num_classes)
    """
    def __init__(self, num_classes=2, embed_dim=64, lstm_hidden=128, lstm_layers=1, dropout=0.1):
        super().__init__()
        self.backbone = ConvNeXtBackboneLite(in_channels=1)
        self.global_pool = GlobalAvgPooling()
        self.projection = nn.Linear(256, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        print(f"Tổng số tham số: {count_parameters(self)/1e6:.2f}M")

    def forward(self, x):
        # x: (B, T, 1, 64, 684)
        B, T = x.shape[:2]
        x = x.view(B * T, 1, 64, 684)  # Ghép batch + time
        features = self.backbone(x)                # (B*T, 256, H, W)
        pooled = self.global_pool(features)        # (B*T, 256)
        embed = self.projection(pooled)            # (B*T, embed_dim)
        embed = self.norm(embed)
        embed = embed.view(B, T, -1)               # (B, T, embed_dim)

        lstm_out, _ = self.lstm(embed)             # (B, T, H*2)
        out = self.fc(self.dropout(lstm_out))      # (B, T, num_classes)
        return out
