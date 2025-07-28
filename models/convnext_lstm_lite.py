import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.convnext_lite import ConvNeXtBackboneLite, GlobalAvgPooling, count_parameters

import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.convnext_lite import ConvNeXtBackboneLite, GlobalAvgPooling, count_parameters

class ConvNeXtZ_LSTMLiteSequence(nn.Module):
    def __init__(self, num_classes=2, embed_dim=256, lstm_hidden=128, lstm_layers=1, dropout=0.3):
        super().__init__()

        dims = [128, 256, 512]
        self.backbone = ConvNeXtBackboneLite(in_channels=1, dims=dims)
        self.global_pool = GlobalAvgPooling()
        self.projection = nn.Linear(self.backbone.out_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.projection_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        lstm_out_dim = lstm_hidden * 2  # bidirectional
        self.fc_dropout1 = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(lstm_out_dim, 192)
        self.fc_dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(192, num_classes)

        print(f"[✔] Tổng số tham số: {count_parameters(self)/1e6:.2f}M")

    def forward(self, x):
        B, T = x.shape[:2]
        x = x.view(B * T, 1, 64, 684)
        features = self.backbone(x)
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)

        embed = self.projection(pooled)
        embed = self.norm(embed)
        embed = self.projection_dropout(embed)
        embed = embed.view(B, T, -1)

        lstm_out, _ = self.lstm(embed)
        out = self.fc_dropout1(lstm_out)
        out = self.fc_hidden(out)
        out = torch.relu(out)
        out = self.fc_dropout2(out)
        out = self.fc(out)
        return out
