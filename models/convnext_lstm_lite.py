import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.convnext_lite import ConvNeXtBackboneLite, GlobalAvgPooling, count_parameters

class ConvNeXtZ_LSTMLiteSequence(nn.Module):
    def __init__(self, num_classes=2, embed_dim=128, lstm_hidden=256, lstm_layers=2, dropout=0.5):
        super().__init__()

        embed_dim = 120
        lstm_hidden = 256
        dims = [64, 120, 256]

        self.backbone = ConvNeXtBackboneLite(in_channels=1, dims=dims)
        self.global_pool = GlobalAvgPooling() 
        self.projection = nn.Linear(self.backbone.out_dim, embed_dim) 
        self.norm = nn.LayerNorm(embed_dim)

        # Thêm dropout sau projection
        self.projection_dropout = nn.Dropout(dropout * 0.3)

        # GRU 2 tầng, kích thước lớn với dropout
        self.lstm = nn.GRU(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout * 0.4 if lstm_layers > 1 else 0
        )

        # Thêm BatchNorm và nhiều dropout layers
        self.lstm_norm = nn.BatchNorm1d(512)
        self.fc_dropout1 = nn.Dropout(dropout * 0.6)
        self.fc_hidden = nn.Linear(512, 256)
        self.fc_dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout)

        print(f"[✔] Tổng số tham số: {count_parameters(self)/1e6:.2f}M")

    def forward(self, x):
        B, T = x.shape[:2]
        x = x.view(B * T, 1, 64, 684)
        features = self.backbone(x)
        pooled = self.global_pool(features)  # shape [B*T, C, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # flatten về [B*T, C]
        embed = self.projection(pooled)
        embed = self.norm(embed)
        embed = self.projection_dropout(embed)  # Dropout sau projection
        embed = embed.view(B, T, -1)
        
        lstm_out, _ = self.lstm(embed)
        
        # Reshape để áp dụng BatchNorm
        lstm_out_reshaped = lstm_out.reshape(-1, lstm_out.size(-1))  # [B*T, 512]
        lstm_out_norm = self.lstm_norm(lstm_out_reshaped)
        lstm_out_norm = lstm_out_norm.reshape(B, T, -1)  # [B, T, 512]
        
        # Thêm nhiều layer với dropout
        out = self.fc_dropout1(lstm_out_norm)
        out = self.fc_hidden(out)
        out = torch.relu(out)
        out = self.fc_dropout2(out)
        out = self.fc(out)
        return out
