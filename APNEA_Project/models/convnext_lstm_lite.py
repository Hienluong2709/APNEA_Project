import torch
import torch.nn as nn
from .convnext_lite import ConvNeXtBackboneLite, GlobalAvgPooling, count_parameters

class ConvNeXtLSTMLite(nn.Module):
    """
    Phiên bản nhẹ của ConvNeXT-LSTM (~2M parameters)
    """
    def __init__(self, num_classes=2, embed_dim=64, lstm_hidden=128, 
                 lstm_layers=1, dropout=0.1):
        super().__init__()
        
        # Sử dụng ConvNeXtBackboneLite
        self.backbone = ConvNeXtBackboneLite(in_channels=1)
        
        # Global Average Pooling
        self.global_pool = GlobalAvgPooling()
        
        # Linear projection
        self.projection = nn.Linear(256, embed_dim)  # 256 là số channel đầu ra của ConvNeXtNano
        
        # LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        
        # LSTM nhỏ hơn
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # FC
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)  # *2 vì bidirectional
        self.dropout = nn.Dropout(dropout)
        
        # Print số lượng tham số
        print(f"Tổng số tham số: {count_parameters(self)/1e6:.2f}M")
        
    def forward(self, x):
        # x shape: (B, 1, 64, 684) - Mel-spectrogram
        
        # Trích xuất đặc trưng
        features = self.backbone(x)  # (B, 256, H', W')
        
        # Global pooling
        pooled = self.global_pool(features)  # (B, 256)
        
        # Project
        embed = self.projection(pooled)  # (B, embed_dim)
        embed = self.norm(embed)
        
        # LSTM
        embed = embed.unsqueeze(1)  # (B, 1, embed_dim)
        lstm_out, _ = self.lstm(embed)  # (B, 1, lstm_hidden*2)
        
        # Classification
        output = self.fc(self.dropout(lstm_out.squeeze(1)))
        
        return output
