import torch
import torch.nn as nn
from .convnext_transformer import ConvNeXtBackbone, GlobalAvgPooling

class ConvNeXtLSTM(nn.Module):
    """
    Mô hình ConvNeXT-LSTM kết hợp:
    - ConvNeXT để trích xuất đặc trưng không gian từ mel-spectrogram
    - Global Average Pooling để có vector đặc trưng
    - LSTM để nắm bắt mối quan hệ tuần tự
    - Fully connected layer để phân loại
    """
    def __init__(self, num_classes=2, pretrained=False, 
                 embed_dim=128, lstm_hidden=256, 
                 lstm_layers=1, dropout=0.1):
        super().__init__()
        
        # ConvNeXT backbone để trích xuất đặc trưng (dùng chung với transformer)
        self.backbone = ConvNeXtBackbone(in_channels=1, pretrained=pretrained)
        
        # Global Average Pooling
        self.global_pool = GlobalAvgPooling()
        
        # Linear projection từ 768 (ConvNeXT output) xuống embed_dim
        self.projection = nn.Linear(768, embed_dim)
        
        # LayerNorm trước khi đưa vào LSTM
        self.norm = nn.LayerNorm(embed_dim)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected cho classification
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)  # *2 vì bidirectional
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (B, 1, 64, 684) - Mel-spectrogram
        
        # Trích xuất đặc trưng bằng ConvNeXT
        features = self.backbone(x)  # (B, 768, H', W')
        
        # Global pooling để có vector đặc trưng
        pooled = self.global_pool(features)  # (B, 768)
        
        # Project xuống chiều embed_dim
        embed = self.projection(pooled)  # (B, embed_dim)
        embed = self.norm(embed)
        
        # LSTM yêu cầu đầu vào là chuỗi, nên thêm chiều giả cho thời gian
        embed = embed.unsqueeze(1)  # (B, 1, embed_dim)
        
        # Đưa qua LSTM
        lstm_out, _ = self.lstm(embed)  # (B, 1, lstm_hidden*2)
        
        # Lấy output cuối cùng của LSTM và đưa qua FC layer
        output = self.fc(self.dropout(lstm_out.squeeze(1)))  # (B, num_classes)
        
        return output
