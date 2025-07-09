import torch
import torch.nn as nn
from .convnext_lite import ConvNeXtBackboneLite, GlobalAvgPooling, count_parameters

class TransformerEncoderLayerLite(nn.Module):
    """
    Phiên bản nhẹ của TransformerEncoder với ít tham số hơn
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention với residual connection
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward với residual connection
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.norm2(x)
        
        return x

class ConvNeXtTransformerLite(nn.Module):
    """
    Phiên bản nhẹ của ConvNeXT-Transformer với ~2M parameters
    """
    def __init__(self, num_classes=2, embed_dim=64, num_heads=2, 
                 num_transformer_layers=2, dropout=0.1):
        super().__init__()
        
        # Sử dụng ConvNeXtBackboneLite
        self.backbone = ConvNeXtBackboneLite(in_channels=1)
        
        # Global Average Pooling
        self.global_pool = GlobalAvgPooling()
        
        # Đặt embedding dimension nhỏ hơn
        self.projection = nn.Linear(256, embed_dim)  # 256 là số channel đầu ra của ConvNeXtNano
        
        # LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Position Embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Giảm số lượng layers của transformer
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerLite(embed_dim, num_heads, embed_dim*2, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # FC layer
        self.fc = nn.Linear(embed_dim, num_classes)
        
        # Print số lượng tham số
        print(f"Tổng số tham số: {count_parameters(self)/1e6:.2f}M")
        
    def forward(self, x):
        # x shape: (B, 1, 64, 684) - Mel-spectrogram
        
        # Trích xuất đặc trưng với backbone nhẹ hơn
        features = self.backbone(x)  # (B, 256, H', W')
        
        # Global pooling
        pooled = self.global_pool(features)  # (B, 256)
        
        # Project xuống chiều nhỏ hơn
        embed = self.projection(pooled)  # (B, embed_dim)
        
        # Thêm chiều token và position embedding
        embed = embed.unsqueeze(1)  # (B, 1, embed_dim)
        embed = embed + self.pos_embedding
        embed = self.norm(embed)
        
        # Đưa qua các lớp Transformer (số lượng ít hơn)
        for layer in self.transformer_layers:
            embed = layer(embed)
        
        # Classification
        output = self.fc(embed.squeeze(1))
        
        return output
