import torch
import torch.nn as nn
import math

# Định nghĩa ConvNeXtBackbone nếu chưa có
class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXT backbone để trích xuất đặc trưng từ mel-spectrogram.
    Có thể tái sử dụng cho nhiều mô hình khác nhau (LSTM, Transformer, etc.)
    """
    def __init__(self, in_channels=1, pretrained=False):
        super().__init__()
        # Khởi tạo ConvNeXT Tiny từ torchvision
        try:
            from torchvision.models import convnext_tiny
            weights = "IMAGENET1K_V1" if pretrained else None
            self.model = convnext_tiny(weights=weights)
            
            # Thay đổi lớp đầu tiên để nhận 1 kênh thay vì 3 kênh
            original_conv = self.model.features[0][0]
            new_conv = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)
            
            # Khởi tạo trọng số cho 1 kênh bằng cách lấy trung bình 3 kênh nếu dùng pretrained
            if pretrained and in_channels != 3:
                with torch.no_grad():
                    new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
            
            self.model.features[0][0] = new_conv
            
            # Chỉ lấy phần features, bỏ classifier
            self.features = self.model.features
        except ImportError:
            # Fallback nếu không có torchvision
            print("⚠️ Không thể import convnext_tiny từ torchvision. Sử dụng CNN đơn giản thay thế.")
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                
                nn.Conv2d(512, 768, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(768),
                nn.ReLU(),
            )
        
    def forward(self, x):
        # x shape: (B, C, H, W) - với C thường là 1 cho mel-spectrogram
        features = self.features(x)  # (B, 768, H', W')
        return features

# Lớp Global Average Pooling để chuyển đổi đặc trưng thành vector
class GlobalAvgPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        # x shape: (B, C, H, W)
        pooled = self.pool(x)  # (B, C, 1, 1)
        return pooled.flatten(1)  # (B, C)

class TransformerEncoderLayer(nn.Module):
    """
    Một layer của Transformer Encoder gồm:
    - Multi-head self-attention
    - Feed-forward network
    - LayerNorm và residual connections
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (B, N, D) - N là số token, D là embed_dim
        
        # Self-attention với residual connection
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward với residual connection
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.norm2(x)
        
        return x

class ConvNeXtTransformer(nn.Module):
    """
    Mô hình ConvNeXT-Transformer kết hợp:
    - ConvNeXT để trích xuất đặc trưng không gian từ mel-spectrogram
    - Global Average Pooling để có vector đặc trưng
    - Transformer để nắm bắt mối quan hệ phức tạp giữa các đặc trưng
    - Fully connected layer để phân loại
    """
    def __init__(self, num_classes=2, pretrained=False, 
                 embed_dim=128, num_heads=4, 
                 num_transformer_layers=6, dropout=0.1):
        super().__init__()
        
        # ConvNeXT backbone để trích xuất đặc trưng
        self.backbone = ConvNeXtBackbone(in_channels=1, pretrained=pretrained)
        
        # Global Average Pooling
        self.global_pool = GlobalAvgPooling()
        
        # Linear projection từ 768 (ConvNeXT output) xuống embed_dim
        self.projection = nn.Linear(768, embed_dim)
        
        # LayerNorm trước khi đưa vào Transformer
        self.norm = nn.LayerNorm(embed_dim)
        
        # Position Embedding cho mỗi token
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Chỉ có 1 token sau global pooling
        
        # Transformer Encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, embed_dim*4, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # Fully connected cho classification
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # x shape: (B, 1, 64, 684) - Mel-spectrogram
        
        # Trích xuất đặc trưng bằng ConvNeXT
        features = self.backbone(x)  # (B, 768, H', W')
        
        # Global pooling để có vector đặc trưng
        pooled = self.global_pool(features)  # (B, 768)
        
        # Project xuống chiều embed_dim
        embed = self.projection(pooled)  # (B, embed_dim)
        
        # Thêm chiều token và cộng position embedding
        embed = embed.unsqueeze(1)  # (B, 1, embed_dim)
        embed = embed + self.pos_embedding
        embed = self.norm(embed)
        
        # Đưa qua các lớp Transformer
        for layer in self.transformer_layers:
            embed = layer(embed)
        
        # Lấy token đầu tiên (CLS token) và đưa qua FC layer
        output = self.fc(embed.squeeze(1))
        
        return output
