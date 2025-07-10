import torch
import torch.nn as nn
from .convnext_lite import ConvNeXtBackboneLite, GlobalAvgPooling, count_parameters

class TransformerEncoderLayerLite(nn.Module):
    """
    Phiên bản nhẹ của TransformerEncoder với ít tham số hơn và hiệu quả cao hơn
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # Thêm Gated Linear Unit (GLU) cho feed-forward network để cải thiện khả năng biểu diễn
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        # Thêm Pre-Normalization để ổn định huấn luyện
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Pre-Normalization với residual connection
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_output)
        
        # Pre-Normalization với residual connection cho feed-forward
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + ff_output
        
        return x

class FocalAttention(nn.Module):
    """
    Focal Attention để tập trung vào các vùng quan trọng của input
    """
    def __init__(self, in_features, reduction=8):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(in_features, in_features // reduction),
            nn.ReLU(),
            nn.Linear(in_features // reduction, in_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (B, T, C) hoặc (B, C)
        if len(x.shape) == 3:
            attention = self.channel_attention(x.mean(dim=1))  # (B, C)
            return x * attention.unsqueeze(1)
        else:
            attention = self.channel_attention(x)  # (B, C)
            return x * attention

class ConvNeXtTransformerLite(nn.Module):
    """
    Phiên bản cải tiến của ConvNeXT-Transformer với hiệu suất cao hơn
    """
    def __init__(self, num_classes=2, embed_dim=128, num_heads=8, 
                 num_transformer_layers=4, dropout=0.3, dropout_path=0.1):
        super().__init__()
        
        # Sử dụng ConvNeXtBackboneLite với stochastic depth (dropout path)
        self.backbone = ConvNeXtBackboneLite(in_channels=1, dropout_path=dropout_path)
        
        # Global Average Pooling
        self.global_pool = GlobalAvgPooling()
        
        # BatchNorm sau Global Pooling
        self.bn = nn.BatchNorm1d(256)
        
        # Focal Attention để tập trung vào các feature quan trọng
        self.focal_attention = FocalAttention(256)
        
        # Đặt embedding dimension lớn hơn để nắm bắt thông tin tốt hơn
        self.projection = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),  # Thêm LayerNorm trước GELU
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Position Embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Tăng số lượng layers của transformer
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerLite(embed_dim, num_heads, embed_dim*4, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # FC layer với dropout cao hơn để giảm overfitting
        self.dropout = nn.Dropout(dropout)
        
        # Thêm lớp trung gian trước lớp phân loại cuối cùng
        self.intermediate = nn.Linear(embed_dim, embed_dim // 2)
        self.final_activation = nn.GELU()
        self.final_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim // 2, num_classes)
        
        # Khởi tạo trọng số tốt hơn
        self._initialize_weights()
        
        # Print số lượng tham số
        print(f"Tổng số tham số: {count_parameters(self)/1e6:.2f}M")
    
    def _initialize_weights(self):
        # Khởi tạo position embedding
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # Khởi tạo lớp projection
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Khởi tạo lớp trung gian và phân loại
        nn.init.trunc_normal_(self.intermediate.weight, std=0.02)
        nn.init.zeros_(self.intermediate.bias)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)
        
    def forward(self, x):
        # Kiểm tra nếu x là chuỗi thời gian (sequence) từ LazyApneaSequenceDataset
        # shape: (B, seq_len, channels, H, W)
        if len(x.shape) == 5:
            batch_size, seq_len, channels, height, width = x.shape
            
            # Xử lý từng frame trong chuỗi
            sequence_features = []
            
            # Reshape để xử lý từng frame một
            x_reshaped = x.view(batch_size * seq_len, channels, height, width)
            
            # Trích xuất đặc trưng với backbone nhẹ hơn
            features = self.backbone(x_reshaped)  # (B*seq_len, 256, H', W')
            
            # Global pooling
            pooled = self.global_pool(features)  # (B*seq_len, 256)
            
            # BatchNorm - cần reshape do batch dimension đã thay đổi
            pooled = self.bn(pooled)
            
            # Apply focal attention
            pooled = self.focal_attention(pooled)
            
            # Reshape lại để lấy lại chiều seq_len
            pooled = pooled.view(batch_size, seq_len, -1)  # (B, seq_len, 256)
            
            # Project xuống chiều embed_dim với activation và dropout
            embed = self.projection(pooled)  # (B, seq_len, embed_dim)
            
            # Thêm position embedding
            embed = embed + self.pos_embedding
            embed = self.norm(embed)
            
        else:
            # x shape: (B, 1, 64, 684) - Mel-spectrogram (dạng thông thường)
            
            # Trích xuất đặc trưng với backbone nhẹ hơn
            features = self.backbone(x)  # (B, 256, H', W')
            
            # Global pooling
            pooled = self.global_pool(features)  # (B, 256)
            
            # BatchNorm
            pooled = self.bn(pooled)
            
            # Apply focal attention
            pooled = self.focal_attention(pooled)
            
            # Project xuống chiều embed_dim với activation và dropout
            embed = self.projection(pooled)  # (B, embed_dim)
            
            # Thêm chiều token và position embedding
            embed = embed.unsqueeze(1)  # (B, 1, embed_dim)
            embed = embed + self.pos_embedding
            embed = self.norm(embed)
        
        # Đưa qua các lớp Transformer (số lượng nhiều hơn)
        for layer in self.transformer_layers:
            embed = layer(embed)
        
        # Áp dụng dropout trước khi phân loại
        embed = self.dropout(embed.squeeze(1))
        
        # Đưa qua lớp trung gian
        intermediate = self.intermediate(embed)
        intermediate = self.final_activation(intermediate)
        intermediate = self.final_dropout(intermediate)
        
        # Classification
        output = self.fc(intermediate)
        
        return output
