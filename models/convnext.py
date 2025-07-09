import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXT backbone để trích xuất đặc trưng từ mel-spectrogram.
    Có thể tái sử dụng cho nhiều mô hình khác nhau (LSTM, Transformer, etc.)
    """
    def __init__(self, in_channels=1, pretrained=False):
        super().__init__()
        # Khởi tạo ConvNeXT Tiny từ torchvision
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
