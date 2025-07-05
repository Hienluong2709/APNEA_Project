import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

class ConvNeXtLSTM(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=256, lstm_layers=1):
        super(ConvNeXtLSTM, self).__init__()

        # ConvNeXt backbone (không dùng classifier gốc)
        self.backbone = convnext_tiny(weights=None)  # để nhanh, bỏ pretrain
        self.backbone.classifier = nn.Identity()     # bỏ FC cuối
        self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM (phải dùng cho chuỗi block)
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, seq_len, 1, 64, 684)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        # Lặp 3 channel
        x = x.repeat(1, 3, 1, 1)  # (B*T, 3, 64, 684)

        # ConvNeXt
        feat = self.backbone(x)              # (B*T, 768, 1, 1)
        feat = feat.view(B, T, -1)           # (B, T, 768)

        lstm_out, _ = self.lstm(feat)        # (B, T, hidden_dim)
        out = self.fc(lstm_out[:, -1])       # (B, num_classes)

        return out
