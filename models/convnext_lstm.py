import torch
import torch.nn as nn
from torchvision.models.convnext import convnext_tiny

class ConvNeXtLSTM(nn.Module):
    """
    Mô hình theo bài báo 2025:
    - ConvNeXt-Tiny backbone trích xuất đặc trưng không gian
    - LSTM 1 layer để nâm bắt mối quan hệ chuỗi
    - FC phân loại apnea / non-apnea
    """
    def __init__(self, lstm_hidden=128, num_classes=2):
        super().__init__()

        # ConvNeXt backbone pretrained
        self.backbone = convnext_tiny(weights="IMAGENET1K_V1")
        self.backbone.classifier = nn.Identity()  # bỏ lớp FC cuối

        # LSTM sau ConvNeXt (768 đầu ra)
        self.lstm = nn.LSTM(input_size=768, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True)

        # Fully connected cho classification
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # x: (B, 1, 64, 684) - Mel-spectrum

        x = x.repeat(1, 3, 1, 1)  # (B, 3, 64, 684)
        feat = self.backbone(x)  # (B, 768)

        # Bổ sung chiều thời gian giả (vì LSTM yêu cầu chuỗi)
        feat = feat.unsqueeze(1)  # (B, 1, 768)

        lstm_out, _ = self.lstm(feat)  # (B, 1, hidden)
        out = self.fc(lstm_out[:, -1])  # lấy đầu ra bước cuối

        return out
