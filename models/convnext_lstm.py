import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

class ConvNeXtLSTM(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=256, lstm_layers=1):
        super(ConvNeXtLSTM, self).__init__()

        # ConvNeXt backbone (không dùng classifier gốc)
        self.backbone = convnext_tiny(weights=None)  # nếu muốn tải pretrain: weights='DEFAULT'
        self.backbone.classifier = nn.Identity()     # bỏ phần phân loại gốc
        self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # đảm bảo đầu ra luôn (B, C, 1, 1)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=768,  # output của convnext_tiny
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Classifier cuối
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape ban đầu: (B, 1, 64, 684) — 1 channel, 64 mel bands, 684 time frames

        if x.dim() == 5:
            x = x.squeeze(1)  # (B, 1, 64, 684) → (B, 64, 684)

        # Đảm bảo x có shape (B, 1, 64, 684)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Lặp lại channel để thành ảnh 3 kênh (B, 3, 64, 684)
        x = x.repeat(1, 3, 1, 1)

        # Trích đặc trưng từ ConvNeXt → (B, 768, 1, 1)
        feat = self.backbone(x)           # (B, 768, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (B, 768)

        # Đưa qua LSTM — phải thêm chiều seq_len = 1 → (B, 1, 768)
        feat = feat.unsqueeze(1)          # (B, 1, 768)
        lstm_out, _ = self.lstm(feat)     # (B, 1, hidden_dim)

        # Lấy output ở thời điểm cuối cùng (vì seq_len = 1 nên là [:, -1])
        out = self.fc(lstm_out[:, -1])    # (B, num_classes)

        return out
