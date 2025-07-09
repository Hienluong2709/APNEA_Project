import torch
import torch.nn as nn

class ConvNeXtNano(nn.Module):
    """
    Phiên bản ConvNeXT thu nhỏ với ít tham số hơn nhiều (~2M)
    """
    def __init__(self, in_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        
        # Stage 1: Downsampling đầu tiên
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(features[0]),
            nn.GELU()
        )
        
        # Stage 2-4: ConvNeXT blocks với downsampling
        self.stage2 = self._make_stage(features[0], features[1])
        self.stage3 = self._make_stage(features[1], features[2])
        self.stage4 = self._make_stage(features[2], features[3])
        
    def _make_stage(self, in_channels, out_channels):
        return nn.Sequential(
            # Downsampling
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            
            # ConvNeXT block (simplified)
            nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels*2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1),
        )
    
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

class ConvNeXtBackboneLite(nn.Module):
    """
    Phiên bản nhẹ của ConvNeXT backbone (~2M parameters)
    """
    def __init__(self, in_channels=1):
        super().__init__()
        # Sử dụng ConvNeXtNano thay vì ConvNeXT Tiny
        self.features = ConvNeXtNano(in_channels=in_channels)
        
    def forward(self, x):
        return self.features(x)

# Lớp Global Average Pooling không thay đổi
class GlobalAvgPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        pooled = self.pool(x)
        return pooled.flatten(1)

# Hàm để đếm số tham số của model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
