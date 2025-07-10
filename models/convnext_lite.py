import torch
import torch.nn as nn
import torch.nn.functional as F

# Dropout path (stochastic depth) cho regularization tốt hơn
class DropPath(nn.Module):
    """
    Dropout path: Bỏ qua kết nối trong residual block với xác suất cho trước
    """
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class LayerNorm2d(nn.Module):
    """
    Layer Normalization cho tensor 2D (B, C, H, W)
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXT block cải tiến với DropPath và Layer Normalization
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        x = self.norm(x)
        
        # Permute từ [B, C, H, W] sang [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Permute trở lại [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        x = input_x + self.drop_path(x)
        return x

class ConvNeXtNano(nn.Module):
    """
    Phiên bản ConvNeXT thu nhỏ với ít tham số hơn nhiều (~2M)
    """
    def __init__(self, in_channels=1, features=[32, 64, 128, 256], dropout_path=0.1):
        super().__init__()
        self.dropout_path = dropout_path
        
        # Stage 1: Downsampling đầu tiên với patchify
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=4),
            LayerNorm2d(features[0])
        )
        
        # Stage 2-4: ConvNeXT blocks với downsampling
        self.stage1 = self._make_stage(features[0], features[1], 2, dropout_path)
        self.stage2 = self._make_stage(features[1], features[2], 2, dropout_path)
        self.stage3 = self._make_stage(features[2], features[3], 2, dropout_path)
        
    def _make_stage(self, in_channels, out_channels, num_blocks=2, dropout_path=0.1):
        # Tăng dần dropout_path theo độ sâu của mạng
        dpr = [x.item() for x in torch.linspace(0, dropout_path, num_blocks)]
        
        blocks = []
        # Downsampling layer
        blocks.append(nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        ))
        
        # ConvNeXT blocks
        for i in range(num_blocks):
            blocks.append(ConvNeXtBlock(out_channels, drop_path=dpr[i]))
        
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x

class ConvNeXtBackboneLite(nn.Module):
    """
    Phiên bản nhẹ của ConvNeXT backbone (~2M parameters) với stochastic depth
    """
    def __init__(self, in_channels=1, dropout_path=0.1):
        super().__init__()
        # Sử dụng ConvNeXtNano thay vì ConvNeXT Tiny
        self.features = ConvNeXtNano(in_channels=in_channels, dropout_path=dropout_path)
        
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
