import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        
    def forward(self, x):
        return self.gamma[:, None, None] * x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.layer_scale = LayerScale(dim, layer_scale_init_value) if layer_scale_init_value > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.layer_scale(x)
        return shortcut + self.drop_path(x)

class ConvNeXtZ(nn.Module):
    def __init__(self, in_channels=1, depths=[2, 2, 6, 2], dims=[32, 64, 128, 256], dropout_path=0.1):
        super().__init__()
        self.dims = dims
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        self.stage1 = self._make_stage(dims[0], dims[1], depths[0], dropout_path)
        self.stage2 = self._make_stage(dims[1], dims[2], depths[1], dropout_path)
        self.stage3 = self._make_stage(dims[2], dims[3], depths[2], dropout_path)
        self.stage4 = self._make_stage(dims[3], dims[3], depths[3], dropout_path)
        self.out_dim = dims[-1]

    def _make_stage(self, in_channels, out_channels, num_blocks, dropout_path):
        dpr = [x.item() for x in torch.linspace(0, dropout_path, num_blocks)]
        blocks = []

        if in_channels != out_channels:
            blocks.append(nn.Sequential(
                LayerNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
            ))

        for i in range(num_blocks):
            blocks.append(ConvNeXtBlock(out_channels, drop_path=dpr[i]))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

class ConvNeXtBackboneLite(nn.Module):
    """
    Nhẹ hơn ConvNeXtZ, dùng để test nhanh hoặc với mô hình nhỏ hơn.
    Có thể chọn `dims` đầu ra để điều chỉnh độ nặng.
    """
    def __init__(self, in_channels=1, dims=[24, 48, 96]):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(dims[0]),
            nn.GELU()
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.GELU()
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.GELU()
        )
        self.out_dim = dims[2]

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x

class GlobalAvgPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(x).flatten(1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
