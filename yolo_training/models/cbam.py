"""
CBAM: Convolutional Block Attention Module
用于增强YOLO模型的小目标检测能力
参考论文: CBAM: Convolutional Block Attention Module (ECCV 2018)
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    """CBAM注意力模块 = 通道注意力 + 空间注意力
    兼容YOLO的c1参数（输入通道数）
    """
    
    def __init__(self, c1=None, c2=None, reduction=16, kernel_size=7):
        super().__init__()
        self.c1 = c1
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.channel_attention = None
        self.spatial_attention = SpatialAttention(kernel_size)
        if c1 is not None:
            self.channel_attention = ChannelAttention(c1, reduction)
    
    def forward(self, x):
        if self.channel_attention is None:
            c1 = x.shape[1]
            self.channel_attention = ChannelAttention(c1, self.reduction).to(x.device)
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class C2f_CBAM(nn.Module):
    """带CBAM的C2f模块，用于替换YOLO中的C2f"""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            for _ in range(n)
        )
        self.cbam = CBAM(c2)
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cbam(self.cv2(torch.cat(y, 1)))


class Conv(nn.Module):
    """标准卷积模块"""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """标准Bottleneck模块"""
    
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


def autopad(k, p=None):
    """自动计算padding"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SE(nn.Module):
    """SE (Squeeze-and-Excitation) 注意力模块"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ECA(nn.Module):
    """ECA (Efficient Channel Attention) 注意力模块"""
    
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        import math
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)
