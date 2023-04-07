import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, scale=None, attn_drop=0, proj_drop=0, ratio=1):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.ratio = ratio
        head_dim = dim // num_heads
        self.scale = scale or head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=bias)
        self.kv = nn.Linear(dim, dim * 2, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=ratio, stride=ratio)
            self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
            
        