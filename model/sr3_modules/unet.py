import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

from UWEnhancement.tools.output_featuremap import model


def exists(x):
    return x is not None
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d



# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim:int):
        super(PositionalEncoding,self).__init__()
        self.dim = dim
    def forward(self, noise_level:torch.Tensor)->torch.Tensor :
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            tensors=[torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, use_affine_level:bool=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(nn.Linear(in_features=in_channels,out_features=out_channels*(1+self.use_affine_level)))
    def forward(self, x:torch.Tensor, noise_embed)->torch.Tensor:
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x

class Swish(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return x * torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, dim:int):
        super(Upsample,self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x=self.up(x)
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim:int):
        super(Downsample,self).__init__()
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.conv(x)

class Block(nn.Module):
    def __init__(self, dim:int, dim_out:int, groups:int=32, dropout:float=0):
        super(Block,self).__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=groups,num_channels= dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(in_channels=dim, out_channels=dim_out, kernel_size=3, padding=1))
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.block(x)
class ResnetBlock(nn.Module):
    def __init__(self, dim:int, dim_out:int, noise_level_emb_dim=None, dropout:float=0.0, use_affine_level:bool=False, norm_groups:int=32):
        super(ResnetBlock,self).__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(in_channels=dim, out_channels=dim_out, kernel_size=1) if dim != dim_out else nn.Identity()
    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)
class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super(SelfAttention,self).__init__()
        self.n_head = n_head
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel * 3,kernel_size= 1, bias=False)
        self.out = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1)
    def forward(self, input:torch.Tensor)->torch.Tensor:
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)
        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))
        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim:int, dim_out:int, *, noise_level_emb_dim=None, norm_groups:int=32, dropout:float=0.0, with_attn:bool=False):
        super(ResnetBlocWithAttn,self).__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(dim=dim, dim_out=dim_out,noise_level_emb_dim=noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(in_channel=dim_out, norm_groups=norm_groups)
    def forward(self, x:torch.Tensor, time_emb)->torch.Tensor:
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class UNet(nn.Module):
    def __init__(
        self,
        in_channel:int=6,
        out_channel:int=3,
        inner_channel:int=32,
        norm_groups:int=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks:int=3,
        dropout:float=0.0,
        with_noise_level_emb:bool=True,
        image_size:int=128
    ):
        super(UNet,self).__init__()
        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(dim=inner_channel),
                nn.Linear(in_features=inner_channel,out_features=inner_channel * 4),
                Swish(),
                nn.Linear(in_features=inner_channel * 4,out_features=inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channels=in_channel,out_channels=inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    dim=pre_channel, dim_out=channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(dim=pre_channel,dim_out= pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(dim=pre_channel,dim_out= pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    dim=ipre_channel+feat_channels.pop(),dim_out=channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(dim=pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)
        self.final_conv = Block(dim=pre_channel,dim_out=default(out_channel, in_channel), groups=norm_groups)
    def forward(self, x:torch.Tensor, time)->torch.Tensor:
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat(tensors=(x, feats.pop()), dim=1), t)
            else:
                x = layer(x)
        return self.final_conv(x)
x=torch.randn(size=(1,3,1080,1080))
model=UNet()