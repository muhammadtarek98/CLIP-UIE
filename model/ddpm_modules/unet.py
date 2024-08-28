import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# model


class TimeEmbedding(nn.Module):
    def __init__(self, dim:int):
        super(TimeEmbedding,self).__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer(name="inv_freq", tensor=inv_freq)
    def forward(self, input:torch.Tensor)->torch.Tensor:
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim:int):
        super(Upsample,self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim:int):
        super(Downsample,self).__init__()
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.conv(x)
# building block modules


class Block(nn.Module):
    def __init__(self, dim:int, dim_out:int, groups:int=32, dropout:float=0.0):
        super(Block,self).__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=dim),
            Swish(),
            nn.Dropout(p=dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(in_channels=dim, out_channels=dim_out, kernel_size=3, padding=1)
        )
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim:int, dim_out:int, time_emb_dim=None, dropout:float=0.0, norm_groups:int=32):
        super(ResnetBlock,self).__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(in_features=time_emb_dim,out_features=dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim=dim, dim_out=dim_out, groups=norm_groups)
        self.block2 = Block(dim=dim_out,dim_out= dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            in_channels=dim, out_channels=dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def forward(self, x:torch.Tensor, time_emb:torch.Tensor)->torch.Tensor:
        h = self.block1(x)
        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel:int, n_head:int=1, norm_groups:int=32):
        super(SelfAttention,self).__init__()
        self.n_head = n_head
        self.norm = nn.GroupNorm(num_groups=norm_groups, num_channels=in_channel)
        self.qkv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel * 3, kernel_size=1, bias=False)
        self.out = nn.Conv2d(in_channels=in_channel,out_channels= in_channel,kernel_size=1)
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
    def __init__(self, dim:int, dim_out:int, *, time_emb_dim=None, norm_groups:int=32, dropout:float=0.0, with_attn:bool=False):
        super(ResnetBlocWithAttn,self).__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)
    def forward(self, x:torch.Tensor, time_emb:tor)->torch.Tensor:
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
        dropout:float=0,
        with_time_emb:bool=True,
        image_size:int=128
    ):
        super(UNet,self).__init__()
        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(dim=inner_channel),
                nn.Linear(in_features=inner_channel,out_features= inner_channel * 4),
                Swish(),
                nn.Linear(in_features=inner_channel * 4, out_features=inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None
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
                    dim=pre_channel, dim_out=channel_mult, time_emb_dim=time_dim, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(dim=pre_channel,dim_out= pre_channel, time_emb_dim=time_dim, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(dim=pre_channel, dim_out=pre_channel, time_emb_dim=time_dim, norm_groups=norm_groups,
                                dropout=dropout, with_attn=False)
        ])
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    dim=pre_channel+feat_channels.pop(),dim_out=channel_mult, time_emb_dim=time_dim, dropout=dropout, norm_groups=norm_groups, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(dim=pre_channel))
                now_res = now_res*2
        self.ups = nn.ModuleList(ups)
        self.final_conv = Block(dim=pre_channel,dim_out=default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x:torch.Tensor, time:torch.Tensor)->torch.Tensor:
        t = self.time_mlp(time) if exists(self.time_mlp) else None
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
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)
        return self.final_conv(x)
x=torch.randn((1,3,720,720))
model=UNet()