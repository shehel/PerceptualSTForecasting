import torch
from typing import Optional
from einops import rearrange
from torch import nn
from openstl.modules import (ConvSC, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                           HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                           SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock)
from timm.models.layers import DropPath, trunc_normal_

import pdb
class UNetQ_Model(nn.Module):
    r"""SimVP Model

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(
        self, in_channels=4, out_ts=4, out_ch=4, depth=2, wf=6, padding=True, batch_norm=True, up_mode="upconv",
        pos_emb=False, **kwargs
    ):
        super(UNetQ_Model, self).__init__()
        assert up_mode in ("upconv", "upsample")
        self.padding = padding
        self.depth = depth
        self.in_channels = in_channels
        prev_channels = in_channels
        self.out_ts = out_ts
        self.out_ch = out_ch

        if pos_emb:
            self.pos_model = Date2Vec()
        else:
            self.pos_model = None
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm, time_emb_dim=6 if pos_emb else None))
            prev_channels = 2 ** (wf + i)

        # self.hid = MidMetaNet(256, 256, 4,
        #          input_resolution=(32, 32), model_type="convsc")

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm, time_emb_dim=6 if pos_emb else None))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, out_ts*out_ch, kernel_size=1)
        self.lower1 = nn.Conv2d(prev_channels, out_ts*out_ch, kernel_size=1)
        #self.lower2 = nn.Conv2d(prev_channels, out_ts*out_ch, kernel_size=1)
        self.upper1 = nn.Conv2d(prev_channels, out_ts*out_ch, kernel_size=1)
        #self.upper2 = nn.Conv2d(prev_channels, out_ts*out_ch, kernel_size=1)

    def forward(self, inputs, *args, **kwargs):
        x_raw = inputs[0]
        quantiles = inputs[1]
        quantiles = None
        B, _, _, H, W = x_raw.shape
        x = x_raw.reshape(-1, self.in_channels, H,W)

        t = self.pos_model(t) if exists(self.pos_model) else None
        blocks = []
        for i, down in enumerate(self.down_path):
            if i == 0:
                x = down(x, t)
            else:
                x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = torch.nn.functional.max_pool2d(x, 2)
        
        #x = self.hid(x)
        # copy translated to x
        translated = x
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        #x=self.last(x)
        x = torch.cat((self.lower1(x).unsqueeze(1), self.last(x).unsqueeze(1), self.upper1(x).unsqueeze(1)), dim=1)
        # add an empty dimension at first axis
        x = torch.unsqueeze(x, 2)
        x = x.reshape(B, 3, self.out_ts, self.out_ch, H, W)
        return x, translated

    def recon(self, x, *args, **kwargs):
        B, _, _, H, W = x.shape
        x = x.reshape(-1, self.in_channels, H,W)

        t = self.pos_model(t) if exists(self.pos_model) else None
        blocks = []
        for i, down in enumerate(self.down_path):
            if i == 0:
                x = down(x, t)
            else:
                x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = torch.nn.functional.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        x=self.last(x)
        # add an empty dimension at first axis
        x = torch.unsqueeze(x, 1)
        x = x.reshape(B, self.out_ts, self.out_ch, H, W)
        return x

    def encode(self, x):
        B, _, _, H, W = x.shape
        x = x.reshape(-1, self.in_channels, H, W)

        t = self.pos_model(t) if exists(self.pos_model) else None
        blocks = []
        for i, down in enumerate(self.down_path):
            if i == 0:
                x = down(x, t)
            else:
                x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = torch.nn.functional.max_pool2d(x, 2)
        
        return x

# helper functions
def exists(x):
    return x is not None

class Date2Vec(nn.Module):
    def __init__(self, k=32, act="sin"):
        super(Date2Vec, self).__init__()

        if k % 2 == 0:
            k1 = k // 2
            k2 = k // 2
        else:
            k1 = k // 2
            k2 = k // 2 + 1
        
        self.fc1 = nn.Linear(1, k1)

        self.fc2 = nn.Linear(1, k2)
        self.d2 = nn.Dropout(0.3)
 
        if act == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.cos

        self.fc3 = nn.Linear(k, k // 2)
        self.d3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(k // 2, 6)
        
        self.fc5 = torch.nn.Linear(6, 6)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.d2(self.activation(self.fc2(x)))
        out = torch.cat([out1, out2], 1)
        out = self.d3(self.fc3(out))
        out = self.fc4(out)
        out = self.fc5(out)
        return out

def hat_activation(p):
    zeros = torch.zeros_like(p, device=p.device)
    ones = torch.ones_like(p, device=p.device)
    twos = 2 * ones

    condition1 = (p >= 0) & (p < 1)
    condition2 = (p >= 1) & (p < 2)

    return torch.where(condition1, p, torch.where(condition2, twos - p, zeros))


    # Scaled Hat activation function
def scaled_hat_activation(alpha=100):
    def scaled_hat(p):
        return alpha * hat_activation(p)
    return scaled_hat

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)
        self.act = nn.ReLU()#scaled_hat_activation()
        self.norm = nn.BatchNorm2d(dim_out)
        #self.apply(self._init_weights)
#


    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, 
                time_emb_dim: Optional[int] = None,):
        super(UNetConvBlock, self).__init__()
        
        self.mlp = (nn.Sequential(
            nn.ReLU(), nn.Linear(time_emb_dim, out_size*2))
            if exists(time_emb_dim) else None)

        self.block1 = Block(in_size, out_size)
        self.block2 = Block(out_size, out_size)
        self.res_conv = nn.Conv2d(in_size, out_size, 1) if in_size != out_size else nn.Identity()


    def forward(self, x, time_emb=None):  # noqa
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        out = self.block1(x, scale_shift = scale_shift)
        out = self.block2(out)

        return out + self.res_conv(x)

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, time_emb_dim: Optional[int] = None,):
        super(UNetUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(nn.Upsample(mode="bilinear", scale_factor=2), nn.Conv2d(in_size, out_size, kernel_size=1),)

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm, time_emb_dim)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        diff_y_target_size_ = diff_y + target_size[0]
        diff_x_target_size_ = diff_x + target_size[1]
        return layer[:, :, diff_y:diff_y_target_size_, diff_x:diff_x_target_size_]

    def forward(self, x, bridge, time_emb=None):  # noqa
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        #crop1 = torch.zeros_like(crop1)
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out, time_emb)

        return out


 
class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        if model_type == 'gsta':
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'convmixer':
            self.block = ConvMixerSubBlock(in_channels, kernel_size=11, activation=nn.GELU)
        elif model_type == 'convsc':
            self.block = ConvSC(in_channels, in_channels, kernel_size=3)   
        elif model_type == 'convnext':
            self.block = ConvNeXtSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'hornet':
            self.block = HorNetSubBlock(in_channels, mlp_ratio=mlp_ratio, drop_path=drop_path)
        elif model_type == 'mlp':
            self.block = MLPMixerSubBlock(
                in_channels, input_resolution, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'moga':
            self.block = MogaSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop_rate=drop, drop_path_rate=drop_path)
        elif model_type == 'poolformer':
            self.block = PoolFormerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'swin':
            self.block = SwinSubBlock(
                in_channels, input_resolution, layer_i=layer_i, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path)
        elif model_type == 'uniformer':
            block_type = 'MHSA' if in_channels == out_channels and layer_i > 0 else 'Conv'
            self.block = UniformerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path, block_type=block_type)
        elif model_type == 'van':
            self.block = VANSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'vit':
            self.block = ViTSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        else:
            assert False and "Invalid model_type in SimVP"

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        return z
