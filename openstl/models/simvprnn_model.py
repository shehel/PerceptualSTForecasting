import torch
from torch import nn
from types import SimpleNamespace
from openstl.modules import (ConvSC, ConvLSTMCell, ConvSC3D, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                           HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                           SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock, TAUSubBlock, UNetConvBlock,
                           UNetUpBlock)

import pdb
from einops import rearrange, reduce

class ConvLSTM_Model(nn.Module):
    r"""ConvLSTM Model

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    """

    def __init__(self, num_layers, num_hidden, inshape, config):
        super(ConvLSTM_Model, self).__init__()
        T, C, H, W = inshape

        self.configs = config
        self.frame_channel = C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        height = H 
        width = W 

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden[i], height, width, self.configs.filter_size,
                                       self.configs.stride, self.configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        #frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        #mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # reverse schedule sampling
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.pre_seq_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()

        return next_frames



class SimVPRnn_Model(nn.Module):
    r"""SimVP Model

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, **kwargs):
        super(SimVPRnn_Model, self).__init__()
        kwargs = SimpleNamespace(**kwargs)
        T, C, H, W = in_shape  # T is pre_seq_length
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        #self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec)
        # self.enc_s = Encoder(1, 16, 2, spatio_kernel_enc)
        # self.enc_sc = Encoder(16*C, hid_S, 2, spatio_kernel_enc)
        # self.dec_sc = Decoder(hid_S,16*C,2, spatio_kernel_dec)
        # self.dec_s = Decoder(16, 1, 2, spatio_kernel_dec)
        # self.enc_s = Encoder(1, 16, 2, spatio_kernel_enc)
        # self.enc_s1 = Encoder(1, 16, 2, spatio_kernel_enc)
        # self.enc_sc = Encoder(16*C, hid_S, 2, spatio_kernel_enc)
        # self.dec_sc = Decoder(hid_S,16*C,2, spatio_kernel_dec)
        # self.dec_s = Decoder(16, 1, 2, spatio_kernel_dec)
        # self.dec_s1 = Decoder(16, 1, 2, spatio_kernel_dec)
        num_hidden = [int(x) for x in kwargs.num_hidden.split(',')]
        num_layers = len(num_hidden)
        self.hid = ConvLSTM_Model(num_layers, num_hidden, (12,32,32,32),kwargs)

    def forward(self, x_raw, mask):
        B, T, C, H, W = x_raw.shape
        # x = x_raw.reshape(B*T*C, 1, H, W)
        # x, skip = self.enc_s(x)
        # x = x.reshape(B*T, 16*C, 64, 64)
        # embed, skip1 = self.enc_sc(x)
        x1, x2 = x_raw[:,:12], x_raw[:,12:]
        
        x1 = x1.reshape(B*12, C, H, W)
        x2 = x2.reshape(B*12, C, H, W)

        embed, skip = self.enc(x1)
        embed2, _ = self.enc(x2)
        _, C_, H_, W_ = embed.shape

        z1 = embed.view(B, T-12, C_, H_, W_)
        z2 = embed.view(B, T-12, C_, H_, W_)
        # combine z1 and z2 along first dimension
        z = torch.cat((z1, z2), dim=1)
        encoded = self.hid(z, mask)
        # concatnetate first timestep of z to the beginning of encoded
        encoded = encoded[:,11:] 
        hid = encoded.reshape(B*(T-12), C_, H_, W_)
        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T-12, C, H, W)

        #Y = Y.reshape(B, T, C, H, W)
        #

        # Y = self.dec_sc(hid, skip1)
        # Y = Y.reshape(B*T*C, 16, 64, 64)
        # Y = self.dec_s(Y, skip)
        # Y = Y.reshape(B, T, C, H, W)
        # Y = Y[:,:,0::2]
        return (Y,encoded)

    # def forward(self, x_raw):
    #     try:
    #         B, T, C, H, W = x_raw.shape # (1,12,8,128,128)
    #         x1 = x_raw[:, :, 0::2]
    #         x2 = x_raw[:, :, 1::2]
    #         x1 = rearrange(x1, 'b t c h w -> (b t c) 1 h w')
    #         x2 = rearrange(x2, 'b t c h w -> (b t c) 1 h w')
            
    #         x1, skip1 = self.enc_s(x1) # (48,16,64,64)
    #         x2, skip2 = self.enc_s1(x2) # (48,16,64,64)
    #         x = torch.cat((x1, x2), dim=0) # (96,16,64,64)
    #         half_B = x.shape[0] // 2

    #         x = rearrange(x, '(b t c) c1 h w -> (b t) (c c1) h w', b=B, t=T, c=C, c1=16)

    #         embed, skip3 = self.enc_sc(x) # (12,32,32,32)
    #         _, C_, H_, W_ = embed.shape

    #         z = rearrange(embed, '(b t) c h w -> b t c h w', b=B, t=T)
    #         encoded = self.hid(z)
    #         hid = rearrange(encoded, 'b t c h w -> (b t) c h w', c=C_, h=H_, w=W_)

    #         Y = self.dec_sc(hid, skip3) # (12,128,64,64)
    #         Y = rearrange(Y, '(b t) (c c1) h w -> (b t c) c1 h w', b=B, t=T, c1=16, c=C)
            
    #         # undo the cat operation
    #         Y1 = Y[:half_B]
    #         Y2 = Y[half_B:]

    #         Y1 = self.dec_s(Y1, skip1)
    #         Y2 = self.dec_s1(Y2, skip2)

    #         Y_out = torch.zeros(B, T, C, H, W, dtype=Y1.dtype, device=Y1.device)  # (12,8,128,128)

    #         Y_out[:,:,0::2,:,:] = rearrange(Y1, '(b t c) 1 h w -> b t c h w', b=B, t=T, c=4)
    #         Y_out[:,:,1::2,:,:] = rearrange(Y2, '(b t c) 1 h w -> b t c h w', b=B, t=T, c=4)

    #         return (Y_out, encoded)
    #     except:
    #         pdb.set_trace()
   
    def recon(self, x_raw):
        B, T, C, H, W = x_raw.shape
        # x = x_raw.reshape(B*T*C, 1, H, W)
        # x, skip = self.enc_s(x)
        # x = x.reshape(B*T, 16*C, 64, 64)
        # embed, skip1 = self.enc_sc(x)
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)

        Y = self.dec(embed, skip)
        Y = Y.reshape(B, T, C, H, W)

        #Y = Y.reshape(B, T, C, H, W)
        #

        # Y = self.dec_sc(hid, skip1)
        # Y = Y.reshape(B*T*C, 16, 64, 64)
        # Y = self.dec_s(Y, skip)
        # Y = Y.reshape(B, T, C, H, W)
        # Y = Y[:,:,0::2]
        return Y
    def encode(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        z = self.hid(z)

        return z
def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3,5,7,11], groups=8, **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        enc_layers = [gInception_ST(
            channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1,N2-1):
            enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers = [
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups)]
        for i in range(1,N2-1):
            dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_in,
                              incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2-1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1,self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1) )

        y = z.reshape(B, T, C, H, W)
        return y


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
        elif model_type in ['mlp', 'mlpmixer']:
            self.block = MLPMixerSubBlock(
                in_channels, input_resolution, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type in ['moga', 'moganet']:
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
        elif model_type == 'tau':
            self.block = TAUSubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'conv3d':
            self.block = ConvSC3D(
                in_channels, in_channels, kernel_size=3)
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
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y

class Mid3DNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(Mid3DNet, self).__init__()
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
        B, T, C, H, W = x.shape
        # re`shape so that the 2nd dimension becomes first and first becomes second using permute
        z = x.permute(0, 2, 1, 3, 4)

        for i in range(self.N2):
            z = self.enc[i](z)

        # permute back
        z = z.permute(0, 2, 1, 3, 4)
        return z