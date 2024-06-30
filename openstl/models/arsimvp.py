import torch
from torch import nn
from openstl.modules import (ConvSC, ConvSC3D, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                           HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                           SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock, TAUSubBlock, UNetConvBlock,
                           UNetUpBlock)

import pdb
from einops import rearrange, reduce
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Time_MLP(nn.Module):
    def __init__(self, dim):
        super(Time_MLP, self).__init__()
        self.sinusoidaposemb = SinusoidalPosEmb(dim)
        self.linear1 = nn.Linear(dim, dim*4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim*4, dim)

    def forward(self, x):
        x = self.sinusoidaposemb(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

    def random_masking(z, mask_ratio=0.75):
        B, T, C, H, W = z.shape
        len_keep = int(T * (1. - mask_ratio))
        noise = torch.rand(B, T, device=z.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.ones(B, T, device=z.device)
        mask.scatter_(1, ids_shuffle[:, :len_keep], 0)
        mask = mask.bool()
        z[mask] = 0
        return z
    """
    def random_masking_train(self, z, t):
        B, T, C , H, W = z.shape
        for i in range(B):
            rand_index = random.choices([0,1], weights=[0.5, 0.5], k=10)
            for index, j in enumerate(rand_index):
                if j==0:
                    z[i,index,:,:,:] = self.mask_token[index,:,:,:]
                if index >= int(t[i]/100):
                    z[i,index,:,:,:] = self.mask_token[index,:,:,:]
        return z
    """ 
class ARSimVP_Model(nn.Module):
    r"""SimVP Model

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, **kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.time_mlp = Time_MLP(dim=32)
        self.lp = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)
        #TODO parameters of mask might be off
        self.mask_token = nn.Parameter(torch.zeros(10, hid_S, 16, 16))
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
        model_type = 'gsta' if model_type is None else model_type.lower()
        if model_type == 'incepu':
            self.hid = MidIncepNet(T*hid_S, hid_T, N_T)
        elif model_type == 'unet':
            self.hid = UnetNet(T*hid_S, N_T, hid_T)
        elif model_type == 'conv3d':
            self.hid = Mid3DNet(hid_S, hid_T, N_T,
                input_resolution=(H, W), model_type=model_type,
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)


        else:
            self.hid = MidMetaNet(T*hid_S, hid_T, N_T,
                input_resolution=(H, W), model_type=model_type,
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)

    def forward(self, x_raw, y_raw=None, t=None):
        if is_train = True:
            B, T, C, H, W = x_raw.shape
            
            # x = x_raw.reshape(B*T*C, 1, H, W)
            # x, skip = self.enc_s(x)
            # x = x.reshape(B*T, 16*C, 64, 64)
            # embed, skip1 = self.enc_sc(x)
            x = x_raw.view(B*T, C, H, W)
            y = y_raw.view(B*T, C, H, W)

            time_emb = self.time_mlp(t)

            
            embed, skip = self.enc(x)
            embed_y, _  = self.enc(y)
            _, C_, H_, W_ = embed.shape

            z = embed.view(B, T, C_, H_, W_)
            z2 = embed.view(B, T, C_, H_, W_)

            z2 = self.random_masking(z2, t)
            
            z2 = torch.cat([z, z2], dim=1)
            
            encoded = self.hid(z, time_emb)
            hid = encoded.reshape(B*T, C_, H_, W_)
            Y = self.dec(hid, skip)
            Y = Y.reshape(B, T, C, H, W)

            #Y = Y.reshape(B, T, C, H, W)
            #

            # Y = self.dec_sc(hid, skip1)
            # Y = Y.reshape(B*T*C, 16, 64, 64)
            # Y = self.dec_s(Y, skip)
            # Y = Y.reshape(B, T, C, H, W)
            # Y = Y[:,:,0::2]
            return (Y,encoded)
        else:
            B, T, C, H, W = x_raw.shape
            
            # x = x_raw.reshape(B*T*C, 1, H, W)
            # x, skip = self.enc_s(x)
            # x = x.reshape(B*T, 16*C, 64, 64)
            # embed, skip1 = self.enc_sc(x)
            x = x_raw.view(B*T, C, H, W)
            y = y_raw.view(B*T, C, H, W)

            time_emb = self.time_mlp(t)

            
            embed, skip = self.enc(x)
            mask_token = self.mask_token.repeat(B,1,1,1,1)
            _, C_, H_, W_ = embed.shape

            z = embed.view(B, T, C_, H_, W_)
            z2 = embed.view(B, T, C_, H_, W_)

            z2 = self.random_masking(z2, t)
            
            z2 = torch.cat([z, z2], dim=1)
            
            encoded = self.hid(z, time_emb)
            hid = encoded.reshape(B*T, C_, H_, W_)
            Y = self.dec(hid, skip)
            Y = Y.reshape(B, T, C, H, W)

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