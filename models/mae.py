import torch.nn as nn
from torch import Tensor
from einops import rearrange
import torch
from functools import partial
from loguru import logger
from typing import Optional, List, Tuple
from torch.nn.modules import conv
import torchaudio.transforms as audio_transforms
from einops import rearrange
from einops.layers.torch import Rearrange
from itertools import repeat

from models.audiotransformer import AudioTransformer, Block, Attention, trunc_normal_


def random_masking(x: Tensor,
                   mask_ratio: float) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(
        noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x,
                            dim=1,
                            index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


class AudioPatchEmbed(nn.Module):

    def __init__(self,
                 input_size=224,
                 patch_size=16,
                 patch_stride=16,
                 in_chans=1,
                 embed_dim=768,
                 norm_layer=None,
                 flatten=False):
        super().__init__()
        input_size = to_2tuple(input_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.input_size = input_size
        self.patch_size = patch_size
        self.grid_size = (input_size[0] // patch_stride[0],
                          input_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = rearrange(x, 'b c f t -> b (f t) c')
        x = self.norm(x)
        return x


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerScale(nn.Module):

    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class AudioTransformerMAE_Encoder(AudioTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_features(self, x, mask_ratio):
        b, c, f, t = x.shape
        x = x + self.time_pos_embed[:, :, :, :t]
        x = x + self.freq_pos_embed[:, :, :, :]  # Just for sin pos embed
        x = rearrange(x, 'b c f t -> b (f t) c')
        x, mask, ids_restore = random_masking(x, mask_ratio)
        if self.pooling == 'token':
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            cls_token = cls_token + self.token_pos_embed[:, :]
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_to_patch(self, x):
        if self.training:
            x = self.wavtransforms(x.unsqueeze(1)).squeeze(1)
        x = self.front_end(x)
        x = rearrange(x, 'b f t -> b 1 f t')
        if self.init_bn is not None:
            x = self.init_bn(x)
        x = self.patch_embed(x)
        return x

    def forward(self, x, mask_ratio: float = 0.75):
        x = self.forward_to_patch(x)
        x, mask, restore_idxs = self.forward_features(x, mask_ratio=mask_ratio)
        return x, mask, restore_idxs


class AudioTransformerMAE_Decoder(nn.Module):

    def __init__(self,
                 input_dim,
                 outputdim,
                 patch_size=16,
                 patch_stride=16,
                 embed_dim=768,
                 num_patches=100,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=None,
                 act_layer=None,
                 attention_type='Attention',
                 init_values=None,
                 **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_stride = patch_stride
        self.patch_size = patch_size

        self.input_proj = nn.Linear(input_dim, embed_dim)

        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * .02)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0,
                norm_layer=norm_layer,
                act_layer=act_layer,
                attention_type=globals()[attention_type],
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.outputlayer = nn.Sequential(nn.LayerNorm(self.embed_dim),
                                         nn.Linear(self.embed_dim, outputdim))
        self.apply(self.init_weights)
        torch.nn.init.normal_(self.mask_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'time_pos_embed', 'cls_token', 'freq_pos_embed', 'token_pos_embed'
        }

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x, ids_restore):
        x = self.input_proj(x)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_,
                          dim=1,
                          index=ids_restore.unsqueeze(-1).repeat(
                              1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        t = x.shape[1]

        x = x + self.pos_embed[:, :t, :]
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x, restore_idxs):
        x = self.forward_features(x, restore_idxs)
        return self.outputlayer(x)[:, 1:, :]


class AudioTransformerMAE(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.unfold = nn.Unfold(
            kernel_size=self.encoder.patch_embed.patch_size,
            stride=self.encoder.patch_embed.patch_size)

    def patchify(self, x):
        return self.unfold(x.unsqueeze(1)).transpose(-2, -1)

    def forward(self, x, mask_ratio: float = 0.75):
        latent, mask, restore_ids = self.encoder(x, mask_ratio=mask_ratio)
        pred = self.decoder(latent, restore_ids)
        targets = self.encoder.front_end(x)
        return pred, self.patchify(targets), mask


def mae_audio_transformer_tiny(**kwargs):
    encoder_kwargs = dict(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
    )
    encoder_kwargs = dict(encoder_kwargs, **kwargs)
    encoder = AudioTransformerMAE_Encoder(**encoder_kwargs)

    decoder_kwargs = dict(embed_dim=512,
                          depth=4,
                          num_heads=8,
                          input_dim=encoder_kwargs['embed_dim'],
                          outputdim=encoder.patch_embed.patch_size[0] *
                          encoder.patch_embed.patch_size[1],
                          num_patches=encoder.patch_embed.num_patches)
    decoder = AudioTransformerMAE_Decoder(**decoder_kwargs)
    return AudioTransformerMAE(encoder, decoder)


def mae_audio_transformer_small(**kwargs):
    encoder_kwargs = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
    )
    encoder_kwargs = dict(encoder_kwargs, **kwargs)
    encoder = AudioTransformerMAE_Encoder(**encoder_kwargs)

    decoder_kwargs = dict(embed_dim=512,
                          depth=8,
                          num_heads=8,
                          input_dim=encoder_kwargs['embed_dim'],
                          outputdim=encoder.patch_embed.patch_size[0] *
                          encoder.patch_embed.patch_size[1],
                          num_patches=encoder.patch_embed.num_patches)
    decoder = AudioTransformerMAE_Decoder(**decoder_kwargs)
    return AudioTransformerMAE(encoder, decoder)


def mae_audio_transformer_base(**kwargs):
    encoder_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
    )
    encoder_kwargs = dict(encoder_kwargs, **kwargs)
    encoder = AudioTransformerMAE_Encoder(**encoder_kwargs)

    decoder_kwargs = dict(embed_dim=512,
                          depth=8,
                          num_heads=8,
                          input_dim=encoder_kwargs['embed_dim'],
                          outputdim=encoder.patch_embed.patch_size[0] *
                          encoder.patch_embed.patch_size[1],
                          num_patches=encoder.patch_embed.num_patches)
    decoder = AudioTransformerMAE_Decoder(**decoder_kwargs)
    return AudioTransformerMAE(encoder, decoder)


if __name__ == "__main__":
    pass
