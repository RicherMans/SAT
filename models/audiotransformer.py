from collections.abc import Callable
from functools import partial
import math
from pathlib import Path
from typing import Optional, Tuple, Union

from einops import rearrange
from einops.layers.torch import Rearrange
from loguru import logger
import torch
import torch.nn as nn
import torchaudio.transforms as audio_transforms

from models.checkpoints import register_model, build_mdl
from models.layers import AudioPatchEmbed, DropPath, Mlp, trunc_normal_, to_2tuple


def drop_patches(x: torch.Tensor, dim: int, frac: float) -> torch.Tensor:
    """drop_patches.

    Drops patches from a tensor (B, *) on dimension dim

    :param x:
    :type x: torch.Tensor
    :param dim:
    :type dim: int
    :param frac:
    :type frac: float
    :rtype: torch.Tensor
    """

    N = x.shape[dim]
    to_keep = N - int(N * frac)
    random_mask = torch.randperm(N, device=x.device)[:to_keep].sort().values
    return x.index_select(dim=dim, index=random_mask)


class Normer(nn.Module):

    def __init__(self, mean, std, fac=2.0):
        super().__init__()
        self.mean = mean
        self.fac = fac
        self.std = std

    def forward(self, x):
        return (x - self.mean) / (self.fac * self.std)


class StreamingAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cache=None):
        B, N, D = x.shape
        q = self.to_q(x).reshape(B, N, self.num_heads,
                                 D // self.num_heads).permute(0, 2, 1, 3)
        if cache is None:
            # length 0 is not used during concat
            cache = torch.zeros(B, 0, D, device=x.device, dtype=x.dtype)

        kv_input = torch.cat((cache, x), dim=1)
        kv_len = kv_input.shape[1]
        kv = self.to_kv(kv_input).reshape(B, kv_len, 2, self.num_heads,
                                          D // self.num_heads).permute(
                                              2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class StreamingSequential(nn.Sequential):

    def __init__(self, *args, cache_size: int = 8):
        super().__init__(*args)
        self.cache_size = cache_size
        self.n_layers = len(self)

    def forward(self, x, in_cache=None):
        hids = []
        B, T, D = x.shape
        if in_cache is None:
            in_cache = torch.empty(self.n_layers,
                                   B,
                                   0,
                                   D,
                                   device=x.device,
                                   dtype=x.dtype)

        for i, module in enumerate(self._modules.values()):
            inp_cache = in_cache[i]
            hids.append(x)
            x = module(x, inp_cache)
        hids = torch.stack(hids)
        if self.cache_size > 0:
            new_mems = torch.cat((in_cache, hids),
                                 dim=-2).detach()[:, :, -self.cache_size:, :]
        else:
            new_mems = None
        return x, new_mems


class StreamingBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        init_values=None,
        drop_path=0.,
        act_layer: Callable = nn.ReLU,
        norm_layer: Callable = nn.LayerNorm,
        attention_type: Callable = StreamingAttention,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attention_type(dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   attn_drop=attn_drop,
                                   proj_drop=drop)
        self.ls1 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)
        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, cache):
        if cache is not None and cache.shape[1] > 0:
            cache = self.norm1(cache)
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), cache)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class LayerScale(nn.Module):

    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        causal: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.causal = causal

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(
            0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.causal:
            mask_value = -torch.finfo(attn.dtype).max
            i, j = attn.shape[-2:]
            mask = torch.ones(i, j, device=q.device,
                              dtype=torch.bool).triu(j - i + 1)
            attn = attn.masked_fill(mask, mask_value)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        init_values=None,
        drop_path=0.,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        attention_type: Callable = Attention,
        attention_kwargs={},
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attention_type(dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   attn_drop=attn_drop,
                                   proj_drop=drop,
                                   **attention_kwargs)
        self.ls1 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)
        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class AudioTransformer(nn.Module):

    def __init__(self,
                 outputdim=527,
                 patch_size=16,
                 patch_stride=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 init_bn: bool = True,
                 norm_layer=None,
                 act_layer=None,
                 init_values=None,
                 target_length=1012,
                 pooling='token',
                 wavtransforms=None,
                 spectransforms=None,
                 time_patch_out: Optional[float] = None,
                 freq_patch_out: Optional[float] = None,
                 block_type=Block,
                 attention_type=Attention,
                 eval_avg='mean',
                 **kwargs):
        super().__init__()
        assert pooling in ('mean', 'token', 'dm', 'logit')
        self.outputdim = outputdim
        self.pooling = pooling
        self.embed_dim = embed_dim
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.n_mels = kwargs.get('n_mels', 64)
        n_fft = kwargs.get('n_fft', 512)
        self.hop_size = kwargs.get('hop_size', 160)
        self.win_size = kwargs.get('win_size', 512)
        f_min = kwargs.get('f_min', 0)
        f_max = kwargs.get('f_max', 8000)
        self.center = kwargs.get('center', True)
        self.pad_last = kwargs.get('pad_last', True)
        self.eval_avg = eval_avg
        self.time_patch_out = time_patch_out
        self.freq_patch_out = freq_patch_out

        self.front_end = nn.Sequential(
            audio_transforms.MelSpectrogram(f_min=f_min,
                                            sample_rate=16000,
                                            win_length=self.win_size,
                                            center=self.center,
                                            n_fft=n_fft,
                                            f_max=f_max,
                                            hop_length=self.hop_size,
                                            n_mels=self.n_mels),
            audio_transforms.AmplitudeToDB(top_db=120),
        )

        if init_bn:
            self.init_bn = nn.Sequential(
                Rearrange('b c f t -> b f c t'),
                torch.nn.BatchNorm2d(self.n_mels, momentum=0.01),
                Rearrange('b f c t -> b c f t'))
        else:
            if self.n_mels == 64:
                self.init_bn = Normer(-10, 20)
            elif self.n_mels == 128:
                self.init_bn = Normer(-14.27, 20.79)
        self.target_length = target_length

        self.maximal_allowed_length = self.target_length + to_2tuple(
            self.patch_stride)[-1] - 1
        self.patch_embed = AudioPatchEmbed(input_size=(self.n_mels,
                                                       target_length),
                                           embed_dim=self.embed_dim,
                                           patch_size=self.patch_size,
                                           flatten=False,
                                           patch_stride=self.patch_stride)
        self.spectransforms = nn.Sequential(
        ) if spectransforms is None else spectransforms
        self.wavtransforms = nn.Sequential(
        ) if wavtransforms is None else wavtransforms

        if self.pooling == 'token':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.token_pos_embed = nn.Parameter(
                torch.randn(1, embed_dim) * .02)

        self.time_pos_embed = nn.Parameter(
            torch.randn(1, embed_dim, 1, self.patch_embed.grid_size[1]) * .02)
        self.freq_pos_embed = nn.Parameter(
            torch.randn(1, embed_dim, self.patch_embed.grid_size[0], 1) * .02)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.Sequential(*[
            block_type(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                attention_type=attention_type,
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.outputlayer = nn.Sequential(nn.LayerNorm(self.embed_dim),
                                         nn.Linear(self.embed_dim, outputdim))
        self.apply(self.init_weights)
        if hasattr(self, 'cls_token'):
            nn.init.normal_(self.cls_token, std=1e-6)

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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        b, c, f, t = x.shape
        x = x + self.time_pos_embed[:, :, :, :t]
        x = x + self.freq_pos_embed[:, :, :, :]  # Just to support __getitem__ in posembed
        if self.training and self.time_patch_out is not None:
            x = drop_patches(x, dim=-1, frac=self.time_patch_out)
        if self.training and self.freq_patch_out is not None:
            x = drop_patches(x, dim=-2, frac=self.freq_patch_out)
        x = rearrange(x, 'b c f t -> b (f t) c')
        if self.pooling == 'token':
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            cls_token = cls_token + self.token_pos_embed
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == 'token':
            x = x[:, 0]
            return self.outputlayer(x).sigmoid()
        elif self.pooling == 'mean':
            x = x.mean(1)
            return self.outputlayer(x).sigmoid()
        elif self.pooling == 'logit':
            x = x.mean(1)
            return self.outputlayer(x)
        elif self.pooling == 'dm':
            # Unpack using the frequency dimension, which is constant
            x = rearrange(x,
                          'b (f t) d -> b f t d',
                          f=self.patch_embed.grid_size[0])
            #First poolin frequency, then sigmoid the (B T D) output
            x = self.outputlayer(x.mean(1)).sigmoid()
            return x.mean(1)
        else:
            return x.mean(1)

    def load_state_dict(self, state_dict, strict=True):
        if 'time_pos_embed' in state_dict and hasattr(
                self, 'time_pos_embed'
        ) and self.time_pos_embed.shape != state_dict['time_pos_embed'].shape:
            logger.debug(
                "Positional Embedding shape not the same with model, resizing!"
            )
            self.change_pos_embedding(state_dict)
        elif hasattr(self, 'pos_conv'):
            # For ConvPos models, just remove the time_pos_embed
            if 'time_pos_embed' in state_dict:
                del state_dict['time_pos_embed']
            if 'freq_pos_embed' in state_dict:
                del state_dict['freq_pos_embed']
        super().load_state_dict(state_dict, strict=strict)

    def change_pos_embedding(self, state_dict):
        target_time_pos_embed_length = self.time_pos_embed.shape[-1]
        target_freq_pos_embed_length = self.freq_pos_embed.shape[-2]

        pretrained_time_pos_embed = state_dict['time_pos_embed']
        pretrained_freq_pos_embed = state_dict['freq_pos_embed']

        if target_time_pos_embed_length <= pretrained_time_pos_embed.shape[-1]:
            state_dict['time_pos_embed'] = pretrained_time_pos_embed[
                ..., :target_time_pos_embed_length]
        else:
            state_dict['time_pos_embed'] = torch.nn.functional.interpolate(
                pretrained_time_pos_embed,
                size=(1, target_time_pos_embed_length),
                align_corners=False,
                mode='bilinear')
        if target_freq_pos_embed_length <= pretrained_freq_pos_embed.shape[-2]:
            state_dict[
                'freq_pos_embed'] = pretrained_freq_pos_embed[:, :, :
                                                              target_freq_pos_embed_length, :]
        else:
            state_dict['freq_pos_embed'] = torch.nn.functional.interpolate(
                pretrained_freq_pos_embed,
                size=(target_freq_pos_embed_length, 1),
                align_corners=False,
                mode='bilinear')

    def forward_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b f t -> b 1 f t')
        if self.init_bn is not None:
            x = self.init_bn(x)
        if x.shape[-1] > self.maximal_allowed_length:
            #When testing with a longer input
            # splits = x.unfold(-1, self.target_length,
            # 16).permute(3, 0, 1, 2, 4)
            # for f in splits:
            splits = x.split(self.target_length, -1)

            if splits[-1].shape[-1] < self.target_length:
                if self.pad_last:
                    pad = torch.zeros(*x.shape[:-1],
                                      self.target_length,
                                      device=x.device)
                    pad[..., :splits[-1].shape[-1]] = splits[-1]
                    splits = torch.stack((*splits[:-1], pad), dim=0)
                else:
                    splits = torch.stack(splits[:-1], dim=0)
            else:
                splits = torch.stack(splits[:-1], dim=0)
            n_splits = len(splits)
            x = rearrange(splits, 'spl b c f t-> (spl b) c f t')
            x = self.forward_head(self.forward_features(x))
            x = rearrange(x, '(spl b) d -> spl b d', spl=n_splits)
            if self.eval_avg == 'mean':
                x = x.mean(0)
            elif self.eval_avg == 'max':
                x = x.max(0)[0]
            else:
                raise ValueError(
                    f'Unknown Eval average function ({self.eval_avg})')

        else:
            x = self.forward_features(x)
            x = self.forward_head(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = self.wavtransforms(x.unsqueeze(1)).squeeze(1)
        x = self.front_end(x)
        if self.training:
            x = self.spectransforms(x)
        x = self.forward_spectrogram(x)
        return x


class Streaming_AudioTransformer(nn.Module):

    def __init__(self,
                 outputdim=527,
                 patch_size=16,
                 patch_stride=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 init_bn: bool = True,
                 norm_layer=None,
                 act_layer=None,
                 init_values=None,
                 target_length=1012,
                 cache_length=0,
                 pooling='mean',
                 wavtransforms=None,
                 spectransforms=None,
                 time_patch_out: Optional[float] = None,
                 freq_patch_out: Optional[float] = None,
                 block_type=StreamingBlock,
                 attention_type=StreamingAttention,
                 eval_avg='mean',
                 **kwargs):
        super().__init__()
        self.outputdim = outputdim
        self.pooling = pooling
        self.embed_dim = embed_dim
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.n_mels: int = kwargs.get('n_mels', 64)
        n_fft = kwargs.get('n_fft', 512)
        self.hop_size = kwargs.get('hop_size', 160)
        self.win_size = kwargs.get('win_size', 512)
        f_min = kwargs.get('f_min', 0)
        f_max = kwargs.get('f_max', 8000)
        self.pad_last = kwargs.get('pad_last', False)
        self.center = kwargs.get('center', True)
        self.depth = depth
        self.eval_avg = eval_avg
        self.time_patch_out = time_patch_out
        self.freq_patch_out = freq_patch_out

        self.front_end = nn.Sequential(
            audio_transforms.MelSpectrogram(f_min=f_min,
                                            sample_rate=16000,
                                            win_length=self.win_size,
                                            center=self.center,
                                            n_fft=n_fft,
                                            f_max=f_max,
                                            hop_length=self.hop_size,
                                            n_mels=self.n_mels),
            audio_transforms.AmplitudeToDB(top_db=120),
        )

        if init_bn:
            self.init_bn = nn.Sequential(
                Rearrange('b c f t -> b f c t'),
                torch.nn.BatchNorm2d(self.n_mels, momentum=0.01),
                Rearrange('b f c t -> b c f t'))
        else:
            if self.n_mels == 64:
                self.init_bn = Normer(-10, 20)
            elif self.n_mels == 128:
                self.init_bn = Normer(-14.27, 20.79)
        self.target_length = target_length
        self.cache_length = cache_length
        self.max_length = self.target_length + self.cache_length
        self.patch_stride = to_2tuple(self.patch_stride)
        self.patch_embed = AudioPatchEmbed(input_size=(self.n_mels,
                                                       self.max_length),
                                           embed_dim=self.embed_dim,
                                           patch_size=self.patch_size,
                                           flatten=False,
                                           patch_stride=self.patch_stride)
        self.spectransforms = nn.Sequential(
        ) if spectransforms is None else spectransforms
        self.wavtransforms = nn.Sequential(
        ) if wavtransforms is None else wavtransforms

        if self.pooling == 'token':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.token_pos_embed = nn.Parameter(
                torch.randn(1, embed_dim) * .02)
        self.time_pos_embed = nn.Parameter(
            torch.randn(1, embed_dim, 1, self.patch_embed.grid_size[1]) * .02)
        self.freq_pos_embed = nn.Parameter(
            torch.randn(1, embed_dim, self.patch_embed.grid_size[0], 1) * .02)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.pos_drop = nn.Dropout(p=drop_rate)
        cache_size = int(
            self.patch_embed.grid_size[0] *
            (self.cache_length // self.patch_embed.patch_stride[1]))

        self.blocks = StreamingSequential(*[
            block_type(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                attention_type=attention_type,
            ) for i in range(depth)
        ],
                                          cache_size=cache_size)
        self.norm = norm_layer(embed_dim)
        self.outputlayer = nn.Sequential(nn.LayerNorm(self.embed_dim),
                                         nn.Linear(self.embed_dim, outputdim))
        self.apply(self.init_weights)
        if hasattr(self, 'cls_token') and self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)

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

    def forward_features(self,
                         x,
                         cache=None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)
        b, c, f, t = x.shape
        x = x + self.time_pos_embed[..., :t]
        x = x + self.freq_pos_embed
        # Update the cache , i.e., check position
        if cache is not None and cache.shape[2] > 0:
            init_cache = cache[0]
            init_cache = rearrange(init_cache,
                                   'b (f t) d -> b d f t',
                                   f=self.patch_embed.grid_size[0])
            m_len = init_cache.shape[-1]
            init_cache = init_cache + self.time_pos_embed[
                ..., -m_len:] + self.freq_pos_embed
            init_cache = rearrange(init_cache, 'b d f t -> 1 b (f t) d')
            # Cat first layer cache with the others
            cache = torch.cat((init_cache, cache[1:]), dim=0)
        if self.training and self.time_patch_out is not None:
            x = drop_patches(x, dim=-1, frac=self.time_patch_out)
        if self.training and self.freq_patch_out is not None:
            x = drop_patches(x, dim=-2, frac=self.freq_patch_out)
        x = rearrange(x, 'b c f t -> b (f t) c')
        if self.pooling == 'token':
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            cls_token = cls_token + self.token_pos_embed
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x)
        x, out_cache = self.blocks(x, in_cache=cache)
        x = self.norm(x)
        return x, out_cache

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == 'token':
            x = x[:, 0]
            return self.outputlayer(x).sigmoid()
        elif self.pooling == 'mean':
            x = x.mean(1)
            return self.outputlayer(x).sigmoid()
        elif self.pooling == 'logit':
            x = x.mean(1)
            return self.outputlayer(x)
        elif self.pooling == 'dm':
            # Unpack using the frequency dimension, which is constant
            x = rearrange(x,
                          'b (f t) d -> b f t d',
                          f=self.patch_embed.grid_size[0])
            #First poolin frequency, then sigmoid the (B T D) output
            x = self.outputlayer(x.mean(1)).sigmoid()
            return x.mean(1)

    def load_state_dict(self, state_dict, strict=True):
        if 'time_pos_embed' in state_dict and self.time_pos_embed.shape != state_dict[
                'time_pos_embed'].shape:
            logger.debug(
                "Positional Embedding shape not the same with model, resizing!"
            )
            self.change_pos_embedding(state_dict)
        super().load_state_dict(state_dict, strict=strict)

    def change_pos_embedding(self, state_dict):
        target_time_pos_embed_length = self.time_pos_embed.shape[-1]
        target_freq_pos_embed_length = self.freq_pos_embed.shape[-2]

        pretrained_time_pos_embed = state_dict['time_pos_embed']
        pretrained_freq_pos_embed = state_dict['freq_pos_embed']

        if target_time_pos_embed_length <= pretrained_time_pos_embed.shape[-1]:
            state_dict['time_pos_embed'] = pretrained_time_pos_embed[
                ..., :target_time_pos_embed_length]
        else:
            state_dict['time_pos_embed'] = torch.nn.functional.interpolate(
                pretrained_time_pos_embed,
                size=(1, target_time_pos_embed_length),
                align_corners=False,
                mode='bilinear')
        if target_freq_pos_embed_length <= pretrained_freq_pos_embed.shape[-2]:
            state_dict[
                'freq_pos_embed'] = pretrained_freq_pos_embed[:, :, :
                                                              target_freq_pos_embed_length, :]
        else:
            state_dict['freq_pos_embed'] = torch.nn.functional.interpolate(
                pretrained_freq_pos_embed,
                size=(target_freq_pos_embed_length, 1),
                align_corners=False,
                mode='bilinear')

    def forward_spectrogram(self,
                            x: torch.Tensor,
                            cache=None,
                            return_cache: bool = False) -> torch.Tensor:
        x = rearrange(x, 'b f t -> b 1 f t')
        if self.init_bn is not None:
            x = self.init_bn(x)
        if x.shape[-1] > self.target_length:
            #When testing with a longer input
            outs = []
            # splits = x.unfold(-1, self.target_length,
            # 16).permute(3, 0, 1, 2, 4)
            # for f in splits:
            # Just drop the last sample, enhances performance
            for f in x.split(self.target_length, -1):
                length_f = f.shape[-1]
                if f.shape[-1] != self.target_length and self.pad_last:
                    # Pad to next token size
                    to_pad = int(
                        math.ceil(length_f / self.patch_stride[-1]) *
                        self.patch_stride[-1]) - length_f
                    f = torch.nn.functional.pad(f, (0, to_pad),
                                                mode='constant')
                elif f.shape[-1] != self.target_length and not self.pad_last:
                    continue

                f, cache = self.forward_features(f, cache)
                clip_out = self.forward_head(f)
                # For the last element in the sequence, clip
                outs.append(clip_out)
            outs = torch.stack(outs, -1)  # Concatenate to 1, T, C
            out_cache = cache
            if self.eval_avg == 'mean':
                x = outs.mean(-1)
            elif self.eval_avg == 'max':
                x = outs.max(-1)[0]
            elif self.eval_avg == 'last':
                x = outs[..., -1]
            else:
                raise ValueError(
                    f'Unknown Eval average function ({self.eval_avg})')

        else:
            x, out_cache = self.forward_features(x, cache)
            x = self.forward_head(x)
        if return_cache:
            return x, out_cache
        return x

    def forward(self, x, cache=None, return_cache: bool = False):
        if self.training:
            x = self.wavtransforms(x.unsqueeze(1)).squeeze(1)
        x = self.front_end(x)
        if self.training:
            x = self.spectransforms(x)
        return self.forward_spectrogram(x,
                                        cache=cache,
                                        return_cache=return_cache)


@register_model
def audiotransformer_tiny(
        num_classes: int = 527,
        pretrained: bool = False,
        pretrained_url:
    str = 'https://zenodo.org/record/7964975/files/audiotransformer_tiny_mAP_44_15.pt?download=1',
        **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=192,
                        depth=12,
                        num_heads=3,
                        init_bn=True,
                        mlp_ratio=4,
                        pooling='mean',
                        outputdim=num_classes)
    model_kwargs = dict(model_kwargs, **kwargs)
    return build_mdl(
        AudioTransformer,
        pretrained=pretrained,
        pretrained_url=pretrained_url,
        **model_kwargs,
    )


@register_model
def SAT_T_2s(
        num_classes: int = 527,
        pretrained: bool = False,
        pretrained_url:
    str = 'https://zenodo.org/record/7964975/files/SAT_T_stream2s_mAP_43_34.pt?download=1',
        **kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        init_bn=True,
        mlp_ratio=4,
        target_length=192,
        cache_length=192,
    )
    model_kwargs = dict(model_kwargs, **kwargs)
    return build_mdl(Streaming_AudioTransformer,
                     pretrained=pretrained,
                     pretrained_url=pretrained_url,
                     **model_kwargs)


@register_model
def SAT_T_1s(
        num_classes: int = 527,
        pretrained: bool = False,
        pretrained_url:
    str = 'https://zenodo.org/record/7964975/files/SAT_T_stream1s_mAP_40_11.pt?download=1',
        **kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        init_bn=True,
        mlp_ratio=4,
        target_length=96,
        cache_length=96,
    )
    model_kwargs = dict(model_kwargs, **kwargs)
    return build_mdl(Streaming_AudioTransformer,
                     pretrained=pretrained,
                     pretrained_url=pretrained_url,
                     **model_kwargs)


@register_model
def audiotransformer_small(
        num_classes: int = 527,
        pretrained: bool = False,
        pretrained_url:
    str = 'https://zenodo.org/record/7964975/files/audiotransformer_small_mAP_45_68.pt?download=1',
        **kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        init_bn=True,
        mlp_ratio=4,
        outputdim=num_classes,
    )
    model_kwargs = dict(model_kwargs, **kwargs)
    return build_mdl(
        AudioTransformer,
        pretrained=pretrained,
        pretrained_url=pretrained_url,
        **model_kwargs,
    )


@register_model
def SAT_S_2s(
        num_classes: int = 527,
        pretrained: bool = False,
        pretrained_url:
    str = 'https://zenodo.org/record/7964975/files/SAT_S_stream2s_mAP_43_37.pt?download=1',
        **kwargs):
    # While its not a big deal, in the paper it is said that we use mean pooling, but I forgot that some defaults were still using token...
    # meanpooling does improve performance generally so that's that
    model_kwargs = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        init_bn=True,
        mlp_ratio=4,
        pooling='token',
        target_length=192,
        cache_length=192,
    )
    model_kwargs = dict(model_kwargs, **kwargs)
    return build_mdl(
        Streaming_AudioTransformer,
        pretrained=pretrained,
        pretrained_url=pretrained_url,
        **model_kwargs,
    )


@register_model
def SAT_S_1s(
        num_classes: int = 527,
        pretrained: bool = False,
        pretrained_url:
    str = 'https://zenodo.org/record/7964975/files/SAT_S_stream1s_mAP_40_55.pt?download=1',
        **kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        init_bn=True,
        pooling='token',
        mlp_ratio=4,
        target_length=192,
        cache_length=192,
    )
    model_kwargs = dict(model_kwargs, **kwargs)
    return build_mdl(
        Streaming_AudioTransformer,
        pretrained=pretrained,
        pretrained_url=pretrained_url,
        **model_kwargs,
    )


@register_model
def audiotransformer_base(
        num_classes: int = 527,
        pretrained: bool = False,
        pretrained_url:
    str = 'https://zenodo.org/record/7964975/files/audiotransformer_base_mAP_47_40.pt?download=1',
        **kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        outputdim=num_classes,
    )
    return build_mdl(
        AudioTransformer,
        pretrained=pretrained,
        pretrained_url=pretrained_url,
        **model_kwargs,
    )


@register_model
def SAT_B_2s(
        num_classes: int = 527,
        pretrained: bool = False,
        pretrained_url:
    str = 'https://zenodo.org/record/7964975/files/SAT_B_stream2s_mAP_45_29.pt?download=1',
        **kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        init_bn=True,
        mlp_ratio=4,
        target_length=192,
        cache_length=192,
    )
    model_kwargs = dict(model_kwargs, **kwargs)
    return build_mdl(
        Streaming_AudioTransformer,
        pretrained=pretrained,
        pretrained_url=pretrained_url,
        **model_kwargs,
    )


@register_model
def SAT_B_1s(
        num_classes: int = 527,
        pretrained: bool = False,
        pretrained_url:
    str = 'https://zenodo.org/record/7964975/files/SAT_B_stream1s_mAP_41_37.pt?download=1',
        **kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        pooling='mean',
        init_bn=True,
        mlp_ratio=4,
        target_length=96,
        cache_length=96,
        attention_type=StreamingAttention,
    )
    model_kwargs = dict(model_kwargs, **kwargs)
    return build_mdl(
        Streaming_AudioTransformer,
        pretrained=pretrained,
        pretrained_url=pretrained_url,
        **model_kwargs,
    )


if __name__ == "__main__":
    pass
