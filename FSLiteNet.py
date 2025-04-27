import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from numpy import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class GlobalFilter1(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x


class FrequencySplitFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        # 低频权重
        self.low_complex_weight = nn.Parameter(
            torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02
        )
        # 高频权重
        self.high_complex_weight = nn.Parameter(
            torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02
        )
        self.w = w
        self.h = h
        # 任务自适应参数：控制高频和低频的权重比例
        self.beta = nn.Parameter(torch.tensor(0.5))  # 初始化为 0.5，表示均衡两种权重

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)
        x = x.to(torch.float32)

        # 傅里叶变换到频域
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        # 转换低频和高频权重到复数
        low_weight = torch.view_as_complex(self.low_complex_weight)
        high_weight = torch.view_as_complex(self.high_complex_weight)

        # 分别对低频和高频进行加权操作
        low_freq_output = x * low_weight
        high_freq_output = x * high_weight

        # 根据任务自适应参数 beta，组合高频和低频特征
        combined_freq_output = self.beta * low_freq_output + (1 - self.beta) * high_freq_output

        # 逆傅里叶变换回空间域
        x = torch.fft.irfft2(combined_freq_output, s=(a, b), dim=(1, 2), norm='ortho')

        # 还原为原始形状
        x = x.reshape(B, N, C)

        return x


class FrequencySplitBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = FrequencySplitFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=32, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class DFIBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, expansion=2, num_splits=8):
        super().__init__()
        self.num_splits = num_splits

        # 深度卷积
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2,
                                groups=in_channels)  # Group convolution
        self.norm = nn.BatchNorm2d(in_channels)

        # Pointwise Convolutions
        self.pwconv1 = nn.Conv2d(in_channels, out_channels * expansion, kernel_size=1)
        self.act = nn.ReLU()
        self.pwconv2 = nn.Conv2d(out_channels * expansion, out_channels, kernel_size=1)

        # 交互信息（PII-like） - 用于拼接后的特征交互
        self.interaction = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=1)  # 1x1卷积用于交互

    def forward(self, x):
        residual = x
        # 深度卷积
        x = self.dwconv(x)
        x = self.norm(x)

        # 将通道分割成8个部分，每个部分包含4个通道
        chunks = torch.chunk(x, self.num_splits, dim=1)  # 分割通道

        # 选择y1和y7进行拼接
        y1 = chunks[0]  # 选择第1个部分
        y7 = chunks[6]  # 选择第7个部分

        # 拼接y1和y7
        new_y1 = torch.cat([y1, y7], dim=1)  # 在通道维度拼接

        # 对拼接后的y1进行卷积交互
        new_y1 = self.interaction(new_y1)

        # 将新的y1与其他通道（y2, y3, y4, y5, y6, y8）拼接
        remaining_channels = chunks[1:6] + chunks[7:]  # 选择剩余的通道y2到y8
        x = torch.cat([new_y1] + list(remaining_channels), dim=1)  # 拼接所有通道

        # 对拼接后的特征图进行卷积交互
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # 残差连接
        return residual + x


class DFIBNet(nn.Module):

    def __init__(self, in_channels=3, num_stages=4, dims=[32, 64, 128, 256], depths=[2, 2, 6, 2]):
        super().__init__()
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        # Stem layer
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU()
        )
        self.downsample_layers.append(stem)

        for i in range(num_stages):
            blocks = nn.Sequential(
                *[DFIBlock(dims[i], dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(blocks)
            if i < num_stages - 1:
                self.downsample_layers.append(
                    nn.Sequential(
                        nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                        nn.BatchNorm2d(dims[i + 1]),
                        nn.ReLU()
                    )
                )

    def forward(self, x):
        for down, stage in zip(self.downsample_layers, self.stages):
            x = down(x)
            x = stage(x)
        return x



class FSFBNet(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=10, embed_dim=768, depth=12,
                 mlp_ratio=4., representation_size=None, uniform_drop=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=None,
                 dropcls=0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        h = img_size // patch_size
        w = h // 2 + 1

        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            FrequencySplitBlock(
                dim=embed_dim, mlp_ratio=mlp_ratio,
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x).mean(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x


class FSLiteNet(nn.Module):

    def __init__(self, img_size=224, num_classes=30, embed_dim=384, fdf_net_depth=6, patch_size=2,
                 mlp_ratio=4., drop_rate=0., drop_path_rate=0., norm_layer=None, dropcls=0,
                 convnext_dims=[32, 64, 128, 256], convnext_depths=[2, 2, 6, 2]):
        super().__init__()

        self.convnext = DFIBNet(
            in_channels=3, dims=convnext_dims, depths=convnext_depths
        )

        self.gf_net = FSFBNet(
            img_size=img_size // 16,
            patch_size=patch_size,
            in_chans=convnext_dims[-1],
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=fdf_net_depth,
            mlp_ratio=mlp_ratio, drop_rate=drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, dropcls=dropcls
        )

        self.residual_conv = nn.Conv2d(convnext_dims[-1], num_classes, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始化 alpha 为 0.5

    def forward(self, x):

        convnext_features = self.convnext(x)  # 输出形状为 (B, C, H, W)
        residual = self.residual_conv(convnext_features)  # (B, embed_dim, H, W)
        residual_pooled = residual.mean(dim=[2, 3])  # 平均池化 (B, embed_dim)
        fdf_net_output = self.gf_net(convnext_features)

        combined_output = 2 * self.alpha * fdf_net_output + 2 * (1 - self.alpha) * residual_pooled
        return combined_output
