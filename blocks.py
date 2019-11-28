import torch
import torch.nn as nn
import numpy as np


class S2Block(nn.Module):
    def __init__(self, n_channels, feature_map_size, kernel_size, padding='same'):
        super().__init__()

        pad = kernel_size // 2 if padding == 'same' else 0

        self.pool = nn.AdaptiveAvgPool2d(feature_map_size)
        self.dconv = nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=pad, groups=n_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.pwconv = nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.act = nn.PReLU()

    def forward(self, input):
        x = self.pool(input)
        x = self.dconv(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pwconv(x)
        x = self.bn2(x)
        x = nn.functional.interpolate(x, size=input.shape[-2:], mode='bilinear')

        return x


class S2Module(nn.Module):

    def __init__(self, n_channels, groups=1):
        super().__init__()

        self.n_channels = n_channels
        out_channels = (n_channels // 2) if (n_channels // 2) != 0 else 1
        self.pwconv = nn.Conv2d(n_channels, out_channels, kernel_size=1, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        '''TODO: make feature_map_size and kernel_size for S2Block as S2Module arguments'''
        self.s2_block_1 = S2Block(out_channels, 4, 3)
        self.s2_block_2 = S2Block(out_channels, 4, 3)
        self.out_act = nn.PReLU()

    def forward(self, input):
        permuted_channel_idxs = torch.randperm(self.n_channels)
        x = input[:, permuted_channel_idxs, :, :]
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)
        x1 = self.s2_block_1(x)
        x2 = self.s2_block_2(x)
        x = torch.cat([x1, x2], dim=1)
        x = torch.add(x, input)
        x = self.out_act(x)

        return x
