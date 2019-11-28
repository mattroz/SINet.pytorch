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