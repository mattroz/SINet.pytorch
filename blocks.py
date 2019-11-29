import torch
import torch.nn as nn
from SE_blocks.blocks import scSE
import numpy as np


class DSConvSE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups):
        super().__init__()

        self.dconv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.se_block = scSE(out_channels)

    def forward(self, input):
        x = self.dconv(input)
        x = self.bn(x)
        x = self.act(x)
        x = self.se_block(x)

        return x


class S2Block(nn.Module):
    def __init__(self, n_channels, avg_pool_kernel, kernel_size, padding='same'):
        super().__init__()

        pad = kernel_size // 2 if padding == 'same' else 0

        self.pool = nn.AvgPool2d(avg_pool_kernel)
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

    def __init__(self, n_channels, block_1_args, block_2_args, groups=1):
        super().__init__()

        self.n_channels = n_channels
        out_channels = (n_channels // 2) if (n_channels // 2) != 0 else 1
        self.pwconv = nn.Conv2d(n_channels, out_channels, kernel_size=1, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        '''TODO: make feature_map_size and kernel_size for S2Block as S2Module arguments'''
        self.s2_block_1 = S2Block(out_channels, **block_1_args)
        self.s2_block_2 = S2Block(out_channels, **block_2_args)
        self.out_act = nn.PReLU()

    def forward(self, input):

        '''Shuffle channels'''
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


class SINetEncoder(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        '''Configs from paper, Table 2'''
        s2_modules_config = [
            [16, {'kernel_size': 3, 'avg_pool_kernel': 1}, {'kernel_size': 5, 'avg_pool_kernel': 1}],
            [48, {'kernel_size': 3, 'avg_pool_kernel': 1}, {'kernel_size': 3, 'avg_pool_kernel': 1}],
            [48, {'kernel_size': 3, 'avg_pool_kernel': 1}, {'kernel_size': 5, 'avg_pool_kernel': 1}],
            [96, {'kernel_size': 3, 'avg_pool_kernel': 1}, {'kernel_size': 3, 'avg_pool_kernel': 1}],
            [96, {'kernel_size': 5, 'avg_pool_kernel': 1}, {'kernel_size': 3, 'avg_pool_kernel': 2}],
            [96, {'kernel_size': 5, 'avg_pool_kernel': 2}, {'kernel_size': 3, 'avg_pool_kernel': 4}],
            [96, {'kernel_size': 3, 'avg_pool_kernel': 1}, {'kernel_size': 3, 'avg_pool_kernel': 1}],
            [96, {'kernel_size': 5, 'avg_pool_kernel': 1}, {'kernel_size': 5, 'avg_pool_kernel': 1}],
            [96, {'kernel_size': 3, 'avg_pool_kernel': 2}, {'kernel_size': 3, 'avg_pool_kernel': 4}],
            [96, {'kernel_size': 3, 'avg_pool_kernel': 1}, {'kernel_size': 5, 'avg_pool_kernel': 2}],
        ]

        self.s2_modules = []
        for i in range(len(s2_modules_config)):
            module_name = f's2_module_{i}'
            _in_channels = s2_modules_config[i][0]
            s2_block_1_args = s2_modules_config[i][1]
            s2_block_2_args = s2_modules_config[i][2]
            setattr(self, module_name, S2Module(_in_channels, s2_block_1_args, s2_block_2_args))
            self.s2_modules.append(getattr(self, module_name))

        self.conv2d = nn.Conv2d(in_channels, 12, kernel_size=2, stride=2)
        self.dconv_se_1 = DSConvSE(12, 16, 2, 2, 2)
        self.dconv_se_2 = DSConvSE(64, 48, 2, 2, 8)
        self.pwconv = nn.Conv2d(144, n_classes, kernel_size=1)
        self.act = nn.ReLU()

    def forward(self, input):
        x = self.conv2d(input)
        print(x.shape)
        x = self.act(x)
        x = self.dconv_se_1(x)
        print(x.shape)

        res = x
        for i in range(2):
            print('RES: ', res.shape)
            res = self.s2_modules[i](res)
            print('RES: ', res.shape)

        x = torch.cat([x, res], dim=1)
        print(x.shape)
        x = self.dconv_se_2(x)
        print(x.shape)

        res = x
        for i in range(2, len(self.s2_modules)):
            res = self.s2_modules[i](res)

        x = torch.cat([x, res], dim=1)
        print(x.shape)
        x = self.act(self.pwconv(x))

        return x





class SINet(nn.Module):

    def __init__(self):
        super().__init__()

    def get_encoder(self):
        pass

    def get_decoder(self):
        pass
