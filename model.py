import torch
import torch.nn as nn

from blocks import SINetDecoder, SINetEncoder


class SINet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.encoder = SINetEncoder(in_channels=3, n_classes=n_classes)
        self.decoder = SINetDecoder(in_channels=n_classes, inf_block_channels=48)

    def forward(self, input):
        bottleneck, encoder_feature_map = self.encoder(input)
        x = self.decoder((bottleneck, encoder_feature_map))

        return x