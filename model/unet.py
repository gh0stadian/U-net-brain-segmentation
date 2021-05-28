""" Full assembly of the parts to form the complete network """

# import torch.nn.functional as F
from torch import optim,sigmoid

from .unet_modules import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        L1 = 8
        L2 = L1 * 2
        L3 = L1 * 4
        L4 = L1 * 8
        L5 = L1 * 16
        factor = 2 if bilinear else 1

        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, L1)

        self.down1 = Down(L1, L2)
        self.down2 = Down(L2, L3)
        self.down3 = Down(L3, L4)
        self.down4 = Down(L4, L4 // factor)

        self.up1 = Up(L5, L3 // factor, bilinear)
        self.up2 = Up(L4, L2 // factor, bilinear)
        self.up3 = Up(L3, L1 // factor, bilinear)
        self.up4 = Up(L2, L1, bilinear)

        self.outc = OutConv(L1, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = torch.sigmoid(self.outc(x))
        return output





