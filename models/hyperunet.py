
import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConv(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.dw = nn.Conv2d(
            ch,
            ch,
            kernel_size=3,
            padding=1,
            groups=ch
        )

        self.pw = nn.Conv2d(
            ch,
            ch,
            kernel_size=1
        )

    def forward(self, x):
        return self.pw(self.dw(x))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),
            DWConv(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class HyperUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()

        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)

        self.mid = ConvBlock(base_ch * 4, base_ch * 4)

        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.out = nn.Conv2d(
            base_ch,
            2,
            kernel_size=1
        )

        self.pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        m = self.mid(e3)

        d2 = F.interpolate(
            m,
            size=e2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        d2 = self.dec2(d2)

        d1 = F.interpolate(
            d2,
            size=e1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        d1 = self.dec1(d1)

        out = self.out(d1)

        delta_mag = 0.1 * out[:, 0:1]
        delta_phi = 0.1 * out[:, 1:2]

        return delta_mag, delta_phi
