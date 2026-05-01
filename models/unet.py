# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


# Double Convolution Block
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


# Encoder Block
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.pool_conv(x)


# Decoder Block
class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dh = x2.size(2) - x1.size(2)
        dw = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# U-Net Architecture
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, features=64):
        super().__init__()
        f      = features
        factor = 2 if bilinear else 1
        self.inc   = DoubleConv(n_channels, f)
        self.down1 = Down(f,     f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16 // factor)
        self.up1   = Up(f * 16,  f * 8  // factor, bilinear)
        self.up2   = Up(f * 8,   f * 4  // factor, bilinear)
        self.up3   = Up(f * 4,   f * 2  // factor, bilinear)
        self.up4   = Up(f * 2,   f,                bilinear)
        self.outc  = nn.Conv2d(f, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)


# Combined Dice + BCE Loss
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce    = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce   = self.bce(logits, targets.float())
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        dice  = 1 - (2 * inter + self.smooth) / (
            probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth)
        return bce + dice.mean()
