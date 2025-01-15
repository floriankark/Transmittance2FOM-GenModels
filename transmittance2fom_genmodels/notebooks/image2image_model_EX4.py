import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True, use_dropout=False):
        super().__init__()
        self.downsample = downsample
        self.use_dropout = use_dropout

        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        else:
            self.conv1 = nn.ConvTranspose2d(
                in_channels, out_channels, 4, 2, 1, bias=False
            )

        self.norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5) if use_dropout else None

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        if self.use_dropout and self.dropout is not None:
            x = self.dropout(x)
        return F.leaky_relu(x, 0.2) if self.downsample else F.relu(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.e1 = nn.Conv2d(1, 64, 4, 2, 1)  # 1 channel input (transmittance)
        self.e2 = UNetBlock(64, 128)
        self.e3 = UNetBlock(128, 256)
        self.e4 = UNetBlock(256, 512)
        self.e5 = UNetBlock(512, 512)
        self.e6 = UNetBlock(512, 512)
        self.e7 = UNetBlock(512, 512)

        # Bottleneck
        self.bottleneck = nn.Conv2d(512, 512, 4, 2, 1)

        # Decoder
        self.d1 = UNetBlock(512, 512, False, True)
        self.d2 = UNetBlock(1024, 512, False, True)
        self.d3 = UNetBlock(1024, 512, False, True)
        self.d4 = UNetBlock(1024, 512, False)
        self.d5 = UNetBlock(1024, 256, False)
        self.d6 = UNetBlock(512, 128, False)
        self.d7 = UNetBlock(256, 64, False)

        self.final = nn.ConvTranspose2d(128, 3, 4, 2, 1)  # 3 channel output (FOM)

    def forward(self, x):
        # Encoder
        e1 = F.leaky_relu(self.e1(x), 0.2)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)

        # Bottleneck
        b = F.leaky_relu(self.bottleneck(e7), 0.2)

        # Decoder with skip connections
        d1 = self.d1(b)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))

        out = torch.tanh(self.final(torch.cat([d7, e1], 1)))
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1))
