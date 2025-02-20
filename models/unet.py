import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.decoder1 = self.upconv_block(1024, 512)
        self.decoder2 = self.upconv_block(512, 256)
        self.decoder3 = self.upconv_block(256, 128)
        self.decoder4 = self.upconv_block(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))
        dec1 = self.decoder1(bottleneck)
        dec1 = torch.cat((enc4, dec1), dim=1)
        dec2 = self.decoder2(dec1)
        dec2 = torch.cat((enc3, dec2), dim=1)
        dec3 = self.decoder3(dec2)
        dec3 = torch.cat((enc2, dec3), dim=1)
        dec4 = self.decoder4(dec3)
        dec4 = torch.cat((enc1, dec4), dim=1)
        return torch.sigmoid(self.final_conv(dec4))
