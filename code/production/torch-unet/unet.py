import torch
import torch.nn as nn
import torch.nn.functional as F

def he_normal_init(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(out_channels)
        )
        self.conv.apply(he_normal_init)

    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        inter_channels = in_channels // 4
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv2d(1, 1, kernel_size=1)

        self.act = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

        self.theta.apply(he_normal_init)
        self.phi.apply(he_normal_init)
        self.psi.apply(he_normal_init)

    def forward(self, x, g):
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        f = self.act(theta_x + phi_g)
        psi_f = self.psi(f)
        rate = self.sigmoid(psi_f)
        return x * rate

class UNet(nn.Module):
    def __init__(self, input_label_channel_count=1, layer_count=64, countbranch=False):
        super(UNet, self).__init__()
        self.input_label_channel_count = input_label_channel_count
        self.countbranch = countbranch

        self.down1 = ConvBlock(1, 1 * layer_count)
        self.down2 = ConvBlock(1 * layer_count, 2 * layer_count)
        self.down3 = ConvBlock(2 * layer_count, 4 * layer_count)
        self.down4 = ConvBlock(4 * layer_count, 8 * layer_count)
        self.bottom = ConvBlock(8 * layer_count, 16 * layer_count)

        self.att4 = AttentionBlock(8 * layer_count)
        self.att3 = AttentionBlock(4 * layer_count)
        self.att2 = AttentionBlock(2 * layer_count)
        self.att1 = AttentionBlock(1 * layer_count)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(16 * layer_count, 8 * layer_count)
        )

        self.conv4 = ConvBlock(16 * layer_count, 8 * layer_count)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(8 * layer_count, 4 * layer_count)
        )

        self.conv3 = ConvBlock(8 * layer_count, 4 * layer_count)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(4 * layer_count, 2 * layer_count)
        )

        self.conv2 = ConvBlock(4 * layer_count, 2 * layer_count)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(2 * layer_count, 1 * layer_count)
        )

        self.conv1 = ConvBlock(2 * layer_count, 1 * layer_count)

        self.output_seg = nn.Sequential(
            nn.Conv2d(1 * layer_count, input_label_channel_count, kernel_size=1),
            nn.Sigmoid()
        )

        if self.countbranch:
            self.output_dens = nn.Conv2d(1 * layer_count, input_label_channel_count, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = F.max_pool2d(d1, 2)

        d2 = self.down2(p1)
        p2 = F.max_pool2d(d2, 2)

        d3 = self.down3(p2)
        p3 = F.max_pool2d(d3, 2)

        d4 = self.down4(p3)
        p4 = F.max_pool2d(d4, 2)

        m = self.bottom(p4)

        u4 = self.up4(m)
        a4 = self.att4(d4, u4)
        u4 = torch.cat([u4, a4], dim=1)
        u4 = self.conv4(u4)

        u3 = self.up3(u4)
        a3 = self.att3(d3, u3)
        u3 = torch.cat([u3, a3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        a2 = self.att2(d2, u2)
        u2 = torch.cat([u2, a2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        a1 = self.att1(d1, u1)
        u1 = torch.cat([u1, a1], dim=1)
        u1 = self.conv1(u1)

        seg_out = self.output_seg(u1)

        if self.countbranch:
            dens_out = self.output_dens(u1)
            return seg_out, dens_out
        else:
            return seg_out
