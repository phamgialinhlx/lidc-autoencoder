import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

"""Adapted from https://github.com/milesial/Pytorch-UNet/tree/master/unet"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderUNet2D(nn.Module):
    def __init__(self,
            n_channels=1,
            n_classes=2,
            base_channel = 64,
            width_multiplier=1, \
            bilinear=True, \
            use_ds_conv=False):
        """A simple 3D Unet, adapted from a 2D Unet from https://github.com/milesial/Pytorch-UNet/tree/master/unet
        Arguments:
          n_channels = number of input channels; 3 for RGB, 1 for grayscale input
          n_classes = number of output channels/classes
          width_multiplier = how much 'wider' your UNet should be compared with a standard UNet
                  default is 1;, meaning 32 -> 64 -> 128 -> 256 -> 512 -> 256 -> 128 -> 64 -> 32
                  higher values increase the number of kernels pay layer, by that factor
          bilinear = use bilinear interpolation to upsample; if false, 3D convtranspose layers will be used instead
          use_ds_conv = if True, we use depthwise-separable convolutional layers. in my experience, this is of little help. This
                  appears to be because with 3D data, the vast vast majority of GPU RAM is the input data/labels, not the params, so little
                  VRAM is saved by using ds_conv, and yet performance suffers."""
        super(EncoderUNet2D, self).__init__()
        # _channels = (16, 32, 64, 128, 256, 512)
        _channels = (base_channel, base_channel * 2, base_channel * 4, base_channel * 8, base_channel * 16)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = [int(c * width_multiplier) for c in _channels]
        self.bilinear = bilinear
        self.convtype = DepthwiseSeparableConv2d if use_ds_conv else nn.Conv2d

        # self.inc = DoubleConv(n_channels, self.channels[0], conv_type=self.convtype)
        # self.down1 = Down(self.channels[0], self.channels[1], conv_type=self.convtype)
        # self.down2 = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down3 = Down(self.channels[2], self.channels[3], conv_type=self.convtype)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.channels[3], self.channels[4] // factor, conv_type=self.convtype)
        self.up1 = Up(self.channels[4], self.channels[3] // factor, bilinear)
        self.up2 = Up(self.channels[3], self.channels[2] // factor, bilinear)
        self.up3 = Up(self.channels[2], self.channels[1] // factor, bilinear)
        self.up4 = Up(self.channels[1], self.channels[0], bilinear)
        self.outc = OutConv(self.channels[0], n_classes)
        # if n_classes == 1:
        #   self.last_conv = nn.Sigmoid()
        # elif n_classes == 2:
        #   self.last_conv = nn.Softmax(dim=1)

    def forward(self, encoder, x):
        from IPython import embed
        x = encoder.conv_first(x)
        hs = [x]
        x_out = [x]
        for i_level in range(encoder.num_resolutions):
            for i_block in range(encoder.num_res_blocks):
                h = encoder.down[i_level].block[i_block](hs[-1])
                if len(encoder.down[i_level].attn) > 0:
                    h = encoder.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != encoder.num_resolutions - 1:
                hs.append(encoder.down[i_level].downsample(hs[-1]))
            x_out.append(hs[-1])
        x1, _, x2, _, x3 = x_out
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv2d, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv_type(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            conv_type(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv2d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, conv_type=conv_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

if __name__ == "__main__":
    # Instantiate UNet model
    n_channels = 1  # Assuming grayscale input
    n_classes = 1   # Number of output classes
    unet_model = EncoderUNet2D(
        n_channels,
        n_classes
    )

    # Generate a random input tensor
    input_tensor = torch.randn(1, 1, 128, 128, 128)

    # Forward pass
    output = unet_model(input_tensor)

    # Print the shape of the output
    print("Output shape:", output.shape)
