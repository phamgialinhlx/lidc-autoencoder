from typing import List

from torch import nn
import torch.nn.functional as F
import torch


class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GroupNorm(nn.Module):
    def __init__(self, num_channels: int, num_groups: int = 32, eps=1e-6) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = GroupNorm(in_channels)
        self.swish1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

        self.norm2 = GroupNorm(out_channels)
        self.swish2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)

        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, 1, stride=1, padding=0
            )
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        h = x

        h = self.norm1(h)
        h = self.swish1(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.swish2(h)
        h = self.conv2(h)

        return self.nin_shortcut(x) + h


class UpSample(nn.Module):
    """
    ## Up-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution mapping
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Up-sample by a factor of $2$
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        # Apply convolution
        return self.conv(x)


class DownSample(nn.Module):
    """
    ## Down-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution with stride length of $2$ to down-sample by a factor of $2$
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Add padding
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        # Apply convolution
        return self.conv(x)


class AttnBlock(nn.Module):
    """
    ## Attention block
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # Group normalization
        self.norm = GroupNorm(channels)
        # Query, key and value mappings
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        # Final $1 \times 1$ convolution layer
        self.proj_out = nn.Conv2d(channels, channels, 1)
        # Attention scaling factor
        self.scale = channels**-0.5

    def forward(self, x: torch.Tensor):
        """
        :param x: is the tensor of shape `[batch_size, channels, height, width]`
        """
        # Normalize `x`
        x_norm = self.norm(x)
        # Get query, key and vector embeddings
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        # Reshape to query, key and vector embeedings from
        # `[batch_size, channels, height, width]` to
        # `[batch_size, channels, height * width]`
        b, c, h, w = q.shape
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        # Compute $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$
        attn = torch.einsum("bci,bcj->bij", q, k) * self.scale
        attn = F.softmax(attn, dim=2)

        # Compute $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$
        out = torch.einsum("bij,bcj->bci", attn, v)

        # Reshape back to `[batch_size, channels, height, width]`
        out = out.view(b, c, h, w)
        # Final $1 \times 1$ convolution layer
        out = self.proj_out(out)

        # Add residual connection
        return x + out


class Encoder(nn.Module):
    """
    ## Encoder module
    """

    def __init__(
        self,
        *,
        channels: int,
        channel_multipliers: List[int],
        n_resnet_blocks: int,
        attn_resolutions,
        img_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = True,
        **kwargs,
    ):
        """
        :param channels: is the number of channels in the first convolution layer
        :param channel_multipliers: are the multiplicative factors for the number of channels in the
            subsequent blocks
        :param n_resnet_blocks: is the number of resnet layers at each resolution
        :param img_channels: is the number of channels in the image
        :param z_channels: is the number of channels in the embedding space
        """
        super().__init__()
        self.num_resolutions = len(channel_multipliers)
        self.num_res_blocks = n_resnet_blocks
        # Number of blocks of different resolutions.
        # The resolution is halved at the end each top level block
        n_resolutions = len(channel_multipliers)

        # Initial $3 \times 3$ convolution layer that maps the image to `channels`
        self.conv_first = nn.Conv2d(img_channels, channels, 3, stride=1, padding=1)

        curr_res = resolution
        # Number of channels in each top level block
        channels_list = [m * channels for m in [1] + channel_multipliers]

        # List of top-level blocks
        self.down = nn.ModuleList()
        # Create top-level blocks
        for i in range(n_resolutions):
            # Each top level block consists of multiple ResNet Blocks and down-sampling
            resnet_blocks = nn.ModuleList()
            # Add ResNet Blocks
            attn = nn.ModuleList()
            # Add Attention Blocks
            for _ in range(n_resnet_blocks):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i + 1]))
                channels = channels_list[i + 1]
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(channels))
            # Top-level block
            down = nn.Module()
            down.block = resnet_blocks
            down.attn = attn
            # Down-sampling at the end of each top level block except the last
            if i != n_resolutions - 1:
                down.downsample = DownSample(channels)
                curr_res = curr_res // 2
            else:
                down.downsample = nn.Identity()
            #
            self.down.append(down)

        # Final ResNet blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels)
        self.mid.block_2 = ResnetBlock(channels, channels)

        # Map to embedding space with a $3 \times 3$ convolution
        self.norm_out = GroupNorm(channels)
        self.swish = Swish()
        self.conv_out = nn.Conv2d(channels, 2 * z_channels if double_z else z_channels, 3, stride=1, padding=1)

    def forward(self, img: torch.Tensor):
        """
        :param img: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """

        # Map to `channels` with the initial convolution
        x = self.conv_first(img)

        hs = [x]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        x = hs[-1]

        # Final ResNet blocks with attention
        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)

        # Normalize and map to embedding space
        x = self.norm_out(x)
        x = self.swish(x)
        x = self.conv_out(x)

        #
        return x


class Decoder(nn.Module):
    """
    ## Decoder module
    """

    def __init__(
        self,
        *,
        channels: int,
        channel_multipliers: List[int],
        n_resnet_blocks: int,
        img_channels: int,
        attn_resolutions,
        z_channels: int,
        resolution: int,
        **kwargs,
    ):
        """
        :param channels: is the number of channels in the final convolution layer
        :param channel_multipliers: are the multiplicative factors for the number of channels in the
            previous blocks, in reverse order
        :param n_resnet_blocks: is the number of resnet layers at each resolution
        :param img_channels: is the number of channels in the image
        :param z_channels: is the number of channels in the embedding space
        """
        super().__init__()

        self.num_resolutions = len(channel_multipliers)
        self.num_res_blocks = n_resnet_blocks

        # Number of blocks of different resolutions.
        # The resolution is halved at the end each top level block
        num_resolutions = len(channel_multipliers)
        # Number of channels in each top level block, in the reverse order
        channels_list = [m * channels for m in channel_multipliers]

        # Number of channels in the  top-level block
        channels = channels_list[-1]
        curr_res = resolution // 2**(self.num_resolutions - 1)

        # Initial $3 \times 3$ convolution layer that maps the embedding space to `channels`
        self.conv_in = nn.Conv2d(z_channels, channels, 3, stride=1, padding=1)

        # ResNet blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels)
        self.mid.block_2 = ResnetBlock(channels, channels)

        # List of top-level blocks
        self.up = nn.ModuleList()
        # Create top-level blocks
        for i in reversed(range(num_resolutions)):
            # Each top level block consists of multiple ResNet Blocks and up-sampling
            resnet_blocks = nn.ModuleList()
            # Add ResNet Blocks
            attn = nn.ModuleList()
            # Add Attention Blocks
            for _ in range(n_resnet_blocks + 1):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i]))
                channels = channels_list[i]
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(channels))
            # Top-level block
            up = nn.Module()
            up.block = resnet_blocks
            up.attn = attn
            # Up-sampling at the end of each top level block except the first
            if i != 0:
                up.upsample = UpSample(channels)
                curr_res = curr_res * 2
            else:
                up.upsample = nn.Identity()
            # Prepend to be consistent with the checkpoint
            self.up.insert(0, up)

        # Map to image space with a $3 \times 3$ convolution
        self.norm_out = GroupNorm(channels)
        self.swish_out = Swish()
        self.conv_out = nn.Conv2d(
            channels, img_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z: torch.Tensor):
        """
        :param z: is the embedding tensor with shape `[batch_size, z_channels, z_height, z_height]`
        """

        # Map to `channels` with the initial convolution
        h = self.conv_in(z)

        # ResNet blocks with attention
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Top-level blocks
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Normalize and map to image space
        h = self.norm_out(h)
        h = self.swish_out(h)
        img = self.conv_out(h)

        #
        return img


if __name__ == "__main__":
    IMG_CHANNELS = 3
    CHANNEL_MULTIPLIERS = [1, 2, 4]
    Z_CHANNELS = 64

    encoder = Encoder(
        channels=64,
        channel_multipliers=CHANNEL_MULTIPLIERS,
        attn_resolutions=[16],
        resolution=64,
        n_resnet_blocks=2,
        img_channels=IMG_CHANNELS,
        z_channels=Z_CHANNELS,
    )
    decoder = Decoder(
        channels=64,
        channel_multipliers=CHANNEL_MULTIPLIERS,
        attn_resolutions=[16],
        resolution=64,
        n_resnet_blocks=2,
        img_channels=IMG_CHANNELS,
        z_channels=2 * Z_CHANNELS,
    )
    img = torch.randn(4, IMG_CHANNELS, 128, 128)
    print(encoder(img).shape)
    print(decoder(encoder(img)).shape)
