import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

"""Adapted from https://github.com/milesial/Pytorch-UNet/tree/master/unet"""
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange
from einops_exts import check_shape, rearrange_many
from src.utils.model_utils import exists, is_odd, default, prob_mask_like

class AttnUNet3D(nn.Module):
    def __init__(self,
            n_channels,
            n_classes,
            base_channel = 16,
            width_multiplier=1, \
            attn_heads=8,
            attn_dim_head=32,
            use_sparse_linear_attn=True,
            trilinear=True, \
            use_ds_conv=False,
            *args, **kwargs):
        """A simple 3D Unet, adapted from a 2D Unet from https://github.com/milesial/Pytorch-UNet/tree/master/unet
        Arguments:
          n_channels = number of input channels; 3 for RGB, 1 for grayscale input
          n_classes = number of output channels/classes
          width_multiplier = how much 'wider' your UNet should be compared with a standard UNet
                  default is 1;, meaning 32 -> 64 -> 128 -> 256 -> 512 -> 256 -> 128 -> 64 -> 32
                  higher values increase the number of kernels pay layer, by that factor
          trilinear = use trilinear interpolation to upsample; if false, 3D convtranspose layers will be used instead
          use_ds_conv = if True, we use depthwise-separable convolutional layers. in my experience, this is of little help. This
                  appears to be because with 3D data, the vast vast majority of GPU RAM is the input data/labels, not the params, so little
                  VRAM is saved by using ds_conv, and yet performance suffers."""
        super(AttnUNet3D, self).__init__()
        _channels = (base_channel, base_channel * 2, base_channel * 4, base_channel * 8, base_channel * 16)
        
        def temporal_attn(dim): return EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
            dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb))

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = [int(c * width_multiplier) for c in _channels]
        self.trilinear = trilinear
        self.convtype = DepthwiseSeparableConv3d if use_ds_conv else nn.Conv3d

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        self.downs = nn.ModuleList([])
        for i in range(3):
            if i == 0 or i == 1:
                continue
            in_channels = self.channels[i]
            out_channels = self.channels[i + 1]
            self.downs.append(nn.ModuleList([
                nn.MaxPool3d(2),
                DoubleConv(in_channels, out_channels, conv_type=self.convtype),
                Residual(PreNorm(out_channels, SpatialLinearAttention(
                        out_channels, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(out_channels, temporal_attn(out_channels))),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
            ]))
        factor = 2 if trilinear else 1
        self.downs.append(nn.ModuleList([
            nn.MaxPool3d(2),
            DoubleConv(self.channels[3], self.channels[4] // factor, conv_type=self.convtype),
            Residual(PreNorm(self.channels[4] // factor, SpatialLinearAttention(
                    self.channels[4] // factor, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
            Residual(PreNorm(self.channels[4] // factor, temporal_attn(self.channels[4] // factor))),
            nn.Conv3d(self.channels[4] // factor, self.channels[4] // factor, kernel_size=3, padding=1)
        ]))
        self.ups = nn.ModuleList([])
        
        for i in range(4, 1, -1):
            in_channels = self.channels[i]
            out_channels = self.channels[i - 1] // factor
            self.ups.append(nn.ModuleList([
                Up(in_channels, out_channels, trilinear),
                Residual(PreNorm(out_channels, SpatialLinearAttention(
                        out_channels, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(out_channels, temporal_attn(out_channels))),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
            ]))
        self.ups.append(nn.ModuleList([
            Up(self.channels[1], self.channels[0], trilinear),
            Residual(PreNorm(self.channels[0], SpatialLinearAttention(
                        self.channels[0], heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
            Residual(PreNorm(self.channels[0], temporal_attn(self.channels[0]))),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        ]))
        self.outc = OutConv(self.channels[0], n_classes)
        # self.conv_last = DoubleConv(n_classes, n_classes, conv_type=self.convtype)

    def forward(self, encoder, x, focus_present_mask=None, prob_focus_present=0.):
        batch, device = x.shape[0], x.device

        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
            (batch,), prob_focus_present, device=device))

        x = encoder.conv_first(x)
        h = [x]
        for block in encoder.conv_blocks:
            x = block.down(x)
            x = block.res(x)
            h.append(x)
        for maxpool, conv, spatial_attn, temporal_attn, conv2 in self.downs:
            x = maxpool(x)
            x = conv(x)
            x = spatial_attn(x)
            x = temporal_attn(x, focus_present_mask=focus_present_mask)
            x = conv2(x)
            h.append(x)
        h.pop()
        for i, (up, spatial_attn, temporal_attn, conv) in enumerate(self.ups):
            x = up(x, h.pop())
            if i <= 2:
                x = spatial_attn(x)
                x = temporal_attn(x, focus_present_mask=focus_present_mask)
            x = conv(x)
        logits = self.outc(x)
        return logits

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance /
                                                        max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x,
        pos_bias=None,
        focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones(
                (n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(
            qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv_type(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            conv_type(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self, 
        in_channels,
        out_channels, 
        use_sparse_linear_attn=True, 
        attn_heads=8,
        conv_type=nn.Conv3d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, conv_type=conv_type),
            Residual(PreNorm(out_channels, SpatialLinearAttention(
                    out_channels, heads=attn_heads))) if use_sparse_linear_attn else nn.Identity(),
            Residual(PreNorm(out_channels, temporal_attn(out_channels)))
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
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
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

if __name__ == "__main__":
    # Instantiate UNet model
    n_channels = 1  # Assuming grayscale input
    n_classes = 1   # Number of output classes
    unet_model = AttnUNet3D(
        n_channels,
        n_classes
    )

    # Generate a random input tensor
    input_tensor = torch.randn(1, 1, 128, 128, 128)

    # Forward pass
    output = unet_model(input_tensor)

    # Print the shape of the output
    print("Output shape:", output.shape)
    print("Ouput max", output.max())
    print("Ouput min", output.min())
