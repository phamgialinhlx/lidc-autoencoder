import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnAct(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding, dilation=1, bias=False, act=True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv3d(in_channel, out_channel, kernel, stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm3d(out_channel)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        if x.shape[2:] != torch.Size([1, 1, 1]):
            x = self.bn(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    

class FeaturePyramidNet(nn.Module):
    def __init__(self, fpn_dim=256):
        self.fpn_dim = fpn_dim
        super(FeaturePyramidNet, self).__init__()
        self.fpn_in = nn.ModuleDict({
            'fpn_layer1': ConvBnAct(256, self.fpn_dim, 1, 1, 0),
            "fpn_layer2": ConvBnAct(512, self.fpn_dim, 1, 1, 0),
            "fpn_layer3": ConvBnAct(1024, self.fpn_dim, 1, 1, 0),
        })
        self.fpn_out = nn.ModuleDict({
            'fpn_layer1': ConvBnAct(self.fpn_dim, self.fpn_dim, 3, 1, 1),
            "fpn_layer2": ConvBnAct(self.fpn_dim, self.fpn_dim, 3, 1, 1),
            "fpn_layer3": ConvBnAct(self.fpn_dim, self.fpn_dim, 3, 1, 1),
        })

    def forward(self, pyramid_features):
        """
        """
        fpn_out = {}
        f = pyramid_features['resnet_layer4']
        fpn_out['fpn_layer4'] = f
        x = self.fpn_in['fpn_layer3'](pyramid_features['resnet_layer3'])
        f = F.interpolate(f, x.shape[2:], mode='trilinear', align_corners=False)
        f = x + f
        fpn_out['fpn_layer3'] = self.fpn_out['fpn_layer3'](f)
        x = self.fpn_in['fpn_layer2'](pyramid_features['resnet_layer2'])
        f = F.interpolate(f, x.shape[2:], mode='trilinear', align_corners=False)
        f = x + f
        fpn_out['fpn_layer2'] = self.fpn_out['fpn_layer2'](f)
        x = self.fpn_in['fpn_layer1'](pyramid_features['resnet_layer1'])
        f = F.interpolate(f, x.shape[2:], mode='trilinear', align_corners=False)
        f = x + f
        fpn_out['fpn_layer1'] = self.fpn_out['fpn_layer1'](f)
        return fpn_out

import torch.nn as nn
import torch

class PPM(nn.ModuleList):
    """Pooling Pyramid Module for 3D operations, adapted for PSPNet 3D.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate, for 3D.
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners, **kwargs):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool3d(pool_scale),
                    ConvBnAct(in_channels, self.channels, 1, 1, 0)
                )
            )
        
    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = torch.nn.functional.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode='trilinear',  # Adjust interpolation mode for 3D
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class UperHead3D(nn.Module):
    def __init__(self, 
                in_channel, 
                out_channel, 
                num_classes, 
                fpn_dim=256
            ):
            # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output

class SegmentationDecoder(nn.Module):
    def __init__(
        self, 
        decode_head,
        neck=None,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.decode_head = decode_head
        self.neck = neck

    def forward(self, x):
        if self.neck is not None:
            x = self.neck(x)
        x = self.decode_head(x)
        return x
    
if __name__ == "__main__":
    import numpy as np
    sample = np.random.rand(1, 64, 32, 32, 32)
    sample = torch.from_numpy(sample).float()
    print('Shape of a sample:', sample.shape)
    sample = sample.unsqueeze(0)
    print('Shape of a sample:', sample.shape)
    # neck = FeaturePyramidNet()
    decoder = SegmentationDecoder(
    )
    out = decoder(sample)
    print(out.shape)
    print(out)