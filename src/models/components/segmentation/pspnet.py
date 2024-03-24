import torch
from torch import nn
import torch.nn.functional as F

from resnet import ResNet183D, ResNet503D, ResNet1013D, ResNet1523D


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool3d(bin),  # Use AdaptiveAvgPool3d for 3D pooling
                nn.Conv3d(in_dim, reduction_dim, kernel_size=1, bias=False),  # Use Conv3d for 3D convolution
                nn.BatchNorm3d(reduction_dim),  # Use BatchNorm3d for 3D normalization
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='trilinear', align_corners=True))  # Use trilinear for 3D interpolation
        return torch.cat(out, 1)


class PSPNet3D(nn.Module):  # Renaming the class to PSPNet3D to signify the change to 3D inputs
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(PSPNet3D, self).__init__()
        assert layers in [18, 50, 101, 152]
        assert 512 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        if layers == 18:
            resnet = ResNet183D()
        elif layers == 50:
            resnet = ResNet503D()
        elif layers == 101:
            resnet = ResNet1013D()
        else:
            resnet = ResNet1523D()
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # Adjusting convolutions for 3D
        # Modify layers' convolutions to 3D convolutions

        fea_dim = 512
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv3d(fea_dim, 512, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv3d(1024, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=dropout),
                nn.Conv3d(256, classes, kernel_size=1)
            )


    def forward(self, x, y=None):
        x_size = x.size()
        # assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='trilinear', align_corners=True)  # Changed mode to trilinear for 3D

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='trilinear', align_corners=True)  # Changed mode to trilinear for 3D
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    
    # Modify input tensor
    input = torch.rand(1, 8, 32, 32, 32).cuda()  # Adjusted input size for 3D
    model = PSPNet3D(layers=18, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, pretrained=True).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('PSPNet', output.size())