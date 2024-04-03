import torch
import torch.nn as nn
import torch.nn.functional as F

class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = SamePadConv3d(in_planes, planes, kernel_size=3, stride=stride, padding_type='replicate', bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = SamePadConv3d(planes, planes, kernel_size=3, stride=1, padding_type='replicate', bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                SamePadConv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, padding_type='replicate', bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, block=BasicBlock3D, num_blocks=[2, 2, 2, 2], num_channels=8, num_classes=10):
        super(ResNet3D, self).__init__()
        self.in_planes = 64

        self.conv1 = SamePadConv3d(num_channels, 64, kernel_size=3, stride=1, padding_type='replicate', bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool3d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet183D():
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes=2)


def ResNet503D():
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3], num_classes=2)

def ResNet1013D():
    return ResNet3D(BasicBlock3D, [3, 4, 23, 3], num_classes=2)

def ResNet1523D():
    return ResNet3D(BasicBlock3D, [3, 8, 36, 3], num_classes=2)

# Testing the network
if __name__ == "__main__":
    net = ResNet183D()
    x = torch.randn(20, 8, 32, 32, 32)
    y = net(x)
    print(y.size())

