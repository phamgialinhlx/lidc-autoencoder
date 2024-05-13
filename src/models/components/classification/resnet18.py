import torch
from torch import nn
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 2, *args, **kwargs) -> None:
        super().__init__()
        self.resnet = resnet18(weights=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_ftrs, num_classes)  # Assuming two classes

    def forward(self, x):
        from IPython import embed
        embed()
        return self.resnet(x)
    
if __name__ == "__main__":
    x = torch.randn(20, 784, 2, 2)
    model = ResNet18()
    y = model(x)
    print(y.shape)
