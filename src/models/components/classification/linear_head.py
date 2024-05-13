import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearHead(nn.Module):
    def __init__(self, in_features, num_classes, **kwargs):
        super(LinearHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
        # return F.softmax(x, dim=1)
    
if __name__ == "__main__":
    x = torch.randn(20, 8, 32, 32, 32)
    in_features = 8 * 32 * 32 * 32
    print(in_features)
    head = LinearHead(in_features, 2)
    y = head(x)
    print(y)