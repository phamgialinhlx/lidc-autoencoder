import torch
from torch import nn
from positional_encodings.torch_encodings import PositionalEncoding3D

class DETR(nn.Module):
    def __init__(self,
        num_classes=2, 
        in_channels=64,
        hidden_dim=256, 
        nheads=8,
        num_encoder_layers=6, 
        num_decoder_layers=6, 
        threshold=0.7,
        **kwargs,

    ):
        super(DETR, self).__init__()
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.conv = nn.Conv3d(in_channels, hidden_dim, 3)

        # Downsample layers
        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        h = self.conv(inputs)
        # print(h.shape)
        h = self.downsample(h)
        # print(h.shape)
        B, C, T, H, W = h.shape

        # construct positional encodings
        pos_model = PositionalEncoding3D(T * H * W).to(h.device)
        pos = pos_model(h)
        pos = pos.view(B, T, H, W, C)[0].flatten(0, 2).unsqueeze(1)
        print(self.query_pos.unsqueeze(1).shape)
        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        h = self.linear_class(h)
        probas = h.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > self.threshold
        if keep.sum() == 0:
            return torch.Tensor([[probas.min(), 1 - probas.min()]]).to(h.device)
        else:
            keep_probas = probas[keep]
            # logits = keep_probas.max()
            return torch.Tensor([[1 - keep_probas.max(), keep_probas.max()]]).to(h.device)
            
if __name__ == "__main__":
    net = DETR()
    x = torch.randn(1, 64, 32, 32, 32)
    y = net(x)
    print(y.size())
