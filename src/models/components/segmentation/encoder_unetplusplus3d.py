import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

"""Adapted from https://github.com/milesial/Pytorch-UNet/tree/master/unet"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderUNetPlusPlus3D(nn.Module):
    def __init__(self, in_channels, base_channels, n_classes, number_unet, 
                 conv_layer, norm_layer, activate_layer, transpconv_layer, 
                 conv_kwargs, norm_kwargs, activate_kwargs, transpconv_kwargs, **kwargs):
        super(EncoderUNetPlusPlus3D, self).__init__()
        self.n_classes = n_classes
        # number of unet
        self.number_unet = number_unet
        # name of convolution layer
        self.conv_layer = conv_layer if type(conv_layer) is not str else getattr(nn, conv_layer)
        # name of normalization layer
        self.norm_layer = norm_layer if type(norm_layer) is not str else getattr(nn, norm_layer)
        # name of activation function layer
        self.activate_layer = activate_layer if type(activate_layer) is not str else getattr(nn, activate_layer)
        # name of transposed convolution layer
        self.transpconv_layer = transpconv_layer if type(transpconv_layer) is not str else getattr(nn, transpconv_layer)
        
        # parameters of convolution layer
        self.conv_kwargs = conv_kwargs
        # parameters of normalization layer
        self.norm_kwargs = norm_kwargs
        # parameters of activation function layer
        self.activate_kwargs = activate_kwargs
        # parameters of transposed convolution layer
        self.transpconv_kwargs = transpconv_kwargs


        # down convolution modules
        self.down_conv_modules = [None] * number_unet
        # up convolution modules
        self.up_modules = [[None] * (i + 1) for i in range(number_unet)]
        # up convolution modules
        self.up_conv_modules = [[None] * (i + 1) for i in range(number_unet)]
        
       
        # # number of channels at each level
        self.channels = [base_channels] + [base_channels * (2 ** (i + 1)) for i in range(number_unet)]
            
        # initial modules for unetplusplus
        for i in range(number_unet):
            # i-th unet

            # i-th down convolution layer of all unets
            if i != 0 and i != 1:
                self.down_conv_modules[i] = self.get_conv_block(self.channels[i], self.channels[i + 1])

            # up layers of i-th unet
            for j in range(i + 1):
                # sum of channels after concat
                in_channels_conv = (j + 2) * self.channels[i - j]
                
                # j-th up layer of i-th unet
                self.up_modules[i][j], self.up_conv_modules[i][j] = \
                    self.get_up_block(self.channels[i + 1 - j], self.channels[i - j], in_channels_conv)            
        
            self.up_modules[i] = nn.ModuleList(self.up_modules[i])
            self.up_conv_modules[i] = nn.ModuleList(self.up_conv_modules[i])
        
        self.down_conv_modules = nn.ModuleList(self.down_conv_modules)
        self.up_modules = nn.ModuleList(self.up_modules)
        self.up_conv_modules = nn.ModuleList(self.up_conv_modules)
        
        # output convolution to n_classes
        self.output_conv = nn.Conv3d(base_channels, n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, encoder, input):
        x = [[None] * (i + 1) for i in range(self.number_unet + 1)]
        # input convolution layer to base_channels 
        x[0][0] = encoder.conv_first(input)
        blocks = encoder.conv_blocks
        for i in range(self.number_unet):
            # i-th down layer of all unets
            if i == 0 or i == 1:
                x[i + 1][0] = blocks[i].down(x[i][0])
                x[i + 1][0] = blocks[i].res(x[i + 1][0])
            else:
                x[i + 1][0] = self.down_conv_modules[i](x[i][0])
            # up layers of i-th unet
            for j in range(i + 1):
                # j-th up layer of i-th unet

                up_element = self.up_modules[i][j](x[i + 1][j])
                cat_elements = [up_element]
                for k in range(j + 1):
                    cat_elements.append(x[i - k][j - k])
                
                # up convolution after concat
                x[i + 1][j + 1] = self.up_conv_modules[i][j](torch.cat(cat_elements, dim=1))

        output = self.output_conv(x[self.number_unet][self.number_unet])
        output = self.sigmoid(output)
        return output        
                
    def get_conv_block(self, in_channels, out_channels, have_pool=True):
        if not have_pool:
            stride = 1
        else:
            stride = 2
            
        return nn.Sequential(
            self.conv_layer(in_channels, out_channels, stride = stride, **self.conv_kwargs),
            self.norm_layer(out_channels, **self.norm_kwargs),
            self.activate_layer(**self.activate_kwargs),
            self.conv_layer(out_channels, out_channels, stride = 1, **self.conv_kwargs),
            self.norm_layer(out_channels, **self.norm_kwargs),
            self.activate_layer(**self.activate_kwargs),
        )
    
    def get_up_block(self, in_channels, out_channels, in_channels_conv):
        up = self.transpconv_layer(in_channels, out_channels, **self.transpconv_kwargs)
        up_conv = nn.Sequential(
            self.get_conv_block(in_channels_conv, out_channels, have_pool=False),
            self.get_conv_block(out_channels, out_channels, have_pool=False)
        )
        return up, up_conv

# Example usage
if __name__ == "__main__":
    # Adjust configuration for 3D data accordingly
    config = {
        "in_channels": 1,
        "base_channels": 16,
        "n_classes": 1,
        "number_unet": 4,
        "conv_layer": "Conv3d",
        "norm_layer": "InstanceNorm3d",
        "activate_layer": "LeakyReLU",
        "transpconv_layer": "ConvTranspose3d",
        "conv_kwargs": {
            "kernel_size": 3,
            "padding": 1
        },
        "norm_kwargs": {
            "eps": 1e-05,
            "affine": True    
        },
        "activate_kwargs": {
            "negative_slope": 0.01,
            "inplace": True
        },
        "transpconv_kwargs": {
            "stride": 2,
            "kernel_size": 2,
            "bias": False
        }
    }

    # Instantiate the model
    model = EncoderUNetPlusPlus3D(**config)

    # Example input
    x = torch.randn(1, 1, 128, 128, 128)  # Adjust dimensions for your 3D data

    # Forward pass
    logits = model(x)

    print(model)
    # Check output shape
    print(logits.shape)
    print("logits range", logits.min().item(), logits.max().item())