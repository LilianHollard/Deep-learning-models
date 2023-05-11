import torch
from torch import nn
from utils_model import config, nano_scale
from torchsummary import summary


class CNNBlock(nn.Module):
    def __init__(self, c1, c2, bn_act = True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(c2)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act
    
    def forward(self, x):
        x = self.leaky(self.bn(self.conv(x))) if self.use_bn_act else self.conv(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(int(num_repeats * nano_scale)):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels//2, kernel_size=1),
                    CNNBlock(channels//2, channels, kernel_size=3, padding=1) #ours implementation -> go up with 1x1 conv
                )
            ]
        self.use_residual= use_residual
        self.num_repeats= num_repeats
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
            
        return x
        
###
#For the COCO dataset with 80 cat :
## Each scale provides an output tensor with a shape of N x N x [3 x ((4+1) + 80)] where 4 + 1 is box cordinate + class
## and 3 indicate the number of boxes per cell.

class ScalePrediction(nn.Module):
    def __init__(self, c1, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(c1, 2*c1, kernel_size=3, padding=1),
            CNNBlock(2*c1, 3 * (num_classes + 5), bn_act=False, kernel_size=1) #[class, x, y, w, h]
        )
        
        self.num_classes = num_classes
        
    def forward(self, x):
        return self.pred(x).reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        self.layers = self._create_conv_layers()
        
    def forward(self, x):
        outputs = []
        route_connections = []
        
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            
            x = layer(x)
            
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        
        return outputs
    
    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        
        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    
                    CNNBlock(
                        in_channels, 
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                    
                )
                
                in_channels = out_channels
            
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))
                
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels//2, kernel_size=1),
                        ScalePrediction(in_channels//2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels // 2
                    
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3 
                    
        return layers

def test():
    num_classes = 20
    model = YOLOv3(num_classes=num_classes)
    
    img_size = 416
    x = torch.randn((2, 3, img_size, img_size))
    summary(model, x)
    model.to("cpu")
    out = model(x)
    assert out[0].shape == (2, 3, img_size//32, img_size//32, 5 + num_classes)
    assert out[1].shape == (2, 3, img_size//16, img_size//16, 5 + num_classes)
    assert out[2].shape == (2, 3, img_size//8, img_size//8, 5 + num_classes)

test()