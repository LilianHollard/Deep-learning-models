import torch
from torch import nn

#class Convolution + Batch + Normalization + Mish Activation 
class CBM(nn.Module):
    def __init__(self, c1, c2, k , s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p)
        self.batch = nn.BatchNorm2d(c2)
        self.mish = nn.Mish()
        
        self.block = nn.Sequential(self.conv, self.batch, self.mish)
    
    def forward(self, x):
        return self.block(x)
    
class CSPResNet(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.part_1 = nn.Sequential(CBM(c, c, 1, 1))
        
        self.conv1 = nn.Sequential(CBM(c, c, 1, 1))
        
        self.conv2 = nn.Sequential(CBM(c, int(c/2), 1, 1), 
                                   CBM(int(c/2), c, 1))
        
        self.conv3 = nn.Sequential(CBM(c, c, 1, 1))
    
    def forward(self, x):
        p_1 = self.part_1(x)
        
        p_2 = self.conv1(x)
        p_2_res = self.conv2(p_2)
        p_2 = p_2 + p_2_res
        
        p_2 = self.conv3(p_2)
        
        #return torch.cat((p_1, p_2), 0)
        return p_1 + p_2
    
def make_csp_layer(num_layer, c):
    layer = []
    for _ in range(num_layer):
        layer.append(CSPResNet(c))
    return layer

class CSPDarknet53(nn.Module):
    def __init__(self, c1=1, c2=1, k=1, s=1, p=0):
        super().__init__()
        
        self.focus = CBM(c1, c2, k, s, p)
        
        self.conv1 = nn.Sequential( nn.Conv2d(c2, c2, 3, s, 2), nn.Sequential(*make_csp_layer(1, c2)) )
        
        self.conv2 = nn.Sequential( nn.Conv2d(c2, c2*2, 3, s, 2), nn.Sequential(*make_csp_layer(2, c2*2)) )
        
        self.conv3 = nn.Sequential( nn.Conv2d(c2*2, c2*2*2, 3, s, 2), nn.Sequential(*make_csp_layer(8, c2*2*2)) )
        
        self.conv4 = nn.Sequential( nn.Conv2d(c2*2*2, c2*2*2*2, 3, s, 2), nn.Sequential(*make_csp_layer(8, c2*2*2*2)) )
        
        self.conv5 = nn.Sequential( nn.Conv2d(c2*2*2*2, c2*2*2*2*2, 3, s, 2), nn.Sequential(*make_csp_layer(4, c2*2*2*2*2)) )
        
    def forward(self, x):
        return self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(self.focus(x))))))
        

    
def main():
    CSPResnet_1 = CSPResNet(64).eval()
    CSPResnet_1(torch.randn(1, 64, 304, 304))
    
    csp = CSPDarknet53(3, 32)
    csp(torch.randn(1, 3, 608, 608))
    
    
    
if __name__ == "__main__":
    main()