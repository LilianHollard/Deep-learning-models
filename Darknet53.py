import torch
from torch import nn

import time

class CBL(nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super().__init__()
        self.c = nn.Conv2d(c1, c2, k, s, p)
        self.b = nn.BatchNorm2d(c2)
        self.lr = nn.LeakyReLU()
        self.block = nn.Sequential(self.c, self.b, self.lr)
    
    def forward(self, x):
        return self.block(x)
    
class Focus(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv1 = CBL(c1, c2, 3, 1 ,1)
        self.conv2 = CBL(c2, c2*2, 3, 2, 1)
        self.block = nn.Sequential(self.conv1, self.conv2)
    def forward(self, x):
        return self.block(x)
    
class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = CBL(c, c, 1, 1, 0)
        self.conv2 = CBL(c, c, 3, 1, 1)
        self.block = nn.Sequential(self.conv1, self.conv2)
    def forward(self, x):
        y = self.block(x)
        return x + y

class Darknet_residual_block(nn.Module):
    def __init__(self, c):
        super().__init__()
        down_block_channel = int(c/2)
        self.down_block = CBL(c, down_block_channel, 1, 1, 0)
        self.up_block = CBL(down_block_channel, c, 3, 1, 1)
        self.res = Residual(c)
    
    def forward(self, x):
        x = self.down_block(x)
        x = self.up_block(x)
        return self.res(x)
    

def make_dark_residual_block(num_layer, c):
    layer = []
    for _ in range(num_layer):
        layer.append(Darknet_residual_block(c))
    return nn.Sequential(*layer)


class Darknet53(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        depth = 0.33
        width = 0.25
        c2 = int(c2*width)
        self.focus = Focus(c1, c2) #3 -> 32 -> 64
        c2 = c2*2 #c2 = c2 * 2 * 2 = 64 if c2 = 32 in param
        
        width = [3, 6, 9, 3]
        self.blocks = []
        self.blocks.append(self.focus)
        
        for repeat in width:   
            repeat = int(repeat * depth)     
            self.blocks.append(
                
                    nn.Sequential(make_dark_residual_block(repeat, c2), nn.Sequential(nn.Conv2d(c2, c2*2, 3, 2, 1)))
                
            )
            c2 = c2*2
        
        self.blocks.append(make_dark_residual_block(int(4*depth), c2))
        self.blocks = nn.Sequential(*self.blocks)
    def forward(self, x):
        return self.blocks(x)
        #return nn.Sequential(*self.blocks)(x)
        
        
        

def main():
    device = "cpu"
    input_ = torch.randn(1, 3, 640, 640).to(device)
    c = make_dark_residual_block(4, 32).to(device).eval()
    c(torch.randn(1, 32, 208, 208).to(device))
    
    """focus = Focus(3, 32).eval()
    x = focus(torch.randn(1, 3, 416, 416))
    print(x.shape)
    
    block = Darknet_residual_block(64).eval()
    x = block(x)
    print(x.shape)"""
    
    darknet = Darknet53(3, 32).to(device).eval()
    t1 = time.time()
    for _ in range(1000):
        x = darknet(input_)
    print(f"time darknet53 from yolov3 : {time.time() - t1}")
    print(x.shape)
    
    
if __name__ == "__main__":
    main()
    