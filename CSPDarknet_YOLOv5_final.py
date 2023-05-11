import torch
from torch import nn
import time

def ConvBNSiLU(c1, c2, k, s, p):
    return nn.Sequential(
            nn.Conv2d(c1, c2, k, s, p),
            nn.BatchNorm2d(c2),
            #nn.SiLU(),
            nn.ReLU()
            )

def BottleNeck_architecture(c1, c2):
    return nn.Sequential(
            ConvBNSiLU(c1, c2, 1, 1, 0),
            ConvBNSiLU(c2, c2, 3, 1, 1),
            )

class BottleNeck(nn.Module):
    def __init__(self,c1, c2, residual=True):
        super().__init__()
        self.inter_ = BottleNeck_architecture(c1, c2)

    def forward(self, input_):
        x = self.inter_(input_)
                
        return input_ +  x

def C3_architecture(c1, c2, k, s, p, num_layer=3):
    bottleneck_layer = []
    for i in range(num_layer):
        bottleneck_layer.append(BottleNeck(c1, c2))
    return nn.Sequential(*bottleneck_layer)

class C3(nn.Module):
    def __init__(self, c1, c2, k, s, p, num_layer=3):
        super().__init__()
        self.residual_block = ConvBNSiLU(c1, int(c2/2), k, s, p)
        self.first_conv = ConvBNSiLU(c1, int(c2/2), k, s, p)
        self.inter_block = C3_architecture(int(c2/2), int(c2/2), k, s, p, num_layer)
        self.out_conv = ConvBNSiLU(int(c2/2), c2, k, s, p)

    def forward(self, input_):
        x = self.residual_block(input_)
        input_ = self.first_conv(input_)
        input_ = self.inter_block(input_)
        
        x = x + input_

        return self.out_conv(x)

class CSPDarknet(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.focus = nn.Sequential(
                    ConvBNSiLU(c1, c2, 6, 2, 2),
                    ConvBNSiLU(c2, c2*2, 3, 2, 1)
                )
        
        self.firstC3 = C3(c2*2, c2*2, 1, 1, 0, int(3*0.33))
        self.inter_conv_1 = ConvBNSiLU(c2*2, c2*4, 3, 2, 1)
        self.secondC3 = C3(c2*4, c2*4, 1, 1, 0, int(6*0.33))
        self.inter_conv_2 = ConvBNSiLU(c2*4, c2*8, 3, 2, 1)
        self.thirdC3 = C3(c2*8, c2*8, 1, 1, 0, int(9*0.33))
        self.inter_conv_last = ConvBNSiLU(c2*8, c2*16, 3, 2, 1)
        self.lastC3 = C3(c2*16, c2*16, 1, 1, 0, int(3*0.33))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            #nn.Linear(int(c2*16)*16*4*4, 3)
            #nn.Linear(256, 3)
            #nn.Linear(int(c2*16)*4, 3),
            nn.Linear(102400, 1000),
            #nn.Linear(2304, 3)
        )

    def forward(self, x):
        backbone = self.lastC3(self.inter_conv_last(self.thirdC3(self.inter_conv_2(self.secondC3(self.inter_conv_1(self.firstC3(self.focus(x))))))))
        return self.classifier(backbone)
    
class batch_CSPDarknet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cut = int(640/8)
        
        self.conv1 = CSPDarknet(3, 16)
        
    def forward(self, x):
        #x = torch.split(x, self.num_batch, dim=2)
        x = x.unfold(2, self.cut, self.cut).unfold(3, self.cut, self.cut).contiguous().view(-1, 3, self.cut, self.cut)
        #print(x[0].shape)
        #for i in range(self.num_batch):
        #    _ = self.layer[i](x[i])
        y = torch.randn(1, 3).to("cuda:4")
        for batch in x:
            batch = batch.unsqueeze(0)
            y += self.conv1(batch)
        y = torch.div(y, len(x))
        return y
    
def test_2():
    csp = batch_CSPDarknet()#.to("cuda:1")
    dummy = torch.randn(1, 3, 80, 80)#.to("cuda:1")
    
    print("start")
    t_start = time.time()
    for _ in range(1000):
        csp(dummy)
        
    print(f"CSP time: {time.time() - t_start}")
            

def test_1():
    print("CSPDarknet - Yolov5 modified version")
    device = torch.device("cpu")
    warmup = 3
        
    bn = BottleNeck(3, 64)
    CSP = CSPDarknet(3, 16)
    
    benchmark_size = [32, 64, 128, 256, 512]
    
    dummy = torch.randn(1, 3, 640,640)
    
    
    CSP = CSP.to(device)
    dummy = dummy.to(device)
    CSP.eval()
    for _ in range(warmup):
        CSP(dummy)
    
    
    print("start")
    t_start = time.time()
    for _ in range(1000):
        CSP(dummy)
        
    print(f"CSP time: {time.time() - t_start}")
    

    
        
if __name__ == "__main__":
    test_2()

    

