import torch
from torch import nn

class yolov1():
    class C1(nn.Module):
        #C1 = input_filter
        #C2 = hidden_units
        #k = Kernel
        #s = stride
        def __init__(self, c1, c2, p=None):
            super().__init__()
            #YOLOv1 Convolution -> Combinaison of 3x3 convolution, 1x1 and focus conv at the start of 7x7
            self.conv1 = nn.Conv2d(c1, c2, 7, 2)
            self.conv2 = nn.Conv2d(c2, c2*3, 3, 1)
            
            self.c1 = nn.Sequential(self.conv1, nn.MaxPool2d(2,2), self.conv2, nn.MaxPool2d(2,2))
        
        def forward(self, x: torch.Tensor):
            return self.c1(x)
    
    class C2(nn.Module):
        #C1 = input_filter
        #C2 = hidden_units
        #k = Kernel
        #s = stride
        def __init__(self, c1, c2, p=None):
            super().__init__()
            
            self.c2 = nn.Sequential(
                nn.Conv2(c1, c2, 1, 1),
                nn.Conv2(c2, c2*2, 3, 1),
                nn.Conv2(c2*2, c2*2, 1, 1),
                nn.Conv2(c2*4, c2*4, 3, 1),
                nn.MaxPool2d(2, 2)
            )
            
            
        def forward(self, x: torch.Tensor):
            return self.c2(x)
        
    class C3(nn.Module):
        #C1 = input_filter
        #C2 = hidden_units
        #k = Kernel
        #s = stride
        def __init__(self, c1, c2, p=None):
            super().__init__()
            
            self.c3 = nn.Sequential(
                nn.Conv2(c1, c2, 1, 1),
                nn.Conv2(c2, c2*2, 3, 1),
            )
            
            self.block = nn.Sequential(
                self.c3, self.c3, self.c3, self.c3
            )
            
            
        def forward(self, x: torch.Tensor):
            return self.block(x)
    
    class Backbone(nn.Module):
        #C1 = input_filter
        #C2 = hidden_units
        #k = Kernel
        #s = stride
        def __init__(self, c1, c2, p=None):
            super().__init__()
            
            self.backbone = nn.Sequential(
                C1(),
                C2(),
                C3(),
            )
    
class yolov3():
    class C1(nn.Module):
        def __init__(self, c1, c2, k ,s, padding=0):
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, padding),
                nn.BatchNorm2d(c2),
                nn.LeakyReLU(),
            )
        def forward(self, x):
            return self.conv_block(x)
    
    
    class Residual(nn.Module):
        def __init__(self, c1):
            super().__init__()
            #pas de changement de dimension
            
            self.conv_block = nn.Sequential(
                yolov3.C1(c1, int(c1/2), 1, 1),
                yolov3.C1(int(c1/2), c1, 3, 1, 1)
            )
        
        def forward(self, x):
            y = self.conv_block(x)
            print(y.shape)
            print(x.shape)
            out = x + y
            return out
            
            
    
class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            yolov3.C1(3, 32, 3, 1),
            yolov3.C1(32, 64, 3, 2),
        )
        
        layers = []
        channels = 64
        for i in range(0, 5):
            layers.append(yolov3.Residual(64))
        
        self.conv2 = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv2(self.conv1(x))
            
if __name__ == "__main__":
    #yolo = yolov3()
    darknet = Darknet53()
    
    dummy = torch.randn(size=(1, 3, 416, 416))
    
    test_image = dummy[0]
    
    output = darknet(dummy)
    
    print(output.shape)