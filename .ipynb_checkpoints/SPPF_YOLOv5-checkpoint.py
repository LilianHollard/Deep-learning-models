import time
import torch
import torch.nn as nn


class SPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(5, 1, padding=2)
        self.maxpool2 = nn.MaxPool2d(9, 1, padding=4)
        self.maxpool3 = nn.MaxPool2d(13, 1, padding=6)

    def forward(self, x):
        o1 = self.maxpool1(x)
        o2 = self.maxpool2(x)
        o3 = self.maxpool3(x)
        return torch.cat([x, o1, o2, o3], dim=1)


class SPPF(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(5, 1, padding=2)

    def forward(self, x):
        o1 = self.maxpool(x)
        o2 = self.maxpool(o1)
        o3 = self.maxpool(o2)
        return torch.cat([x, o1, o2, o3], dim=1)
    
class SPP_test(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(5, 1, padding=2)
        self.maxpool2 = nn.MaxPool2d(9, 1, padding=4)
        self.maxpool3 = nn.MaxPool2d(13, 1, padding=6)

    def forward(self, x):
        #o1 = self.maxpool1(x)
        #o2 = self.maxpool2(x)
        o3 = self.maxpool3(x)
        return torch.cat([x, o3], dim=1)

def main():
    input_tensor = torch.rand(1, int(1024*0.25), 20, 20) #int(1024*0.25) nano conversion (0.25 layer chanel width multiple compare to L Large model)
    spp = SPP()
    sppf = SPPF()
    output1 = spp(input_tensor)
    output2 = sppf(input_tensor)
    
    spp_test = SPP_test()
    out = spp_test(input_tensor)

    print(torch.equal(output1, output2))
    
    t_start = time.time()
    for _ in range(1000):
        spp_test(input_tensor)
    print(f"spp test time: {time.time() - t_start}")


    t_start = time.time()
    for _ in range(1000):
        spp(input_tensor)
    print(f"spp time: {time.time() - t_start}")

    t_start = time.time()
    for _ in range(1000):
        sppf(input_tensor)
    print(f"sppf time: {time.time() - t_start}")


if __name__ == '__main__':
    main()