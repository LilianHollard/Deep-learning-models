import torch
from torch import nn
import time
import onnx
from torch.profiler import profile, record_function, ProfilerActivity

def ConvBNSiLU(c1, c2, k, s, p):
    return nn.Sequential(
            nn.Conv2d(c1, c2, k, s, p),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            )

def BottleNeck_architecture(c1, c2):
    return nn.Sequential(
            ConvBNSiLU(c1, c2, 1, 1, 0),
            ConvBNSiLU(c2, c2, 3, 1, 1),
            )

class Focus(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv1 = ConvBNSiLU(c1, c2, 6, 2, 2)
        self.conv2 = ConvBNSiLU(c2, c2*2, 3, 2, 1)
        self.block = nn.Sequential(self.conv1, self.conv2)
    def forward(self, x):
        return self.block(x)

class Focus_pointsize(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 6, 2, 2, groups=c1),
            nn.Conv2d(c1, c2, 1, 1, 0),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

        #self.conv1 = ConvBNSiLU(c1, c2, 6, 2, 2)
        self.conv2 = ConvBNSiLU(c2, c2*2, 3, 2, 1)
        self.block = nn.Sequential(self.conv1, self.conv2)
    def forward(self, x):
        return self.block(x)

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
        self.focus = Focus(c1, c2)
        #self.focus = Focus_pointsize(c1, c2)
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
            nn.Linear(int(c2*16), 3),
            #nn.Linear(102400, 3),
            #nn.Linear(2304, 3)
        )

    def forward(self, x):
        t1 = time.time()
        x = self.focus(x)
        focus_time = time.time() - t1
        print(f"Focus time: {time.time() - t1}")
        
        t1 = time.time()
        x = self.firstC3(x)
        print(f"first c3 time: {time.time() - t1}")
        t1 = time.time()
        x = self.inter_conv_1(x)
        print(f"inter 1 time: {time.time() - t1}")
        t1 = time.time()
        x = self.secondC3(x)
        print(f"2 c3 time: {time.time() - t1}")
        t1 = time.time()
        x = self.inter_conv_2(x)
        print(f"inter time: {time.time() - t1}")
        t1 = time.time()
        x = self.thirdC3(x)
        print(f"3 c3 time: {time.time() - t1}")
        t1 = time.time()
        x = self.inter_conv_last(x)
        print(f"inter time: {time.time() - t1}")
        t1 = time.time()
        x = self.lastC3(x)
        print(f"last c3 time: {time.time() - t1}")
        
        #backbone = self.lastC3(self.inter_conv_last(self.thirdC3(self.inter_conv_2(self.secondC3(self.inter_conv_1(self.firstC3(self.focus(x))))))))
        #return self.classifier(backbone)
        return x, focus_time
        #

class batch_CSPDarknet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cut = int(224/8)
        
        self.conv1 = CSPDarknet(3, 16)
        
    def forward(self, x):
        #x = torch.split(x, self.num_batch, dim=2)
        x = x.unfold(2, self.cut, self.cut).unfold(3, self.cut, self.cut).contiguous().view(-1, 3, self.cut, self.cut)
        #print(x[0].shape)
        #for i in range(self.num_batch):
        #    _ = self.layer[i](x[i])
        y = torch.randn(1, 3)
        for batch in x:
            batch = batch.unsqueeze(0)
            #y += self.conv1(batch)
            self.conv1(batch)
        return y

def test_2():
    """    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})"""
    csp = batch_CSPDarknet().eval()#.to("cuda:1")

    #ImageNet size
    dummy = torch.randn(1, 3, 224, 224)#.to("cuda:1")
    for _ in range(3):
        csp(dummy)
    print("start")
    t_start = time.time()
    for _ in range(100):
        out, t1 = csp(dummy)
        time_ += t1
        
    print(f"CSP time: {time.time() - t_start}")
    
    """with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            csp(dummy)
            
    print(prof.key_averages().table(sort_by="cpu_memory_usage"))
    """
    #torch.onnx.export(csp, dummy, "csp_batch.onnx", export_params=True, do_constant_folding=True, input_names=['input'], output_names=['output'])


def test_1():
    print("CSPDarknet - Yolov5 modified version")
    device = torch.device("cuda")
    warmup = 3
        
    bn = BottleNeck(3, 64)
    CSP = CSPDarknet(3, 16)
    
    benchmark_size = [32, 64, 128, 256, 512]
    
    dummy = torch.randn(1, 3, 224,224)
    #
    
    CSP = CSP.to(device)
    dummy = dummy.to(device)
    CSP.eval()
    for _ in range(warmup):
        CSP(dummy)
    
    
    print("start")
    t_start = time.time()
    time_ = 0.0
    for _ in range(1000):
        out, t1 = CSP(dummy)
        time_ += t1
        
    print(f"CSP time: {time.time() - t_start}")
    
    print(f"focus time {time_ / 100}")
        
    """with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            CSP(dummy)
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))"""
    
        
    #print(f"CSP time: {time.time() - t_start}")

    #torch.onnx.export(CSP, dummy, "csp.onnx", export_params=True, do_constant_folding=True, input_names=['input'], output_names=['output'])


if __name__ == "__main__":
    #test_2()
    test_1()

    

