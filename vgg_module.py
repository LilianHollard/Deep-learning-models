import torch
from torch import nn


#Very deep conv Net for large scale img recognition
#Architecture : 3x3 small receptive field, 1x1 conv filters as a linear transform
# each hidden layer followed by ReLU non linearity
#stride & padding = 1
#max_pool stride = 2
def conv_block(c1, c2):
    conv1 = nn.Conv2d(c1, c2, 3, 1)
    conv2 = nn.Conv2d(c2, c2, 3, 1)
    return nn.Sequential(conv1, nn.ReLU(), conv2, nn.ReLU())

#input 224x224
class VGG(nn.Module):
        
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(
            conv_block(3, 64), 
            self.max_pool,
            conv_block(64, 128),
            self.max_pool,
            conv_block(128, 256),
            nn.Conv2d(256, 256, 1, 1),
            nn.ReLU(),
            self.max_pool,
            
            conv_block(256, 512),
            nn.Conv2d(512, 512, 1, 1),
            nn.ReLU(),
            self.max_pool, 
            conv_block(512, 512),
            nn.Conv2d(512, 512, 1, 1),
            nn.ReLU(),
            self.max_pool
        )
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*3*3, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Softmax(),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.size(0), -1)
        x = self.head(x)
        
        return x
    

if __name__ == "__main__":
    my_vgg = VGG()
    
    dummy = torch.randn(size=(1, 3, 224, 224))
    
    my_vgg(dummy)
    print(my_vgg)
        
    
    
