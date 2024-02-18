import torch
import torch.nn as nn
from torchsummary import summary
import timm
from utils import exportWeight









class ConvBlock(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(ConvBlock, self).__init__()
        self.convBlock =  nn.Sequential(
                nn.Conv2d(input_dims, output_dims, kernel_size=3),
                nn.BatchNorm2d(output_dims),
                nn.ReLU()
        )
        
    def forward(self, x):
        x = self.convBlock(x)
        return x
    

class head(nn.Module):
    def __init__(self, ):
        super(head, self).__init__()
        self.sharedHead = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 1024),
            ConvBlock(1024, 2048),
        )
        self.regHead = nn.Linear(2048, 4)
        

    def forward(self, x):
        x = self.sharedHead(x)
        x = torch.flatten(x, start_dim=1)
        x = self.regHead(x)
        return x








# for test only
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = head().to(device)
    # summary(model, input_size=[(16, 256, 7, 7)])
    print(model)

    x = torch.rand((16, 256, 7, 7)).to(device)
    out = model(x)
    print(out.shape)

