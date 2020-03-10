import torch.nn as nn
import torch.nn.functional as F
                                                  

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dropout_value=0.15
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False), # kernel_size=3, in=3, out=16, RF=3, jump=1
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
        
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), # kernel_size=3, in=16, out=32, RF=5, jump=1
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), # kernel_size=3, in=32, out=64, RF=7, jump=1
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
        ) 
            

      # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # kernel_size=2, RF=8, jump=1
        
        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3, 3), padding=1, bias=False, dilation =2), # kernel_size=5, in=64, out=16, RF=16, jump=2
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
        
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False, dilation =1), # kernel_size=3, in=16, out=32, RF=20, jump=2
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, dilation =1), # kernel_size=3, in=32, out=64, RF=24, jump=2
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, dilation =1), # kernel_size=3, in=64, out=64, RF=26, jump=2
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
        ) 

        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2)  # kernel_size=2, RF=28, jump=2
            

        # CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False, groups=64), # kernel_size=3, in=64, out=128, RF=36, jump=4
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
        
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=1, bias=False), # kernel_size=1, in=128, out=64, RF=36, jump=4
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), # kernel_size=3, in=64, out=128, RF=44, jump=4
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False), # kernel_size=3, in=128, out=256, RF=52, jump=4
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value),
        ) 
            
        # TRANSITION BLOCK 3
        self.pool3 = nn.MaxPool2d(2, 2) # # kernel_size=2, RF=56, jump=4

        # OUTPUT BLOCK
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), # kernel_size=3, in=256, out=128, RF=72, jump=8
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value),
         
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(3, 3), padding=1, bias=False), 
        )
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        ) # output_size = 1
        

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False), #out=1
        ) # output_size = 1 

    def forward(self, x):
        x = self.convblock1(x)
        x = self.pool1(x)
        x = self.convblock2(x)
        x = self.pool2(x)
        x = self.convblock3(x)
        x = self.pool3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.gap(x)
        x = self.convblock5(x)
        x = x.view(-1, 10)
        return x
