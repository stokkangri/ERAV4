import torch.nn.functional as F
import torch.nn as nn

dropout_value = 0.1

def calc_rf (rf_in, kernel_size, jump_in):
    # jump_out = jump_in * stride
    # jump_in_n = jump_out_n-1
    #nin    rin jin s   p   nout    rout    jout
    # 32	1	1	1	1	32	    3	    1
    # 32	3	1	2	0	15	    5	    2
    # 15	5	2	1	0	13	    9	    2
    # 13	9	2	2	1	7	    13	    4
    # 7	    13	4	1	1	7	    21	    4
    # 7	    21	4	2	0	3	    29	    8
    # 3	    29	8	1	0	1	    45	    8
    return rf_in + (kernel_size - 1) * jump_in

def calc_out_size(in_size, padding, stride, kernel_size):
    # nin: number of input features
    # nout : number of output features
    # k : conv kernel size
    # p : padding size
    # s : conv stride size
    return 1 + (in_size + 2 * padding - kernel_size) / stride

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Input Block
        self.input = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(16)  #1
        )
        
        # CONVOLUTION BLOCK 0
        self.convblock0 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),  # 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64), #
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, groups=64, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64), # 4
            nn.Dropout(dropout_value)
        )
        
        self.shortcut0 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(64),
        )

        # TRANSITION BLOCK 0
        self.transitionblock_0 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=2, padding=0, bias=False))
        # 5

        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),  # 6
            #nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, groups=64, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64), # 7
            nn.Dropout(dropout_value)
        )
        
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(64),
        )

        # TRANSITION BLOCK 1
        self.transitionblock_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=2, padding=0, bias=False))
        # 8

        # CONVOLUTION BLOCK 2       
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64), # 9
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, groups=64, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64), # 10
            nn.Dropout(dropout_value)
        ) # 

        self.shortcut2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(64),
        )
                
        # TRANSITION BLOCK 2
        self.transitionblock_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=2, padding=0, bias=False))
        # 11
        
        # CONVOLUTION BLOCK 3       
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64), # 12
            nn.Dropout(dropout_value)
        ) # 12
        
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, groups=64, padding=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
        )
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1))
        )  # 13 

        self.linear = nn.Linear(64, 10, bias=False)  # 14
        # self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        debug = False
        if debug: print ("input", x.shape)
        
        x = self.input(x)
        if debug: print ("I0", x.shape)
        
        x1 = self.convblock0(x)
        if debug: print ("C0", x1.shape)
        x2 = self.shortcut0(x)
        if debug: print ("S0", x2.shape)
        
        x3 = x1 + x2
        
        if debug: print ("C0+S0", x3.shape)
         
        x4 = self.transitionblock_0(x3)
        #x4 = x3
        if debug: print ("T0", x4.shape)
        
        x5 = self.convblock1(x4)
        if debug: print ("C1", x5.shape)
        x6 = self.shortcut1(x4)
        if debug: print ("S1", x6.shape)
        x7 = x5 + x6
        
        x8 = self.transitionblock_1(x7)
        #x8 = x7
        if debug: print ("T1", x8.shape)
        
        
        x9 = self.convblock2(x8)
        if debug: print ("C2", x9.shape)
        x10 = self.shortcut2(x8)
        x11 = x10 + x9
        
        x12 = self.transitionblock_2(x11)
        #x12 = x11
        if debug: print ("T2", x12.shape)
        
        
        x13 = self.convblock3(x12)
        if debug: print ("C3", x13.shape)
        x14 = self.shortcut3(x12)
        if debug: print ("S3", x14.shape)
        
        x15 = x13 + x14
        
        out = self.gap(x15)
        
        if debug: print ("Gap", out.shape)        
        out = out.view(out.size(0), -1)
        if debug: print ("View", out.shape) 
        out = self.linear(out)
        if debug: print ("Out", out.shape)

        return F.log_softmax(out , dim=1)  # Apply log-softmax here