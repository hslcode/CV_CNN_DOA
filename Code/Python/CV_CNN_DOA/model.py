#Complex Conv ref:
#https://github.com/Medabid1/ComplexValuedCNN/tree/master/src
#http://t.csdn.cn/hX7nA
#http://t.csdn.cn/IqfOu
#http://t.csdn.cn/xoyxO
#https://github.com/wavefrontshaping/complexPyTorch
#https://github.com/williamFalcon/pytorch-complex-tensor
#https://github.com/ChihebTrabelsi/deep_complex_networks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from complexLayers import ComplexBatchNorm2d,ComplexConv2d,ComplexReLU,ComplexLinear,ComplexDropout,C_CSELU,ZReLU
from complexFunctions import complex_relu, complex_max_pool2d

class CV_CNN_Net(nn.Module):
    def __init__(self,device):
        super(CV_CNN_Net, self).__init__()
        self.device = device
        self.nn = nn.Sequential(
                                ComplexConv2d(1, 128, kernel_size = (3, 3),stride = (2, 2),padding=(1,1)),
                                ComplexBatchNorm2d(128),
                                # ComplexReLU(),
                                C_CSELU(),
                                # ZReLU(),

                                ComplexConv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                ComplexBatchNorm2d(128),
                                # ComplexReLU(),
                                C_CSELU(),
                                # ZReLU(),

                                ComplexConv2d(128, 128, kernel_size=(2, 2), stride=(1, 1)),
                                ComplexBatchNorm2d(128),
                                # ComplexReLU(),
                                C_CSELU(),
                                # ZReLU(),

                                ComplexConv2d(128, 128, kernel_size=(2, 2), stride=(1, 1)),
                                ComplexBatchNorm2d(128),
                                # ComplexReLU(),
                                C_CSELU(),
                                # ZReLU(),

                                nn.Flatten(),
                                ComplexLinear(4608,2048),
                                # ComplexReLU(),
                                C_CSELU(),
                                # ZReLU(),

                                ComplexDropout(p=0.3,device = self.device),
                                ComplexLinear(2048, 1024),
                                # ComplexReLU(),
                                C_CSELU(),
                                # ZReLU(),

                                ComplexDropout(p=0.3,device = self.device),
                                ComplexLinear(1024, 121),
                                # ComplexReLU(),
                                C_CSELU(),
                                # ZReLU(),
                                ComplexDropout(p=0.3, device=self.device),
                                )
        self.FNN = nn.Sequential(
            nn.Linear(242, 121, bias=False),
            nn.Sigmoid()
        )
    def forward(self, data_train):
        Outputs = self.nn(data_train)
        Outputs = torch.cat([Outputs.real,Outputs.imag],dim=1)
        Outputs = self.FNN(Outputs)
        return Outputs
