import torch
import torch.nn as nn

class CNN3D(nn.Module):

    """
    A simple three-dimensional convolutional neural network for classifcation.

    Model consists of two convoluational blocks and two linear layers. The first convolutional block encorperates 
    residuals before pooling. 

    The number of classes is specified by the input argument "num_classes", which then specifies the output dimension
    of the last linear layer. 

    """

    # num_filters=[32,64,64], [64,128,256] or [96,128,128]
    def __init__(self, feat_dim=19, num_classes=2,verbose=0):
        super(CNN3D, self).__init__()

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.verbose = verbose 

        self.conv_block_one = self.__conv_layer_set__(self.feat_dim, 32, 7, 3, 3)
        self.conv_block_two = self.__conv_layer_set__(32, 64, 5, 2, 2)

        self.res_block_one =  self.__conv_layer_set__(32, 32, 7, 1, 3)
        self.res_block_two =  self.__conv_layer_set__(32, 32, 7, 1, 3)

        self.pool = nn.MaxPool3d(2)

        self.linear_one = nn.Linear(3*3*3*64, 100)
        torch.nn.init.normal_(self.linear_one.weight, 0, 1)
        self.linear_bottleneck = nn.BatchNorm1d(num_features=100, affine=True, momentum=0.1).train()
        self.linear_two = nn.Linear(100, self.num_classes) 
        torch.nn.init.normal_(self.linear_two.weight, 0, 1)
        self.relu = nn.ReLU()

    def __conv_layer_set__(self, in_c, out_c, k_size, stride, padding):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=k_size, stride=stride, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_c),
            )

        return conv_layer

    def forward(self, x):
        
        if self.verbose != 0:
            print(x.shape)

        x = self.conv_block_one(x)
        if self.verbose != 0:
            print(x.shape)

        r = self.res_block_one(x)
        if self.verbose != 0:
            print(r.shape)

        r = self.res_block_two(r + x)
        if self.verbose != 0:
           print(r.shape)

        x = self.conv_block_two(x+r)
        if self.verbose != 0:
            print(x.shape)

        x = self.pool(x)
        if self.verbose != 0:
            print(x.shape)

        x = x.view(x.size(0), -1)
        if self.verbose != 0:
            print(x.shape)

        x = self.linear_one(x)
        x = self.relu(x)
        f1 = self.linear_bottleneck(x) if x.shape[0]>1 else x  #batchnorm train require more than 1 batch
        if self.verbose != 0:
            print(f1.shape)

        f2 = self.linear_two(f1)
        if self.verbose != 0:
            print(f2.shape)

        return f2, f1   