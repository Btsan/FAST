import torch
import torch.nn as nn

from math import floor

class CNN3D(nn.Module):

    """
    A simple three-dimensional convolutional neural network for classifcation.

    Model consists of two convoluational blocks and two linear layers. The first convolutional block encorperates 
    residuals before pooling. 

    The number of classes is specified by the input argument "num_classes", which then specifies the output dimension
    of the last linear layer. 

    Given the number of filters, the model computes the size of the outputs from the first two convolutional blocks 
    and the pooling layer. This is then used to compute the input dimension of the first linear layer.

    """

    # num_filters=[32,64,64], [64,128,256] or [96,128,128]
    def __init__(self, feat_dim=19, vol_dim=48, num_filters = [64,128], num_classes=2,verbose=False):
        super(CNN3D, self).__init__()
        
        # input arguments
        self.feat_dim = feat_dim
        self.vol_dim = vol_dim
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.verbose = verbose 

        # intermediate sizes
        self.conv_dim_one = floor( (vol_dim+2)/3 )
        self.conv_dim_two = floor( (self.conv_dim_one+1)/2)
        self.pool_dim = floor(self.conv_dim_two/2)
        self.lin_dim = ((self.pool_dim) ** 3) * self.num_filters[1] 


        self.conv_block_one = self.__conv_layer_set__(self.feat_dim, self.num_filters[0], 7, 3, 3)
        self.conv_block_two = self.__conv_layer_set__(self.num_filters[0], self.num_filters[1], 5, 2, 2)

        self.res_block_one =  self.__conv_layer_set__(self.num_filters[0], self.num_filters[0], 7, 1, 3)
        #self.res_block_two =  self.__conv_layer_set__(32, 32, 7, 1, 3)

        self.pool = nn.MaxPool3d(2)

        self.linear_one = nn.Linear(self.lin_dim, 100)
        torch.nn.init.normal_(self.linear_one.weight, 0, 1)
        self.linear_bottleneck = nn.BatchNorm1d(num_features=100, affine=True, momentum=0.1).train()
        self.linear_two = nn.Linear(100, self.num_classes) 
        torch.nn.init.normal_(self.linear_two.weight, 0, 1)
        self.relu = nn.ReLU()

    def __conv_layer_set__(self, in_channel, out_channel, k_size, stride, padding):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=k_size, stride=stride, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channel),
            )

        return conv_layer

    def forward(self, x):
        
        if self.verbose:
            print(x.shape)

        x = self.conv_block_one(x)
        if self.verbose:
            print(x.shape)

        r = self.res_block_one(x)
        if self.verbose:
            print(r.shape)

        #r = self.res_block_two(r + x)
        #if self.verbose:
        #   print(r.shape)

        x = self.conv_block_two(x+r)
        if self.verbose:
            print(x.shape)

        x = self.pool(x)
        if self.verbose:
            print(x.shape)

        x = x.view(x.size(0), -1)
        if self.verbose:
            print(x.shape)

        x = self.linear_one(x)
        x = self.relu(x)
        f1 = self.linear_bottleneck(x) if x.shape[0]>1 else x  #batchnorm train require more than 1 batch
        if self.verbose:
            print(f1.shape)

        f2 = self.linear_two(f1)
        if self.verbose:
            print(f2.shape)

        return f2, f1   