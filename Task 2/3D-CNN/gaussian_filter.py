import math
import numbers
import torch
import torch.nn as nn

class GaussianFilter(nn.Module):
    """
    Gaussian Filter PyTorch Layer

    Original Gaussian Filter model from FAST. Expects inputs are in batches. 

    Example:
        >>> from img_util import GaussianFilter
        >>> gaussian_filter = GaussianFilter(dim=3, channels=19, kernel_size=11, sigma=1, use_cuda=use_cuda)

    """
    def __init__(self, dim=3, channels=19, kernel_size=11, sigma=1, use_cuda=True):
        super(GaussianFilter, self).__init__()

        self.use_cuda = use_cuda
        if isinstance(kernel_size, numbers.Number):
            self.kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            self.sigma = [sigma] * dim

        self.padding = kernel_size // 2

        # Gaussian kernel is the product of the gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in self.kernel_size])
        for size, std, mgrid in zip(self.kernel_size, self.sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size()) #-> doesn't need to add one more axis
        #kernel = kernel.view(1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        if self.use_cuda:
            kernel = kernel.cuda()

        self.register_buffer('weight', kernel)
        self.groups = channels
        if dim == 1:
            self.conv = nn.functional.conv1d
        elif dim == 2:
            self.conv = nn.functional.conv2d
        elif dim == 3:
            self.conv = nn.functional.conv3d

    def forward(self, input):
        """
        Forward pass method of the Gaussian Filter layer. 

        Args: 
            inputs: A torch tensor of shape ()
        
        Returns
        -------
        outputs: 
             A torch tensor of shape ()
            
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.padding)