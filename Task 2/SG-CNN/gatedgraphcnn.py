#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import Tensor
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax #scatter_
#from torch_geometric.utils import softmax
#from torch_scatter import scatter as scatter_
from torch_geometric.nn.inits import uniform, reset


# In[ ]:


class GatedGraphConv(MessagePassing):
    def __init__(self, out_channels, num_layers, edge_network, aggr="add", bias=True):
        super(GatedGraphConv, self).__init__(aggr)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.edge_network = (
            edge_network 
        )

        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        size = self.out_channels
        uniform(size, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        h = x if x.dim() == 2 else x.unsqueeze(-1)
        assert h.size(1) <= self.out_channels

        if h.size(1) < self.out_channels:
            zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(h, self.weight[i])

            m = self.propagate(edge_index=edge_index, x=h, aggr="add")
            h = self.rnn(m, h)

        return h

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return "{}({}, num_layers={})".format(
            self.__class__.__name__, self.out_channels, self.num_layers
        )


# In[ ]:


class PotentialNetAttention(torch.nn.Module):
    def __init__(self, net_i, net_j):
        super(PotentialNetAttention, self).__init__()

        self.net_i = net_i
        self.net_j = net_j

    def forward(self, h_i, h_j):
        return torch.nn.Softmax(dim=1)(
            self.net_i(torch.cat([h_i, h_j], dim=1))
        ) * self.net_j(h_j)

