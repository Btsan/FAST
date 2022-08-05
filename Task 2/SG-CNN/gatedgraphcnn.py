#!/usr/bin/env python
# coding: utf-8


import torch
from torch import Tensor
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax #scatter_
#from torch_geometric.utils import softmax
#from torch_scatter import scatter as scatter_
from torch_geometric.nn.inits import uniform, reset

class GatedGraphConv(MessagePassing):
   

    def __init__(self, out_channels, num_layers, edge_network, aggr="add", bias=True):
        super(GatedGraphConv, self).__init__(aggr)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.edge_network = (
            edge_network  # TODO: make into a list of neural networks for each edge_attr
        )

        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        size = self.out_channels
        uniform(size, self.weight)
        self.rnn.reset_parameters()

    # TODO: remove none defautl for edge_attr
    def forward(self, x, edge_index, edge_attr):
        """"""
        h = x if x.dim() == 2 else x.unsqueeze(-1)
        assert h.size(1) <= self.out_channels

        # if input size is < out_channels, pad input with 0s to equal the sizes
        if h.size(1) < self.out_channels:
            zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(h, self.weight[i])

           
            
            m = self.propagate(edge_index=edge_index, x=h, aggr="add")
            h = self.rnn(m, h)

        return h

    def message(self, x_j):  # pragma: no cover
    

        return x_j

    def update(self, aggr_out):  # pragma: no cover
        

        return aggr_out

    def __repr__(self):
        return "{}({}, num_layers={})".format(
            self.__class__.__name__, self.out_channels, self.num_layers
        )


class GlobalAttention(torch.nn.Module):
  

    def __init__(self, gate_nn, nn=None):
        super(GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, size)
        out = scatter_("add", gate * x, batch, size)

        return out

    def __repr__(self):
        return "{}(gate_nn={}, nn={})".format(
            self.__class__.__name__, self.gate_nn, self.nn
        )


class PotentialNetAttention(torch.nn.Module):
    def __init__(self, net_i, net_j):
        super(PotentialNetAttention, self).__init__()

        self.net_i = net_i
        self.net_j = net_j

    def forward(self, h_i, h_j):
        return torch.nn.Softmax(dim=1)(
            self.net_i(torch.cat([h_i, h_j], dim=1))
        ) * self.net_j(h_j)
