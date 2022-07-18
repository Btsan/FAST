#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch
import torch.nn as nn # neural network library 
import torch.nn.functional as F 
from torch.nn import BatchNorm1d # applies batch normalization over the 3d input data 

from torch_geometric.nn import (
    GCNConv, GlobalAttention, global_add_pool, 
    NNConv, avg_pool_x, avg_pool,
    max_pool_x, GatedGraphConv)

from torch_geometric.utils import (
    to_dense_batch, add_self_loops, remove_self_loops, normalized_cut,
    dense_to_sparse, is_undirected, to_undirected, contains_self_loops)

#from torch_geometric.utils import scatter_

from torch_geometric.nn import DataParallel as GeometricDataParallel
from torch_geometric.data import Batch
from gatedgraphcnn import GatedGraphConv, PotentialNetAttention
from torch.nn import init
from torch_sparse import coalesce
from torch_geometric.nn.pool import avg_pool_x


# In[ ]:


class PotentialNetPropagation(torch.nn.Module):
    """
    Creating the neural network propagator  
    
    args:

    feat_size: int, the number of features in each input sample 
    gather_width: int,
    k: int, 
    neighbor_threshold: 
    output_pool_result: bool
    bn_track_running_stats: 
    """
    def __init__(
        self, 
        feat_size=19, 
        gather_width=64, 
        k=2, 
        neighbor_threshold=None, 
        output_pool_result=False, 
        bn_track_running_stats=False
    ):
        super(PotentialNetPropagation, self).__init__()
        
        self.neighbor_threshold = neighbor_threshold 
        self.bn_track_running_stats = bn_track_running_stats
        self.edge_attr_size = 1 
        self.k = k
        self.gather_width = gather_width
        self.feat_size = feat_size

        """
        creating the edge network using a sequential container 
        
        
        """
        self.edge_network_nn = nn.Sequential(
            nn.Linear(self.edge_attr_size, int(self.feat_size / 2)),
            nn.Softsign(),
            nn.Linear(int(self.feat_size /2), self.feat_size),
            nn.Softsign(),
        )  # softsign applies function (x/(1+|x|)) to the integer calculated by linear transformation
        
        self.edgeNnetwork = NNConv(
            self.feat_size, 
            self.edge_attr_size * self.feat_size, 
            nn=self.edge_network_nn, 
            root_weight=True, 
            aggr='add'
        )
        
        self.gate = PotentialNetAttention(
            net_i = nn.Sequential(
                nn.Linear(self.feat_size * 2, self.feat_size),
                nn.Softsign(), 
                nn.Linear(self.feat_size, self.gather_width), 
                nn.Softsign(),
            ), 
            net_j = nn.Sequential(
                nn.Linear(self.feat_size, self.gather_width), 
                nn.Softsign()
            ),
        )
        
        self.output_pool_result = output_pool_result
        if self.output_pool_result:
            self.global_add_pool = global_add_pool 
    
    def forward(self, data, edge_index, edge_attr):
        """
        nn propagation
        
        might need to fix depending on reading in data 
        """
        h_0 = data
        h_1 = self.gate(h_0, edge_index, edge_attr)
        h_1 = self.attention(h_1, h_0)
        
        return h_1

    


# In[ ]:


class GraphThreshold(torch.nn.Module):
    def __init__(self, t):
        super(GraphThreshold, self).__init__()
        self.t = nn.Parameter(t, requires_grad) # assuming no cuda, however if cuda, uncomment the below 
        #self.t = nn.Parameter(t, requires_grad=True).cuda()
    
    def filter_adj(self, row, col, edge_attr, mask):
        mask = mask.squeeze()
        return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]
    
    def forward(self, edge_index, edge_attr):
        """
        Args: 
        edge_index: edge indices
        edge_attr: edge weights or multi-dimensional edge features
        force_undirected: optional bool, if true it forces undircted output 
        num_nodes: number of nodes 
        
        """
        N = maybe_num_nodes(edge_index, None)
        row, col = edge_index 
        
        mask = edge_attri <= self.t
        
        row, col, edge_attr = self.filter_adj(row, col, edge_attr, mask)
        edge_index = torch.stack([torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)], dim=0)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
        
        return edge_index, edge_attr


# In[ ]:


class PotentialNetFullyConnected(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(potentialNetFullyConnected, self).__init__()
        
        self.output = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / 1.5)), 
            nn.ReLU(), 
            nn.Linear(int(in_channels / 1.5), int(in_channels / 2)), 
            nn.ReLU(), 
            nn.Linear(int(in_channels / 2), out_channels),
        )
    def forward(self, data, return_hidden_feature=False):
        if return_hidden_feature:
            return self.output[:2](data), self.output[:-4](data), self.output(data)
        else:
            return self.output(data)


# In[ ]:


class PotentialNetParallel(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        covalent_gather_width=128,
        non_covalent_gather_width=64,
        covalent_k=1,
        non_covalent_k=1,
        covalent_neighbor_threshold=None,
        non_covalent_neighbor_threshold=None,
        always_return_hidden_feature=False,
    ):
        super(PotentialNetParallel, self).__init__()
        
        self.covalent_neighbor_threshold = GraphThreshold(
            torch.ones(1) * covalent_neighbor_threshold)
        """
        if using cuda: 
        self.covalent_neighbor_threshold = GraphThreshold(
        torch.ones(1).cuda() * covalent_neighbor_threshold)
        """
        self.non_covalent_neighbor_threshold = GraphThreshold(
            torch.ones(1) * non_covalent_neighborhood)
        """
        if using cuda 
        self.non_covalent_neighbor_threshold = GraphThreshold(
        torch.ones(1).cuda() * non_covalent_neighbor_threshold)
        """
        self.always_return_hidden_feature = always_return_hidden_feature
        
        self.global_add_pool = global_add_pool
        
        self.covalent_propagation = PotentialNetPropagation(
            feat_size=in_channels, 
            gather_width=covalent_gather_width, 
            neighbor_threshold=self.covalent_neighbor_threshold, 
            k=non_covalent_k,
        )
        self.global_add_pool = global_add_pool 
        self.output = PotentialNetFullyConnected(non_covalent_gather_width, out_channels)
        
    def forward(self, data, return_hidden_feature=False):
        
        """
        if using torch cuda
        
        if torch.cuda.is_available():
            data.x = data.x.cuda()
            data.edge_attr = data.edge_attr.cuda()
            data.edge_index = data.edge_index.cuda()
            data.batch = data.batch.cuda()
        
        """
        
        if not is_undirected(data.edge_index):
            data.edge_index = to_undirected(data.edge_index)

        # make sure that nodes can propagate messages to themselves
        if not contains_self_loops(data.edge_index):
            data.edge_index, data.edge_attr = add_self_loops(
                data.edge_index, data.edge_attr.view(-1)
            )

        # covalent_propagation
        # add self loops to enable self propagation
        covalent_edge_index, covalent_edge_attr = self.covalent_neighbor_threshold(
            data.edge_index, data.edge_attr
        )
        (
            non_covalent_edge_index,
            non_covalent_edge_attr,
        ) = self.non_covalent_neighbor_threshold(data.edge_index, data.edge_attr)

        # covalent_propagation and non_covalent_propagation
        covalent_x = self.covalent_propagation(
            data.x, covalent_edge_index, covalent_edge_attr
        )
        non_covalent_x = self.non_covalent_propagation(
            covalent_x, non_covalent_edge_index, non_covalent_edge_attr
        )

        # zero out the protein features then do ligand only gather...hacky sure but it gets the job done
        non_covalent_ligand_only_x = non_covalent_x
        non_covalent_ligand_only_x[data.x[:, 14] == -1] = 0
        pool_x = self.global_add_pool(non_covalent_ligand_only_x, data.batch)

        # fully connected and output layers
        if return_hidden_feature or self.always_return_hidden_feature:
            # return prediction and atomistic features (covalent result, non-covalent result, pool result)

            avg_covalent_x, _ = avg_pool_x(data.batch, covalent_x, data.batch)
            avg_non_covalent_x, _ = avg_pool_x(data.batch, non_covalent_x, data.batch)

            fc0_x, fc1_x, output_x = self.output(pool_x, return_hidden_feature=True)

            return avg_covalent_x, avg_non_covalent_x, pool_x, fc0_x, fc1_x, output_x
        else:
            return self.output(pool_x)


            
            

