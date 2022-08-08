#!/usr/bin/env python
# coding: utf-8

import torch
import torch_geometric


class SGNN(torch.nn.Module):

    def __init__(self,
                 conv_widths,
                 linear_widths,
                 conv_type,
                 conv_activation,
                 pooling_type,
                 pooling_activation,
                 linear_activation,
                 output_activation):


        super(SGNN, self).__init__()
        """
        The GNN has a set of convolutional layers, one pooling layer, and a set of linear layers
        """


        self.conv_widths = conv_widths
        self.num_conv_layers = len(conv_widths) - 1;
        self.conv_layers = torch.nn.ModuleList()
        self.conv_type = conv_type.strip()

        for i in range(self.num_conv_layers):
            if (self.conv_type.lower() == "gcn"):
                self.conv_layers.append(torch_geometric.nn.GCNConv(
                    in_channels=conv_widths[i],
                    out_channels=conv_widths[i + 1],
                    bias=False))
            elif (self.conv_type.lower() == "graphsage"):
                self.conv_layers.append(torch_geometric.nn.SAGEConv(
                    in_channels=conv_widths[i],
                    out_channels=conv_widths[i + 1],
                    bias=False))
            elif (self.conv_type.lower() == "gat"):
                self.conv_layers.append(torch_geometric.nn.GATConv(
                    in_channels=conv_widths[i],
                    out_channels=conv_widths[i + 1],
                    bias=False))
            else:
                raise ValueError("Invalid conv type. Accepted types are gcn, gat, and graphsage. Please try again")

  
        self.conv_activation = self._Get_Activation_Function(encoding=conv_activation)


        self.pooling_layer = self._Get_Pooling_Function(pooling_type)

        self.pooling_activation = self._Get_Activation_Function(encoding=pooling_activation)


        self.linear_widths = linear_widths;
        self.num_linear_layers= len(linear_widths) - 1;
        self.linear_layers = torch.nn.ModuleList()

        for i in range(self.num_linear_layers):
            self.linear_layers.append(torch.nn.Linear(
                in_features=linear_widths[i],
                out_features=linear_widths[i + 1]))


        for i in range(self.num_linear_layers):
            torch.nn.init.xavier_uniform_(self.linear_layers[i].weight)
            torch.nn.init.zeros_(self.linear_layers[i].bias)


        self.linear_activation = self._Get_Activation_Function(encoding=linear_activation)


        self.output_activation = self._Get_Activation_Function(encoding=output_activation)

    def _Get_Pooling_Function(self, encoding):
 

        # strip the encoding and make it lower case 
        encoding = encoding.strip().lower();

        if (encoding == "add"):
            return torch_geometric.nn.global_add_pool
        elif (encoding == "mean"):
            return torch_geometric.nn.global_mean_pool
        elif (encoding == "max"):
            return torch_geometric.nn.global_max_pool
        else:
            raise ValueError(
                "Invalid Pooling string. Valid options are \"add\", \"max\", and \"mean\". Got %s" % encoding)

    def _Get_Pooling_String(self, f):

        # Return the corresponding string.
        if (f == torch_geometric.nn.global_add_pool):
            return "add"
        elif (f == torch_geometric.nn.global_mean_pool):
            return "mean"
        elif (f == torch_geometric.nn.global_max_pool):
            return "max"
        else:
            raise ValueError("Invalid Pooling function.")

    def _Get_Activation_Function(self, encoding):


        
        encoding = encoding.strip().lower()

 
        if (encoding == "none"):
            return None
        elif (encoding == "relu"):
            return torch.relu
        elif (encoding == "sigmoid"):
            return torch.sigmoid
        elif (encoding == "tanh"):
            return torch.tanh
        elif (encoding == "elu"):
            return torch.nn.functional.elu
        elif (encoding == "softmax"):
            return torch.nn.functional.softmax
        else:
            raise ValueError("Invalid activation function string. Got %s" % encoding)

    def _Get_Activation_String(self, f):

        if (f is None):
            return "none"
        elif (f == torch.relu):
            return "relu"
        elif (f == torch.sigmoid):
            return "sigmoid"
        elif (f == "tanh"):
            return torch.tanh
        elif (f == torch.nn.functional.elu):
            return "elu"
        elif (f == torch.nn.functional.softmax):
            return "softmax"
        else:
            raise ValueError("Unknown activation function.")

    def forward(self, data):

    
        x = data.x
        edge_index= data.edge_index
        batch = data.batch

        r = [x]


        for i in range(self.num_conv_layers):
            
            y = self.conv_layers[i](r[i], edge_index)

            if (self.conv_activation is not None):
                r.append(self.conv_activation(y))
            else:
                r.append(y)

        combined_r = torch.hstack(r[1:])


        x = self.pooling_layer(combined_r, batch)

        if (self.pooling_activation is not None):
            x = self.pooling_activation(x)
        else:
            x = y


        for i in range(self.num_linear_layers - 1):
            y = self.linear_layers[i](x)

            if (self.linear_activation is None):
                x = y
            else:
                x = self.linear_activation(y)

        y = self.linear_layers[-1](x)

        if (self.output_activation is None):
            return y
        else:
            return self.output_activation(y)

    def get_state(self):

        state = {"Conv Activation": self._Get_Activation_String(self.conv_activation),
                 "Conv Type": self.conv_type,
                 "Pooling Activation": self._Get_Activation_String(self.pooling_activation),
                 "Linear Activation": self._Get_Activation_String(self.linear_activation),
                 "Output Activation": self._Get_Activation_String(self.output_activation),
                 "Conv Widths": self.conv_widths,
                 "Linear Widths": self.linear_widths,
                 "Pooling Type": self._Get_Pooling_String(self.pooling_layer)}


        for i in range(self.num_conv_layers):
       
            key = "Conv " + str(i)

     
            state[key] = self.conv_layers[i].state_dict()

   
        for i in range(self.num_linear_layers):
            
            key= "Linear " + str(i)
            
            state[key] = self.linear_layers[i].state_dict()

  
        return state

    def load_state(self, state):


        self.conv_activation = self._Get_Activation_Function(state["Conv Activation"])
        self.pooling_activation = self._Get_Activation_Function(state["Pooling Activation"])
        self.linear_activation = self._Get_Activation_Function(state["Linear Activation"])
        self.output_activation = self._Get_Activation_Function(state["Output Activation"])

        self.pooling_layer = self._Get_Pooling_Function(state["Pooling Type"])

   
        for i in range(self.num_conv_layers):

            key = "Conv " + str(i)
            self.conv_layers[i].load_state_dict(state[key])

       
        for i in range(self.num_linear_layers):
            
            key = "Linear " + str(i)

           
            self.linear_layers[i].load_state_dict(state[key])

    def copy(self):

        state = self.get_state()


        copy = SGNN(conv_widths=state["Conv Widths"],
                        conv_type=state["Conv Type"],
                        linear_widths=state["Linear Widths"], 
                        conv_activation=state['Conv Activation'], 
                        pooling_type=state['Pooling Type'], 
                        pooling_activation = state['Pooling Activation'], 
                        linear_activation=state['Linear Activation'], 
                        output_activation=state['Output Activation'])


        copy.load_state(state)
        return copy

