#!/usr/bin/env python
# coding: utf-8


import torch



def l2_squared(model):
    """
    args:
        model - torch.nn.Module instance

    -----------------------------------------------------------------------------------------------
    returns torch tensor
    """
    parameter_list = model.parameters()

    squared_sum : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = next(model.parameters()).device)

    for i in parameter_list:
        squared_sum += torch.sum(torch.multiply(i, i))

    # Now, return the Squared Sum.
    return squared_sum


def sse_loss(preds, targets):
    """
    Returns the sum of the squares of the components of the difference of predictions
    and trgets.
    
    args:
        preds - a set of predictions 

        targets - set of target values
    """
    residual= torch.subtract(preds, targets)

    return torch.sum(torch.multiply(residual, residual))



bce_loss = torch.nn.BCELoss(reduction = "sum")

