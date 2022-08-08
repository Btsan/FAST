#!/usr/bin/env python
# coding: utf-8


import numpy
import torch

from model import SGNN
from loss_func import l2_squared
from loss_func import bce_loss


def train(model, optimizer, loader, lambda_dict):

    model.train()


    for data in loader:
        def closure():

            optimizer.zero_grad()
            
            preds = model(data).reshape(-1)

            loss = (bce_loss(preds, data.y) * lambda_dict["data"] +
                    l2_squared(model) * lambda_dict["l2"])

            loss.backward()

            return loss

        optimizer.step(closure)

def test(model, loader, lambda_dict):

    model.eval()
    
    num_data = 0

    true_positives = 0
    false_negatives  = 0
    true_negatives = 0
    false_positives = 0

    total_data_loss = 0

    with torch.no_grad():
        for data in loader:
  
            pred = model(data).reshape(-1)

            num_data += torch.numel(pred)

            total_data_loss += bce_loss(pred, data.y)

            rounded_predictions = torch.round(pred).to(torch.int32)


            y = data.y.to(torch.int32)

            correct_predictions = torch.eq(rounded_predictions, y)


            true_positives += torch.sum(torch.logical_and(correct_predictions, y))
            false_positives += torch.sum(torch.logical_and(torch.logical_not(correct_predictions), y))
            true_negatives += torch.sum(torch.logical_and(correct_predictions, torch.logical_not(y)))
            false_negatives += torch.sum(
                torch.logical_and(torch.logical_not(correct_predictions), torch.logical_not(y)))

    mean_data_loss = total_data_loss / num_data
    l2_loss = l2_squared(model).item()
    total_loss= (mean_data_loss * loader.batch_size) * lambda_dict["data"] + l2_loss * lambda_dict["l2"]

    results_dict = {"True Positives": true_positives,
                    "False Negatives": false_negatives,
                    "True Negatives": true_negatives,
                    "False Positives": false_positives,
                    "Mean Data Loss": mean_data_loss,
                    "L2 Loss": l2_loss,
                    "Total Loss": total_loss}


    return results_dict


def report_results(test_results):
    
    
    actual_positives = test_results["True Positives"] + test_results["False Negatives"]
    actual_negatives = test_results["True Negatives"] + test_results["False Positives"]
    predicted_positives = test_results["True Positives"] + test_results["False Positives"]
    predicted_negatives = test_results["True Negatives"] + test_results["False Negatives"]

    correct = test_results["True Positives"] + test_results["True Negatives"]

  
    TPR = test_results["True Positives"] / actual_positives
    PPV  = test_results["True Positives"] / predicted_positives
    TNR = test_results["True Negatives"] / actual_negatives
    NPV  = test_results["True Negatives"] / predicted_negatives

  
    F1 = 2 * (PPV * TPR) / (PPV + TPR)
    accuracy = correct / (actual_positives + actual_negatives)

 
    print("Accuracy  = %8.6f | Loss = %8.2e" % (accuracy, test_results["Mean Data Loss"]))

