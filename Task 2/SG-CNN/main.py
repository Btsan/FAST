#!/usr/bin/env python
# coding: utf-8


import os

import sys

main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(main_path, "data")
sys.path.append(data_path)


from data_utils import PosteraDataset

import numpy
import torch
import torch_geometric
from typing import List, Dict
import time
from datetime import datetime

from model import SGNN
from train_test import train, test, report_results


def train_model(parser,
                loaders,
                device):

    if (parser["Load From File"] == True):
        
        state = torch.load("../saves/" + parser["Load File Name"])

        
        conv_widths = state["Conv Widths"]
        conv_type = state["Conv Type"]
        conv_activation = state["Conv Activation"]
        pooling_type = state["Pooling Type"]
        pooling_activation = state["Pooling Activation"]
        linear_widths = state["Linear Widths"]
        linear_activation = state["Linear Activation"]
        output_activation = state["Output Activation"]

        
        model = SGNN(conv_widths=conv_widths,
                    linear_widths=linear_widths,
                    conv_type=conv_type)

        model.load_state(state)

    else:
      
        model = SGNN(conv_widths=parser["Conv Widths"],
                    conv_activation=parser["Conv Activation"],
                    conv_type=parser["Conv Type"],
                    pooling_type=parser["Pooling Type"],
                    pooling_activation=parser["Pooling Activation"],
                    linear_widths=parser["Linear Widths"],
                    linear_activation=parser["Linear Activation"],
                    output_activation=parser["Output Activation"])


    model = model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=parser["Learning Rate"])


    epoch_timer = time.perf_counter()
    print("Running %u epochs..." % parser["Number of Epochs"])


    max_correct = 0
    best_model = model.copy()

    for i in range(parser["Number of Epochs"]):

        print("Epoch %4u / %u" % (i + 1, parser["Number of Epochs"]))

        train(model=model,
              optimizer=optimizer,
              loader=loaders["Train"],
              lambda_dict=parser["lambda_dict"])


        train_results = test(model=model,
                             loader=loaders["Train"],
                             lambda_dict=parser["lambda_dict"])

        val_results = test(model=model,
                           loader=loaders["Validation"],
                           lambda_dict=parser["lambda_dict"])

        num_correct = val_results["True Positives"] + val_results["True Negatives"]
        if (num_correct >= max_correct):
  
            max_correct = num_correct


            best_model = model.copy()


        print("Training:");
        report_results(train_results)
        print("Validation:");
        report_results(val_results)

    epoch_runtime = time.perf_counter() - epoch_timer
    print("It took %7.2fs," % epoch_runtime)
    print("average length of %7.2fs per epoch." % (epoch_runtime / parser["Number of Epochs"]))


  
    train_results = test(model=best_model,
                         loader=loaders["Train"],
                         lambda_dict=parser["lambda_dict"])

    val_results = test(model=best_model,
                       loader=loaders["Validation"],
                       lambda_dict=parser["lambda_dict"])

    test_results = test(model=best_model,
                        loader=loaders["Test"],
                        lambda_dict=parser["lambda_dict"])
    print("\nBest model results:")
    print("Training:");
    report_results(train_results)
    print("Validation:");
    report_results(val_results)
    print("Testing:");
    report_results(test_results)


    return {"Model": best_model,
            "Train Results": train_results,
            "Test Results": test_results,
            "Validation Results": val_results}


def setup_loaders(device,
                  batch_size):

    train_data = PosteraDataset(root='./', hdf5_file_name='postera_protease2_pos_neg_train.hdf5')
    test_data = PosteraDataset(root='./', hdf5_file_name='postera_protease2_pos_neg_test.hdf5')
    val_data = PosteraDataset(root='./', hdf5_file_name='postera_protease2_pos_neg_val.hdf5')

    train_data.data.x = train_data.data.x.to(device)
    train_data.data.edge_index = train_data.data.edge_index.to(device)
    train_data.data.y = train_data.data.y.to(device)

    test_data.data.x = test_data.data.x.to(device)
    test_data.data.edge_index = test_data.data.edge_index.to(device)
    test_data.data.y = test_data.data.y.to(device)

    val_data.data.x = val_data.data.x.to(device)
    val_data.data.edge_index = val_data.data.edge_index.to(device)
    val_data.data.y = val_data.data.y.to(device)


    train_loader = torch_geometric.loader.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True)

    test_loader = torch_geometric.loader.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False)

    val_loader = torch_geometric.loader.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False)

    loaders = {"Train": train_loader,
                     "Validation": val_loader,
                     "Test": test_loader}

    return loaders


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64

    loaders = setup_loaders(device=device, batch_size=batch_size)


    parser = {} #parsing dictionary

    parser["Load From File"] = False;
    parser["Load File Name"] = ""


    parser["Conv Widths"] = [19, 7, 7, 7, 7, 7]
    parser["Conv Type"] = "GCN"
    parser["Conv Activation"] = "elu"
    parser["Pooling Type"] = "mean"
    parser["Pooling Activation"] = "elu"
    parser["Linear Widths"] = [35, 15, 1]
    parser["Linear Activation"]  = "elu"
    parser["Output Activation"]= "sigmoid"


    parser["Learning Rate"] = 0.01


    parser["Number of Epochs"] = 5
    parser["lambda_dict"]= {"data": 1.0, "l2": 0.003}


    results = train_model(parser=parser, loaders=loaders, device=device)

    time = datetime.now();
    save_file_name = "../" + parser["Conv Type"] + "_" + str(time.day) + "_" + str(time.hour) + "_" + str(
        time.minute)


    state = results["model"].get_state()

    torch.save(state, save_file_name)

