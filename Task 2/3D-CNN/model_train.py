"""
Main training script for the 3DCNN bind/no-bind classification model. 

"""

import os
import argparse
from typing_extensions import dataclass_transform

import torch

from torch.utils.data import DataLoader
from data_reader import LigandDataset
import h5py

from model import CNN3D
from data_transformations import VoxelTransform

from torch.nn import CrossEntropyLoss
from torch.optim import Adam, RMSprop, lr_scheduler

from main_train_evaluate import train, evaluate



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-count", type=int, default=1, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="mini-batch size")
    parser.add_argument("--vol-dim", type=int, default=48, help="voxelizer grid size")
    parser.add_argument("--feat-dim", type=int, default=19, help="number of features in dataset")
    parser.add_argument("--train-set", type=str, default=os.path.join("datasets","postera_protease2_pos_neg_train.hdf5") )
    parser.add_argument("--val-set", type=str, default=os.path.join("datasets","postera_protease2_pos_neg_val.hdf5") )
    parser.add_argument("--model_name", type=str, default='3DCNN_model')
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    """ 
    Get training and validation dataset from local datset folder.
    """
    train_batch_size = args.batch_size
    print(f'training dataset {args.train_set}')
    # train_data = LigandDataset(args.train_set,parse_features=False)
    train_data = LigandDataset(args.train_set)
    train_dataloader = DataLoader(
        train_data, 
        batch_size=train_batch_size,
        shuffle=True, 
        drop_last=True,
        num_workers=0,
        )
    print(train_data[0][0].shape)

    val_batch_size = args.batch_size
    print(f'validation dataset {args.val_set}')
    file = h5py.File(args.val_set,"r")
    val_data = LigandDataset(args.val_set)
    val_dataloader = DataLoader(
        val_data, 
        batch_size=val_batch_size,
        shuffle=True,
        drop_last=True
        )


    """
    Set model instances.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device count: {torch.cuda.device_count()}, Device selected: {device} ")

    data_transform  = VoxelTransform(batch_size=args.batch_size,feat_dim=args.feat_dim,vol_dim=args.vol_dim)
    model = CNN3D(
        num_classes=2,
        feat_dim=args.feat_dim,
        vol_dim=args.vol_dim,
        verbose=False)

    """
    Set Training objects.

    We need a loss function, an optimizer, and a learning rate scheduler. 
    """
    learning_rate = 7e-4
    decay_iter = 100
    decay_rate = 0.95

    loss_fn = CrossEntropyLoss()

    #optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    optimizer = RMSprop(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_iter, gamma=decay_rate)

    """
    Train!

    This is the main training loop. We pass through the data multiple times. After each pass (epoch),
    we check the accuracy of the trained model on the validation set and save the model information to a .pth file extension. 

    """

    prior_epochs = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        prior_epochs = checkpoint['epoch']

    for epoch in range(args.epoch_count):
        print(f"Epoch {epoch+1}\n-------------------------------")
        
        losses, train_accuracy = train(
            train_dataloader, 
            data_transform,
            model,
            loss_fn, 
            optimizer, 
            device,
            scheduler,
            )

        avg_loss, accuracy = evaluate(
            val_dataloader, 
            data_transform,
            model,
            loss_fn, 
            device,
            )

        checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": losses[-1],
        "epoch": prior_epochs + epoch + 1
        }
        model_path = os.path.join("models", f"{args.model_name}_{prior_epochs + epoch + 1}.pth")
        torch.save(checkpoint_dict, model_path)



    print("Done!")