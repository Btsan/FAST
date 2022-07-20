"""
Main training script for the 3DCNN bind/no-bind classification model. 

"""

import os
import argparse

import torch

from torch.utils.data import DataLoader
from data_reader import LigandDataset

from model import CNN3D
from data_transformations import VoxelTransform

from torch.nn import CrossEntropyLoss
from torch.optim import Adam, RMSprop, lr_scheduler

from main_train_validate import train, validate


"""
Program input arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument("--epoch-count", type=int, default=10, help="number of training epochs")
parser.add_argument("--batch-size", type=int, default=64, help="mini-batch size")
args = parser.parse_args()

""" 
Get training and validation dataset from local datset folder.
"""
train_batch_size = args.batch_size
path = os.path.join("datasets","postera_protease2_pos_neg_train.hdf5")
train_data = LigandDataset(path,parse_features=False)
train_dataloader = DataLoader(
    train_data, 
    batch_size=train_batch_size,
    shuffle=True, 
    drop_last=True
    )


val_batch_size = args.batch_size
path = os.path.join("datasets","postera_protease2_pos_neg_val.hdf5")
val_data = LigandDataset(path,parse_features=False)
val_dataloader = DataLoader(
    val_data, 
    batch_size=val_batch_size,
    shuffle=True,
    drop_last=True
    )


"""
Set model instances.
"""
use_cuda = torch.cuda.is_available()
cuda_count = torch.cuda.device_count()

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Use Cuda: {use_cuda}, Device count: {cuda_count}, Device selected: {device} ")

data_transform = VoxelTransform(batch_size=args.batch_size,vol_dim=32,use_cuda=use_cuda)
model = CNN3D(num_classes=2,verbose=0)

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

epochs = args.epoch_count
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    
    
    losses, train_accuracy = train(
    train_dataloader, 
    data_transform,
    model,
    loss_fn, 
    optimizer, 
    device,
    scheduler
    )

    avg_loss, accuracy = validate(
    val_dataloader, 
    data_transform,
    model,
    loss_fn, 
    device
    )

    checkpoint_dict = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": losses[-1],
    "epoch": epoch+1
    }
    model_path = os.path.join("models","3DCNN_model_" + "checkpoint" + str(epoch+1) + ".pth")
    torch.save(checkpoint_dict, model_path)



print("Done!")