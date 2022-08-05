#!/usr/bin/env python
# coding: utf-8

import torch 
import argparse
from dataset import PosteraDataset
from sgcnn_model import PotentialNetParallel, GraphThreshold
from torch_geometric.data import DataListLoader
from torch_geometric.data import DataLoader
from torch_geometric.nn import DataParallel
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", default=32, type=int, help="batch size for model")
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
parser.add_argument("--epochs", default=100, type=int, help="number of epochs for training")
parser.add_argument("--train-data", type=str, help="path to training data")
parser.add_argument("--val-data", type=str, help="path to validation data ")
parser.add_argument("--feature-size", type=int, default=75, help='feature size for the model')

parser.add_argument("--covalent-gather-width", type=int, default=128)
parser.add_argument("--non-covalent-gather-width", type=int, default=128)
parser.add_argument("--covalent-k", type=int, default=1)
parser.add_argument("--non-covalent-k", type=int, default=1)
parser.add_argument("--covalent-threshold", type=float, default=1.5)
parser.add_argument("--non-covalent-threshold", type=float, default=7.5)

args=parser.parse_args()


def worker_init_fn(worker_id):
    np.random.seed(int(0))


def collate_fn_none_filter(batch):
    return [x for x in batch if x is not None]


def train():
    train_bs = args.batch_size
    val_bs = args.batch_size
    for data in args.train_data:
        train_dataset.append(PosteraDataset(data))
    for data in args.val_data:
        val_dataset.append(PosteraDataset(data))
    train_dataloader = DataListLoader(train_dataset,batch_size=train_bs, shuffle=False, worker_init_fn=worker_init_fn) 
    # can add drop_last=True in the instance of shuffling dataset
    val_dataloader = DataListLoader(val_dataset, batch_size=val_bs, shuffle=False, worker_init_fn=worker_init_fn)
    
    model = (PotentialNetParallel(
        in_channels=args.feature_size,
        out_channels=1,
        covalent_gather_width=args.covalent_gather_width,
        non_covalent_gather_width=args.non_covalent_gather_width,
        covalent_k=args.covalent_k,
        non_covalent_k=args.non_covalent_k,
        covalent_neighbor_threshold=args.covalent_threshold,
        non_covalent_neighbor_threshold=args.non_covalent_threshold,)).float()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)
    
    model.train()
    
    running_loss=0
    correct=0
    total=0
    
    for batch in tqdm(train_dataloader):
        batch = [x for x in batch if x is not None]
        optimizer.zero_grad()
        
        data = [x[2] for x in batch]
        y_ = model(data)
        y = torch.cat([x[2].y for x in batch])
        
        loss = criterion(y.float(), y_cpu().float())
        losses.append(loss.cpu().data.item())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
      
        train_loss=running_loss/len(trainloader)
        accu=100.*correct/total

        train_accu.append(accu)
        train_losses.append(train_loss)
        print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))

    # calculate loss

            
            # Add accuracy 
            
            

