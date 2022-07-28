import os
from time import time
import multiprocessing as mp

import torch

from torch.utils.data import DataLoader

from data_transformations import VoxelTransform
from data_reader import LigandDataset, LazyLigandDataset

def main():
    batch_size = 128
    train_set = os.path.join("datasets","postera_protease2_pos_neg_train.hdf5")
    train_data = LigandDataset(train_set)

    for num_workers in range(2, mp.cpu_count()+2, 2):  

        train_loader = DataLoader(
        train_data, 
        batch_size=batch_size,
        shuffle=True, 
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False
        )

        start = time()
        for epoch in range(1, 3):

            for i, data in enumerate(train_loader, 0):
                
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
        
if __name__ == '__main__':
    main()