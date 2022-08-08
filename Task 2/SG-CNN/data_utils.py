#!/usr/bin/env python
# coding: utf-8



import h5py as h5
import numpy
import os
import torch

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from sklearn.metrics import pairwise_distances

#parameters 
bond_threshold = 1.8
max_bars = 50
#track: bool = True
keep_features = False


class PosteraDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 hdf5_file_name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        """
        Creating dataloader 
        args:
            root - root to the path of the file

            hdf5_file_name - name of the file you're using
        """
        self.hdf5_file_name = hdf5_file_name

        
        super().__init__(root, transform, pre_transform, pre_filter)

        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):

        if self.hdf5_file_name != None:
            return [self.hdf5_file_name]
        else:
            return []

    @property
    def processed_file_names(self):

        return ['data.pt'];

    def process(self):

        
        data_list = []

        hdf5_file_path = os.path.join(self.hdf5_file_name)
        hdf5_file = h5.File(hdf5_file_path, 'r')


        molecule_names = list(hdf5_file.keys())
        molecule_count  = len(molecule_names)

        for m in range(molecule_count):
            num_bars = int((m / molecule_count) * max_bars)
            progress_bar = "[" + "#" * num_bars + " " * (max_bars - num_bars) + "]"
            print(progress_bar, end='')
            print(" (%5u / %u)" % (m, molecule_count), end='\r')


            molecule_name = molecule_names[m]

            ligand_data = hdf5_file[molecule_name]['ligand']


            label= hdf5_file[molecule_name].attrs['label']


            num_atoms = 0;
            for i in range(ligand_data.shape[0]):
  
                if (numpy.sum(numpy.abs(ligand_data[i, :])) == 0):

                    num_atoms = i
                    break


            ligand_data = ligand_data[0:num_atoms, :]
            
            num_features= ligand_data.shape[1] - 3;

            total_atoms = 0
            running_sum = torch.zeros(num_features, dtype=torch.float32)
            running_max = torch.zeros(num_features, dtype=torch.float32)
            running_min = torch.zeros(num_features, dtype=torch.float32)


            coordinates = ligand_data[:, 0:3]


            distance = pairwise_distances(coordinates, metric='euclidean')


            matrix_adj = numpy.less(distance, bond_threshold)


            num_edges = numpy.sum(matrix_adj)


            edge_index = torch.empty(size=(2, num_edges), dtype=torch.long)
            edge_counter = 0

            for i in range(num_atoms):
                for j in range(num_atoms):


                    if (matrix_adj[i, j] == 1):
                        edge_index[0, edge_counter] = j
                        edge_index[1, edge_counter] = i
                        edge_counter += 1

            if keep_features == True:
                feature_matrix = torch.tensor(ligand_data)
            else:
                feature_matrix = torch.tensor(ligand_data[:, 3:])

       
            graph_data = Data(x=feature_matrix,
                              edge_index=edge_index,
                              y=label)

   
            data_list.append(graph_data)

        progress_bar: str = "[" + "#" * max_bars + "]"
        print(progress_bar, end='');
        print("(%5u / %5u)" % (molecule_count, molecule_count))

        hdf5_file.close();


        data, slices = self.collate(data_list)


        data.y = torch.tensor(data.y, dtype=torch.float32)

        torch.save((data, slices), self.processed_paths[0])


def main():

    train_dataset = PosteraDataset(
        root='./',
        hdf5_file_name='postera_protease2_pos_neg_train.hdf5')

    test_dataset = PosteraDataset(
        root='./',
        hdf5_file_name='postera_protease2_pos_neg_test.hdf5')

    val_dataset = PosteraDataset(
        root='./',
        hdf5_file_name='postera_protease2_pos_neg_val.hdf5')


if __name__ == '__main__':
    main()

