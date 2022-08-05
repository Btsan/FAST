#!/usr/bin/env python
# coding: utf-8


from torch.utils.data import Dataset # load dataset
import h5py # data reader for hdf5 files 
from torch_geometric.data import DataLoader as GeometricDataLoader
from tqdm import tqdm
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch


class PosteraDataset(Dataset):
    """
    opening and using the postera protease datasets for model
    
    args:
    self
    file_path: the file path to the datasets
    h5_file_driver: bool, false is standard
    cache_data: bool, if true: store data temporarily 
    features = bool, if true: parse features 
    
    """
    #cache_data=True
    def __init__(self, file_path, features=True
    ):
        super(PosteraDataset, self).__init__()
        self.file_path = file_path
        self.file = h5py.File(self.file_path, 'r') #reading files
        self.molecule_name = list(self.file.keys()) 
        
        self.data_dict = {} #store data if cache_data=True
        
    def __len__(self):
        return(len.self.molecule_name) # returning length
    
    def __getitem__(self, item):
        """
        
        returning the key from the data dictionary if cache_data=True
        
        
        if self.cache_data:
            
            if item in self.data_dict.keys(): 
                return self.data_dict[item] 
            
            else: 
                pass
        """
        
        name = self.molecule_name[index]
        ligand = self.file[name]["ligand"]
        label = self.file[name].attrs["label"]
        
        if features:
            atom_coordinates = ligand[:, :3] #taking all rows and columns except for the 4th 
            atom_features = ligand[:, 3:] # taking all the rows and 4th column
            return coordinates, features, label
        else:
            return ligand[:], label # return shallow copy of ligand, label
            

