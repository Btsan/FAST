import math

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset



class LigandDataset(Dataset):
    """
    Custom Dataset object that opens the file path for the postera_protease2 datasets. 
    """
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.file = None
        with h5py.File(self.file_path,"r") as file:
            self.ligand_names = list(file.keys())
        
        
    def __len__(self):
        """ 
        Returns the total number of ligands. 
        """
        return len(self.ligand_names)
        
    def __getitem__(self, index):
        """
        Returns coordinates and/or features as well as labels for each ligand pose.
        """
        if self.file is None:
            self.file = h5py.File(self.file_path,"r")
        name = self.ligand_names[index]
        ligand = torch.tensor(self.file[name]["ligand"][:])
        label = torch.tensor(self.file[name].attrs["label"])

        return ligand, label