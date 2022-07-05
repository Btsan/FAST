from torch.utils.data import Dataset
import h5py

class LigandDataset(Dataset):
    """
    Custom Dataset object that opens the file path for the postera_protease2 datasets. 
    """
    def __init__(self, hdf_path, parse_features=False):
        super().__init__()
        self.hdf_path = hdf_path
        self.file = h5py.File(self.hdf_path,"r")
        self.ligand_names = list(self.file.keys())
        self.parse_features = parse_features
        
    def __len__(self):
        """ 
        Returns the total number of ligands. 
        """
        return len(self.ligand_names)
        
    def __getitem__(self, index):
        """
        Returns coordinates and/or features as well as labels for each ligand pose.
        """
        name = self.ligand_names[index]
        ligand = self.file[name]["ligand"]
        label = self.file[name].attrs["label"]
        
        
        if self.parse_features:
            coordinates = ligand[:,:3]
            features = ligand[:,3:]
            return coordinates, features, label
        else:
            return ligand[:], label