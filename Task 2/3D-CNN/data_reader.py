import math

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def getVoxelDataset(hdf_path, vol_dim=64):
    file = h5py.File(hdf_path,"r")
    file_cpy = dict()
    for name in file.keys():
        file_cpy[name] = (file[name]['ligand'][()], file[name].attrs['label'])
    file.close()
    dataset = VoxelDataset(file_cpy, vol_dim=vol_dim)
    return dataset

class LigandDataset(Dataset):
    """
    Custom Dataset object that opens the file path for the postera_protease2 datasets. 
    """
    def __init__(self, hdf_path, parse_features=False):
        super().__init__()
        self.hdf_path = hdf_path
        self.file = h5py.File(self.hdf_path,"r")
        self.ligand_names = tuple(self.file.keys())
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

class VoxelDataset(Dataset):
    """
    Custom Dataset object that opens the file path for the postera_protease2 datasets. 
    """
    def __init__(self, data_dict,
        vol_dim=32, ang_size=48, relative_size=True, atom_radius=1,
        kernel_size=11, sigma=1):
            super().__init__()
            # self.hdf_path = hdf_path
            # file = h5py.File(self.hdf_path,"r")
            # self.ligand_names = tuple(file.keys())

            # # warning: stupidity ahead
            # # save entire HDF5 file datasets into in-memory dict
            # self.file_cpy = dict()
            # for name in self.ligand_names:
            #     self.file_cpy[name] = (file[name]['ligand'][()], file[name].attrs['label'])
            # file.close()
            self.file_cpy = data_dict
            self.ligand_names = tuple(data_dict.keys())

            # voxel stuff
            self.vol_dim = vol_dim
            self.ang_size = ang_size
            self.relative_size = relative_size
            ligand, _ = self.file_cpy[self.ligand_names[0]]
            features = ligand[:,3:]
            self.feat_dim = features.shape[-1]

            # gaussian stuff
            self.kernel_size = [kernel_size] * 3
            self.sigma = [sigma] * 3

            self.padding = kernel_size // 2

            # Gaussian kernel is the product of the gaussian function of each dimension.
            kernel = 1
            meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in self.kernel_size])
            for size, std, mgrid in zip(self.kernel_size, self.sigma, meshgrids):
                mean = (size - 1) / 2
                kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

            # make sure sum of values in gaussian kernel equals 1.
            kernel = kernel / torch.sum(kernel)

            # reshape to depthwise convolutional weight
            kernel = kernel.view(1, 1, *kernel.size()) #-> doesn't need to add one more axis
            #kernel = kernel.view(1, *kernel.size())
            self.kernel = kernel.repeat(self.feat_dim, *[1] * (kernel.dim() - 1))
            self.kernel = self.kernel.cuda()
        
    def __len__(self):
        """ 
        Returns the total number of ligands. 
        """
        return len(self.ligand_names)
        
    def __getitem__(self, index):
        """
        Returns voxels (19, 32, 32, 32) and label
        """
        name = self.ligand_names[index]
        ligand, label = self.file_cpy[name]
        
        coords = ligand[:,:3]
        features = ligand[:,3:]

        mask = (coords[:,0] != 0) & (coords[:,1] != 0) & (coords[:,2] != 0)
        coords = coords[mask]
        features = torch.tensor(features[mask]).cuda()

        # get 3d bounding box
        xmin, ymin, zmin = min(coords[:,0]), min(coords[:,1]), min(coords[:,2])
        xmax, ymax, zmax = max(coords[:,0]), max(coords[:,1]), max(coords[:,2])
        if self.relative_size:
            # voxel size (assuming voxel size is the same in all axis)
            vox_size = (zmax - zmin) / self.vol_dim
        else:
            vox_size = self.ang_size / self.vol_dim
            xmid, ymid, zmid = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
            xmin, ymin, zmin = xmid - (self.ang_size / 2), ymid - (self.ang_size / 2), zmid - (self.ang_size / 2)
            xmax, ymax, zmax = xmid + (self.ang_size / 2), ymid + (self.ang_size / 2), zmid + (self.ang_size / 2)

        # initialize vol data
        vol_data = torch.zeros((self.vol_dim, self.vol_dim, self.vol_dim, self.feat_dim), dtype=torch.float32, device=torch.device('cuda'))

        # assign each atom to voxels
        for ind in range(coords.shape[0]):
            x, y, z = coords[ind, 0], coords[ind, 1], coords[ind, 2]
            if x < xmin or x > xmax or y < ymin or y > ymax or z < zmin or z > zmax:
                continue

            cx = int((x-xmin) / (xmax-xmin) * (self.vol_dim-1))
            cy = int((y-ymin) / (ymax-ymin) * (self.vol_dim-1))
            cz = int((z-zmin) / (zmax-zmin) * (self.vol_dim-1))
            vx_from = max(0, int(cx-1))
            vx_to = min(self.vol_dim-1, int(cx+1))
            vy_from = max(0, int(cy-1))
            vy_to = min(self.vol_dim-1, int(cy+1))
            vz_from = max(0, int(cz-1))
            vz_to = min(self.vol_dim-1, int(cz+1))

            vol_feat = features[ind,:].repeat(vz_to-vz_from+1, vy_to-vy_from+1, vx_to-vx_from+1, 1)
            vol_data[vz_from:vz_to+1, vy_from:vy_to+1, vx_from:vx_to+1, :] += vol_feat

        vol_data = vol_data.permute(3,0,1,2) #-> doesn't need as we already initialized 19x48x48x48

        blurred_voxels = F.conv3d(vol_data.unsqueeze(0), weight=self.kernel, groups=self.feat_dim, padding=self.padding).squeeze(0)
        return blurred_voxels, label