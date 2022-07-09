import torch
import torch.nn as nn


class Voxelizer3D(nn.Module):
    """
    3-Dimensional Voxelization Layer

    Converts 3D coordinate information into volume information. 

    Orignal Voxelizer from FAST fusion model. Note that data can only be passed in one-at-a-time. If batching, parse data
    individually before passing to the guassian filter layer. 

    Forward pass expects 

    Example:
        >>> from img_util import Voxelizer3D
        >>> model = Voxelizer3D(use_cuda=True, verbose=0)

    """
    def __init__(self, feat_dim=19, vol_dim=48, ang_size=48, relative_size=True, atom_radius=1, use_cuda=True, verbose=0):
        super(Voxelizer3D, self).__init__()

        self.feat_dim = feat_dim
        self.vol_dim = vol_dim
        self.ang_size = ang_size
        self.relative_size = relative_size
        self.atom_radius = atom_radius
        self.use_cuda = use_cuda
        self.verbose = verbose

    def forward(self, xyz, feat, atom_radii=None):
        """
        Forward pass of Voxelizer3D layer. 

        Args:
            xyz:
                A torch tensor of shape (100,3) containing the 3D coordinate data (zero padded).
            feat:
                A torch tensor of shape (100,19) containing the 19 features for each of the 100 
                molecules (zero padded).
        
        Returns
        --------
        vol_data:
            A torch tensor of shape (vol_dim,vol_dim,vol_dim,19). The first three indices correspond 
            to the chosen volume dimension while the last to the number of features excluding the 
            coordinate data. 
        
        """
        # filter out zero padded rows
        mask = (xyz[:,0] != 0) & (xyz[:,1] != 0) & (xyz[:,2] != 0)
        xyz = xyz[mask]
        feat = feat[mask]

        # get 3d bounding box
        xmin, ymin, zmin = min(xyz[:,0]), min(xyz[:,1]), min(xyz[:,2])
        xmax, ymax, zmax = max(xyz[:,0]), max(xyz[:,1]), max(xyz[:,2])
        if self.relative_size:
            # voxel size (assuming voxel size is the same in all axis)
            vox_size = float(zmax - zmin) / float(self.vol_dim)
        else:
            vox_size = float(self.ang_size) / float(self.vol_dim)
            xmid, ymid, zmid = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
            xmin, ymin, zmin = xmid - (self.ang_size / 2), ymid - (self.ang_size / 2), zmid - (self.ang_size / 2)
            xmax, ymax, zmax = xmid + (self.ang_size / 2), ymid + (self.ang_size / 2), zmid + (self.ang_size / 2)

        # initialize vol data
        if self.use_cuda:
            vol_data = torch.cuda.FloatTensor(self.vol_dim, self.vol_dim, self.vol_dim, self.feat_dim).fill_(0)
        else:
            vol_data = torch.zeros((self.vol_dim, self.vol_dim, self.vol_dim, self.feat_dim)).float()

        # assign each atom to voxels
        for ind in range(xyz.shape[0]):
            x, y, z = xyz[ind, 0], xyz[ind, 1], xyz[ind, 2]
            if x < xmin or x > xmax or y < ymin or y > ymax or z < zmin or z > zmax:
                continue

            # compute van der Waals radius and atomic density, use 1 if not available
            if not atom_radii is None:
                vdw_radius = atom_radii[ind]
                atom_radius = 1 + vdw_radius * vox_size
            else:
                atom_radius = self.atom_radius

            cx = int((x-xmin) / (xmax-xmin) * (self.vol_dim-1))
            cy = int((y-ymin) / (ymax-ymin) * (self.vol_dim-1))
            cz = int((z-zmin) / (zmax-zmin) * (self.vol_dim-1))
            vx_from = max(0, int(cx-atom_radius))
            vx_to = min(self.vol_dim-1, int(cx+atom_radius))
            vy_from = max(0, int(cy-atom_radius))
            vy_to = min(self.vol_dim-1, int(cy+atom_radius))
            vz_from = max(0, int(cz-atom_radius))
            vz_to = min(self.vol_dim-1, int(cz+atom_radius))

            vol_feat = feat[ind,:].repeat(vz_to-vz_from+1, vy_to-vy_from+1, vx_to-vx_from+1, 1)
            vol_data[vz_from:vz_to+1, vy_from:vy_to+1, vx_from:vx_to+1, :] += vol_feat

        vol_data = vol_data.permute(3,0,1,2) #-> doesn't need as we already initialized 19x48x48x48
        return vol_data
    
    
