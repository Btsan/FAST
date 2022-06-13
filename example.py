import h5py

f = h5py.File('postera_protease2_pos_neg_train.hdf5', 'r')
ligand_names = list(f.keys())

for name in ligand_names:
	ligand = f[name]['ligand']
	label = f[name].attrs['label']

	atom_coordinates = ligand[:, :3]
	atom_coordinates.shape

	atom_features = ligand[:, 3:]
	atom_features.shape
	
	break