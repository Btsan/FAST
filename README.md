# Fusion models for Atomic and molecular STructures (FAST)
Reimplementation of the original work at https://github.com/LLNL/FAST. Predicting protein-ligand binding affinity in PyTorch.

### Task 1
Using SMILES string or [rdkit descriptors](https://www.rdkit.org/docs/source/rdkit.ML.Descriptors.MoleculeDescriptors.html) of ligands.

### Task 2
Using 3D ligand data. Sample code for reading HDF5 data is [available](data/ligands.py).

In the original paper:
- 3D Convolution networks are used on a voxelized representation of ligands.
- Spatial-graph Convolution networks are used the spatial-graph representation.

### Optional Fusion task:
Combine 2 different feature representations to do the final prediction.

### Optional Comprehensive Model Analysis task:
Analyse the following:
- input feature importance
- spatial region relevance
- performance per compound group (clustered by SMILES/fingerprint/descriptors/etc.)
