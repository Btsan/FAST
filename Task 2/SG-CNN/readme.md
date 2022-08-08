# Simple Graph Convolutional Neural Network

See [PyTorch Geometric for details](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SGConv)

Model saver not up and running at the moment. 

data_utils.py - used to process hdf5 files and prepare for a sgnn.

train_test.py - contains the main training and validation functions 

model.py - contains the sgnn model

main.py - main training script.

To change dataset access location and/or data file name, go to main.py, line 144-146 under setup_loaders and change the root and hdf5_file_name. 

To execute the script, locate your terminal and activate the virtual environment you have the required libraries ``` conda activate my-rdkit-env ``` (replace my-rdkit-env with your designated venv). After activating, run python main.py! 
