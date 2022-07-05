# Embeddings

The motivation here is to learn higher dimensional representations for our atoms. 

Currently, each atom is represented by a 19-dimensional feature vector, that contains a mix of boolean, categorical, and continuous types.
- Features currently include a 9-dimensional one-hot encoding of the atom type
- and a 10-dimensional vector of other descriptors e.g., valence, # of rings, etc.

Higher dimensional (and more homogenous) feature representations may be suitable for training other models for the classification task.
