import argparse

import h5py
import torch
import numpy as np

# get_id returns the index of the row in embed that is equal to vec
def get_id(embed, vec):
    if np.all(vec == 0):
        return -2
    assert vec.shape[-1] == embed.shape[-1], (vec.shape, embed.shape)
    element_comp = np.equal(embed, vec)
    row_comp = np.all(element_comp, axis=1)
    indices = np.argwhere(row_comp)
    if indices.size != 1:
        print(f'duplicate or no matching embeddings in {embed.shape}\nindices {indices}\n')
        return -1
    return indices.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='loads autoregressive embedding models and update h5py datasets with their embeddings')
    parser.add_argument('--dataset', type=str, required=True, metavar='HDF5_File')
    parser.add_argument('--atoms', type=lambda x: None if not x else str(x), default=None, metavar='Atom_Model')
    parser.add_argument('--descriptors', type=lambda x: None if not x else str(x), default=None, metavar='Descriptor_Model')
    parser.add_argument('--output', type=str, default='postera_protease2_embed.hdf5', metavar='Output_HDF5')
    args = parser.parse_args()
    if not args.atoms and not args.descriptors:
        raise Exception('Incorrect Arguments: at least one of --atoms and --descriptors must be defined')

    # load the saved model(s) onto CPU for its embedding matrix
    if args.atoms:
        atom_chkpt = torch.load(args.atoms, map_location=torch.device('cpu'))
        atom_embeddings = atom_chkpt['model']['embedding.weight'].numpy()
    if args.descriptors:
        desc_chkpt = torch.load(args.descriptors, map_location=torch.device('cpu'))
        desc_embeddings = desc_chkpt['model']['embedding.weight'].numpy()

    # load the map of vectors to their embedding ids
    atom_map = torch.load('data/atoms.pt').numpy()
    desc_map = torch.load('data/descriptors.pt').numpy()

    # read in the dataset to replace
    dataset = h5py.File(args.dataset, 'r')
    outfile = h5py.File(args.output, 'w')

    for i, name in enumerate(dataset.keys()):
        old_ligand = dataset[name]['ligand']

        # slice ligand data into components
        atom_coordinates = old_ligand[:, :3] # 100, 3
        atom_onehot = old_ligand[:, 3:12] # 100, 9
        atom_desc = np.hstack((old_ligand[:, 12:15], old_ligand[:, 16:])) # 100, 9
        atom_unchanged = np.expand_dims(old_ligand[:, 15], axis=-1) # 100, 1

        # print('.', end='')

        new_ligand = [atom_coordinates]

        # change atomic number components
        if args.atoms:
            new_atoms = []
            for vec in atom_onehot:
                embed_id = get_id(atom_map, vec)
                embed = np.zeros(atom_embeddings.shape[1]) if embed_id == -2 else atom_embeddings[embed_id + 1]
                new_atoms.append(embed)
            
            new_atoms = np.array(new_atoms) # 100, atom_embed_dim
            new_ligand.append(new_atoms)
        else:
            new_ligand.append(atom_onehot)

        # print('.', end='')

        # change descriptor components
        if args.descriptors:
            new_desc = []
            for vec in atom_desc:
                embed_id = get_id(desc_map, vec)
                embed = np.zeros(desc_embeddings.shape[1]) if embed_id == -2 else desc_embeddings[embed_id + 1]
                new_desc.append(embed)

            new_desc = np.array(new_desc) # 100, desc_embed_dim
            new_ligand += [new_desc, atom_unchanged]
        else:
            new_ligand.append(old_ligand[:, 12:])

        # print('.', end='')

        # save new ligand data as a matrix
        new_ligand = np.hstack(new_ligand)

        # write new ligand data with embeddings to file
        grp = outfile.create_group(name)
        copy_ligand = grp.create_dataset('ligand', data=new_ligand)

        # write label to file
        copy_label = grp.attrs.create('label', dataset[name].attrs['label'])

        # include dropout embeddings to handle unknown inputs
        if args.atoms:
            unk_atom = grp.create_dataset('unk_atom', data=np.expand_dims(atom_embeddings[0], axis=0))
        if args.descriptors:
            unk_desc = grp.create_dataset('unk_descriptor', data=np.expand_dims(desc_embeddings[0], axis=0))

        # print(f'({i}/{len(dataset)}) \t\t {old_ligand.shape}-->{new_ligand.shape}')

    dataset.close()
    outfile.close()