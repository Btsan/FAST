from time import time

from torch.utils.data import DataLoader

from data_reader import getVoxelDataset

def timeSingle(batch_sz, dataset):
    t_for = []
    ts = time()
    for i, data in enumerate(dataset):
        if i+1 in batch_sz:
            t_for.append(time()-ts)
            if i+1 == max(batch_sz):
                break
    print(t_for)
    return t_for

def timeMulti(batch_sz, dataset):
    t_enum = []
    for sz in batch_sz:
        loader = DataLoader(dataset, batch_size=sz, num_workers=4)
        ts = time()
        for i, batch in enumerate(loader):
            t_enum.append(time()-ts)
            print(t_enum[-1])
            break
    return t_enum

if __name__ == '__main__':
    dataset = getVoxelDataset('../../datasets/postera_protease2_pos_neg_train.hdf5')
    batch_sz = [16, 32, 64]

    # t_for = timeSingle(batch_sz, dataset)
    # assert len(t_for) == len(batch_sz), t_for
    # [118.69432473182678, 237.13410663604736, 475.0523509979248]

    t_enum = timeMulti(batch_sz, dataset)
    assert len(t_enum) == len(batch_sz), t_enum

    for sz, t_f, t_e in zip(batch_sz, t_for, t_enum):
        print(f'{sz}\t{t_f}\t{t_e}')
