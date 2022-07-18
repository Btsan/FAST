import argparse

import torch
import torch.nn.functional as F
from torch import tensor, nn
from torch.nn.utils import rnn
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, csv_filepath, pad_len=100):
        super(SequenceDataset, self).__init__()
        self.pad_len = pad_len

        # read comma-separated integer sequences from file
        # also records the set of distinct values
        with open(csv_filepath, 'rt') as file:
            self.sequences = file.readlines()
            self.distinct = set()
            self.seq_len = []
            for i, seq in enumerate(self.sequences):
                seq = tuple(map(int, seq.split(',')))
                self.sequences[i] = seq
                self.distinct.update(set(seq))

    def __len__(self):
        return len(self.sequences)

    # returns the padded sequence and sequence length
    def __getitem__(self, i):
        seq = tensor(self.sequences[i], dtype=torch.long)
        padded_seq = torch.ones(self.pad_len, dtype=torch.long) * -1
        padded_seq[:len(seq)] = seq

        # increment by 1 so that padding is 0
        # all non-padding values will be nonzero
        padded_seq += 1
        return padded_seq, tensor(len(seq))

class LSTM(nn.Module):
    def __init__(self, hidden_size, num_embed, embed_dim=32, num_layers=1):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(num_embed, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, embed_dim)

    # input_seq: batch of padded sequences
    # input_len: list of sequence lengths
    def forward(self, input_seq, input_len):
        embed = self.embedding(input_seq)

        # pack padded batch of sequences
        packed = rnn.pack_padded_sequence(embed, input_len, batch_first=True, enforce_sorted=False)

        # forward pass through recurrent layers
        outputs, _ = self.lstm(packed)

        # unpack padded sequences
        unpacked, _ = rnn.pad_packed_sequence(outputs, batch_first=True, total_length=100)

        # forward pass through fully-connected layer
        outputs = self.fc(unpacked)

        return outputs

# x: shape(B, L, embed) where L is the sequence length
# multiply with embedding weights (C, embed) to get C-dimensional logits
# returns logits shape(B, L, C)
def get_logits(pred, embedding):
    logits = pred.matmul(embedding.t())
    return logits

def train(model, dataset, optimizer, device=None, epochs=1, batch_size=128, num_workers=4, dropout=0.3, print_every=100):
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)

    # one-cycle learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        1e-3,
        epochs=epochs, 
        steps_per_epoch=len(dataloader))

    for epoch in range(epochs):
        for it, batch in enumerate(dataloader):
            sequences, seq_len = batch
            optimizer.zero_grad()

            sequences = sequences.to(device)
            dropout_mask = torch.rand_like(sequences, dtype=torch.float) > dropout
            pred = model(sequences * dropout_mask, seq_len)

            # calculate logits for C classes
            logits = get_logits(pred, model.embedding.weight)

            # calculate loss, ignoring dropout token at index 0
            loss = F.cross_entropy(torch.flatten(logits[:, :-1], 0, 1), torch.flatten(sequences[:, 1:]), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print current progress and loss
            if it % print_every == 0:
                print(f' \t{it+1}/{len(dataloader)} \tloss={loss.sum().item()}')
        print(f'{epoch+1}/{epochs} done')
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train and saves autoregressive embedding models')
    parser.add_argument('--dataset', type=str, required=True, metavar='NAME', choices=['atom', 'descriptor'])
    parser.add_argument('--num_epochs', type=int, default=10, metavar='EPOCHS')
    parser.add_argument('--batch_size', type=int, default=64, metavar='BATCH')
    parser.add_argument('--hidden_dim', type=int, default=64, metavar='HIDDEN')
    parser.add_argument('--embed_dim', type=int, default=32, metavar='EMBED')
    parser.add_argument('--num_layers', type=int, default=1, metavar='LAYERS')
    parser.add_argument('--num_workers', type=int, default=0, metavar='WORKERS')
    parser.add_argument('--device', type=str, default=None, metavar='DEVICE')
    args = parser.parse_args()

    # create dataset
    dataset = SequenceDataset(f'data/{args.dataset}_sequences.csv')

    # create the model with as many embeddings as classes in the data
    num_distinct = len(dataset.distinct)
    model = LSTM(args.hidden_dim, num_distinct+1, embed_dim=args.embed_dim, num_layers=args.num_layers)
    print(f'{num_distinct} distinct values in dataset')


    # move device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device:
        device = torch.device(args.device)
    model = model.to(device)

    # create optimizer and begin training
    optimizer = torch.optim.Adam(model.parameters())
    model = train(
        model,
        dataset,
        optimizer,
        device,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    torch.save({
            'epoch': args.num_epochs,
            'dataset': args.dataset,
            'hidden_dim': args.hidden_dim,
            'embed_dim': args.embed_dim,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f'models/{args.dataset}_embedding_{args.embed_dim}.pt')