import random
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from ProcessingData import Vocab
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


class BuildDataset(Dataset):

    def __init__(self, bug_path, fix_path, bug_vocab_path, fix_vocab_path): # Read the text data and convert it into pairs of Tensors (bug_ids, fix_ids)
        self.bv = Vocab()
        self.bv.load_vocab(bug_vocab_path)

        self.fv = Vocab()
        self.fv.load_vocab(fix_vocab_path)

        self.data = []
        with open(bug_path, "r", encoding= 'utf-8') as fb, \
            open(fix_path, "r", encoding= 'utf-8') as ff:

            for b_line, f_line in zip(fb, ff):
                b_line = b_line.strip()
                f_line = f_line.strip()

                if not b_line or not f_line:
                    continue

                b_ids = self.bv.encode(b_line, add_sos_eos= False) + [self.bv.token2idx[self.bv.eos_token]]
                f_ids = self.fv.encode(f_line)

                b_tensor = torch.tensor(b_ids, dtype=torch.long)
                f_tensor = torch.tensor(f_ids, dtype=torch.long)
                self.data.append((b_tensor, f_tensor))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle = None):
        self.data = dataset
        self.bs = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indicies_lengths = [(i, len(self.data[i][0]), len(self.data[i][1])) for i in range(len(self.data))]

        if self.shuffle:
            random.shuffle(indicies_lengths)

        indicies_lengths.sort(key= lambda x : (x[1], x[2])) #Arrange the patterns in ascending order of length so that sentences of similar length are placed next to each other.

        batches = []
        for i in range(0, len(indicies_lengths), self.bs):
            batch = [x[0] for x in indicies_lengths[i : i + self.bs]]
            batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.data) + self.bs - 1) // self.bs


def pad_fn(batch): # used to pad sentences within the same batch to an equal length.
    bug_seq, fix_seq = zip(*batch)

    bug_len = torch.tensor([len(s) for s in bug_seq], dtype= torch.long)
    fix_len = torch.tensor([len(s) for s in fix_seq], dtype= torch.long)

    buggy_padded = pad_sequence(bug_seq, batch_first= True, padding_value= 0)
    fixed_padded = pad_sequence(fix_seq, batch_first= True, padding_value= 0)

    return buggy_padded, fixed_padded, bug_len, fix_len

    
  