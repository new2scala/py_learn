
import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, data_file, voc):
        self.voc = voc
        self._data = []
        with open(data_file, 'r') as f:
            for line in f:
                self._data.append(line.strip())

    def __getitem__(self, item):
        d = self._data[item]
        encoded = self.voc.enc(d)
        return encoded #torch.tensor(encoded)

    def __len__(self):
        return len(self._data)

    def __str__(self):
        return "Dataset(#%d)"%len(self)

    @classmethod
    def normalize_batch(cls, batch):
        max_len = max([len(seq) for seq in batch])
        #max_len += 1 # END token
        res = torch.zeros(len(batch), max_len).long()
        for i, seq in enumerate(batch):
            res[i, :len(seq)] = torch.tensor(seq)
            #res[i, len(seq)] = self.voc.end_encoded()
        return res


