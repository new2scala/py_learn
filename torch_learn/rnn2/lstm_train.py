
from torch_learn.rnn2.lstm_net import LstmNet
from torch_learn.rnn2.vocab import Vocab

from torch.utils.data import DataLoader

from torch_learn.rnn2.train_data_set import TrainDataset
import torch.optim as optim

import tqdm

def train_pass1():

    voc = Vocab('rnn2/tests/vocab.txt')

    lstm = LstmNet(
        vocab=voc,
        input_size=128,
        hidden_size=512
    )

    train_data = TrainDataset('rnn2/tests/train_data', voc)

    data = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True
    )

    opt = optim.Adam(lstm.parameters(), lr=1e-3)

    for epoch in range(5):
        print('step: %d'%epoch)

        for step, batch in tqdm(enumerate(data), total=len(data)):
            batch_long = batch.long()
            print('batch size: {}'.format(batch_long.size()))
            batch_size, seq_len = batch_long.size()
