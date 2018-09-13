
from torch_learn.rnn2.lstm_net import LstmNet
from torch_learn.rnn2.vocab import Vocab

from torch.utils.data import DataLoader

from torch_learn.rnn2.train_data_set import TrainDataset

def train_pass1():

    voc = Vocab('rnn2/tests/vocab.txt')

    lstm = LstmNet(
        vocab=voc,
        input_size=128,
        hidden_size=512
    )

    train_data = TrainDataset()

    data_loader = DataLoader('rnn2/tests/train_data')