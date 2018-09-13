
from torch_learn.rnn2.lstm_net import LstmNet
from torch_learn.rnn2.vocab import Vocab
from torch_learn.rnn2.train_data_set import TrainDataset
import unittest
from torch.utils.data import DataLoader

class LstmNetTest(unittest.TestCase):

    VOCAB = Vocab('rnn2/tests/vocab.txt')
    LN = LstmNet(VOCAB, 128, 256)
    TRAIN_DATA = TrainDataset('rnn2/tests/train_data.txt', VOCAB)

    def test_forward(self):

        lstm_net = LstmNetTest.LN

        data = DataLoader(
            LstmNetTest.TRAIN_DATA,
            batch_size=5,
            shuffle=False,
            collate_fn=TrainDataset.normalize_batch
        )

        for i, batch in enumerate(data):
            print(batch.size())
            print(batch[0])
            outputs = lstm_net(batch)
            print(outputs.size())


if __name__ == "__main__":
    unittest.main()
