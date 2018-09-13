
from torch_learn.rnn2.train_data_set import TrainDataset
from torch_learn.rnn2.vocab import Vocab
import unittest

class TrainDatasetTest(unittest.TestCase):

    VOCAB = Vocab('rnn2/tests/vocab.txt')
    TRAIN_DATA = TrainDataset('rnn2/tests/train_data.txt', VOCAB)

    def test_general(self):
        td = TrainDatasetTest.TRAIN_DATA

        self.assertEqual(len(td), 23)

        for d in td:
            print(d)


if __name__ == '__main__':
    unittest.main()