
from torch_learn.rnn2.train_data_set import TrainDataset
from torch_learn.rnn2.vocab import Vocab
import unittest
from torch.utils.data import DataLoader

class TrainDatasetTest(unittest.TestCase):

    VOCAB = Vocab('rnn2/tests/vocab.txt')
    TRAIN_DATA = TrainDataset('rnn2/tests/train_data.txt', VOCAB)

    def test_general(self):
        td = TrainDatasetTest.TRAIN_DATA

        self.assertEqual(len(td), 23)
        dl = DataLoader(
            td,
            batch_size=5,
            shuffle=False,
            drop_last=True,
            collate_fn=TrainDataset.normalize_batch
        )
        for i, batch in enumerate(dl):
            print(i)
            print(batch.size())

if __name__ == '__main__':
    unittest.main()