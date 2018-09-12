import unittest

from torch_learn.rnn2.vocab_impl import Vocab

class VocabTest(unittest.TestCase):

    VOC = Vocab('rnn2/tests/TestVocab.txt')

    def test_voc(self):
        voc = VocabTest.VOC
        self.assertEqual(len(voc), 6+2)
        de_idx = voc.lookup('De')
        print(de_idx)
        self.assertEqual(de_idx, 7)


if __name__ == '__main__':
    unittest.main()
