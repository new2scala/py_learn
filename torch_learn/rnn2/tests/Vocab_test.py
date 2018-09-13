import unittest

from torch_learn.rnn2.vocab import Vocab
from unittest_data_provider import data_provider

class VocabTest(unittest.TestCase):

    VOC1 = Vocab('rnn2/tests/TestVocab.txt')

    def test_voc(self):
        voc = VocabTest.VOC1
        self.assertEqual(len(voc), 6+2)
        de_idx = voc.lookup('De')
        print(de_idx)
        self.assertEqual(de_idx, 7)

    VOC2 = Vocab('rnn2/tests/vocab.txt')

    VOC2_TOKENIZE_TESTDATA = lambda: (
        ('O(BrBr)', ['O', '(', 'R', 'R', ')', 'END']),
        ('O=C', ['O', '=', 'C', 'END']),
        ('OBrC', ['O', 'R', 'C', 'END']),
        ('OClC', ['O', 'L', 'C', 'END']),
        ('O[nH]C', ['O', '[nH]', 'C', 'END']),
    )

    @data_provider(VOC2_TOKENIZE_TESTDATA)
    def test_tokenize(self, data_line, expected_output):
        voc = VocabTest.VOC2
        output = voc.tokenize(data_line)
        self.assertEqual(output, expected_output)

if __name__ == '__main__':
    unittest.main()
