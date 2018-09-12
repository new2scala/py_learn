
class Vocab():
    START_TOKEN = 'START'
    END_TOKEN = 'END'
    SPECIAL_TOKENS = [ START_TOKEN, END_TOKEN ]
    def __init__(self, from_file, max_len=140):
        self._init_from_file(from_file)
        # self._additional_chars = set()
        # self._chars = Vocab.SPECIAL_TOKENS
        # self._vocab_size = len(self._chars)
        # self._vocab = dict(zip(self._chars, range(len(self._chars))))
        # self._rev_vocab = { v : k for v, k in self._vocab.items() }
        # self._max_len = max_len
        #
        # if from_file:

    def _init_from_file(self, file):
        self._chars = []
        self._chars = self._chars + Vocab.SPECIAL_TOKENS
        with open(file, 'r') as f:
            chars = f.read().split()
            self._chars = self._chars + chars
            self._vocab_size = len(self._chars)
            self._vocab = dict(zip(self._chars, range(len(self._chars))))
            self._rev_vocab = { v : k for v, k in self._vocab.items() }

    def __len__(self):
        return self._vocab_size

    def __str__(self):
        return "Vocab ({} tokens): {}".format(len(self), self._chars)

    def lookup(self, char):
        return self._vocab[char]