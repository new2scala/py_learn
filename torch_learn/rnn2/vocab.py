
import re

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
            self._rev_vocab = { v : k for k, v in self._vocab.items() }

    def __len__(self):
        return self._vocab_size

    def __str__(self):
        return "Vocab ({} tokens): {}".format(len(self), self._chars)

    def lookup(self, char):
        return self._vocab[char]

    def start_encoded(self):
        return self.lookup(Vocab.START_TOKEN)

    def end_encoded(self):
        return self.lookup(Vocab.END_TOKEN)

    REGEX_BRACKETS = re.compile('(\[[^\[\]]{1,6}\])')
    TOKEN_CL = 'Cl'
    TOKEN_BR = 'Br'
    TOKEN_CL_REPL = 'L'
    TOKEN_BR_REPL = 'R'


    def tokenize(self, data_line):
        data_line = data_line\
            .replace(Vocab.TOKEN_CL, Vocab.TOKEN_CL_REPL)\
            .replace(Vocab.TOKEN_BR, Vocab.TOKEN_BR_REPL)
        parts = Vocab.REGEX_BRACKETS.split(data_line)
        res = [Vocab.START_TOKEN]
        for part in parts:
            if part.startswith('['):
                res.append(part)
            else:
                flatten = [c for c in part]
                res = res + flatten
        res.append(Vocab.END_TOKEN)
        return res

    def enc(self, data_line):
        tokens = self.tokenize(data_line)
        return [self.lookup(t) for t in tokens]

    def dec(self, seq):
        tokens_b4_end = []
        for s in seq:
            if s == self.end_encoded():
                break
            tokens_b4_end.append(self._rev_vocab[s])
        res = ''.join(tokens_b4_end)
        smi = res\
            .replace(Vocab.TOKEN_CL_REPL, Vocab.TOKEN_CL)\
            .replace(Vocab.TOKEN_BR_REPL, Vocab.TOKEN_BR)
        return smi
