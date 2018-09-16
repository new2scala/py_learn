
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F
from torch_learn.rnn2.vocab import Vocab

def trace_size(name, value):
    print('{} size: {}'.format(name, value.size()))

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# def pack_padding(input_emb, input_lens):
#     #print(input_emb.size())
#     zipped = [(input_lens[i], input_emb[i]) for i in range(len(input_lens))]
#     zipped.sort(key = lambda tp: -tp[0])
#     #print(zipped)
#     for i, tp in enumerate(zipped):
#         input_lens[i] = tp[0]
#         input_emb[i] = tp[1]
#     res = pack_padded_sequence(input_emb, input_lens, batch_first=True)
#     return res

class GruNet(nn.Module):
    def __init__(self,
                 vocab, input_size, hidden_size,
                 layer_num=3):
        super(GruNet, self).__init__()

        self.vocab = vocab
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num

        self._embedding = nn.Embedding(len(vocab), input_size)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layer_num
            #bidirectional=False
        )

        self._linear = nn.Linear(hidden_size, len(vocab))

        if torch.cuda.is_available():
            self.cuda()
            self.gru.cuda()
            self._linear.cuda()


    def _step(self, input, input_lens, h):
        input_emb = self._embedding(input)
        #packed = pack_padded_sequence(input_emb, input_lens, batch_first=True)
        #packed_out, ht = self.gru(packed, h)
        #output, _ = pad_packed_sequence(packed_out, batch_first=True)

        output, ht = self.gru(input_emb, h)

        output = self._linear(output)
        return output, ht

    def _step_pred(self, input, h):
        if torch.cuda.is_available():
            input.cuda()
        sz = input.size()
        batch_size = sz[0]
        batch_len = sz[1]
        batch_lens = [batch_len for _ in range(batch_size)]
        input_emb = self._embedding(input)
        # packed = pack_padded_sequence(input_emb, batch_lens, batch_first=True)
        # packed_out, ht = self.gru(packed, h)
        # output, _ = pad_packed_sequence(packed_out, batch_first=True)
        output, ht = self.gru(input_emb, h)
        output = self._linear(output)
        return output, ht

    def _init_hidden(self, batch_size):
        h = torch.zeros(self.layer_num, batch_size, self.hidden_size, dtype=torch.float32)
        #c = torch.zeros(self.layer_num, batch_size, self.hidden_size, dtype=torch.float32)
        if torch.cuda.is_available():
            h = h.cuda()
            #c = c.cuda()
        return h
        # layer_params = []
        # for i in range(3):
        #     h = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32)
        #     c = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32)
        #     # trace_size('h_%d'%i, h)
        #     # trace_size('c_%d'%i, c)
        #     if torch.cuda.is_available():
        #         h = h.cuda()
        #         c = c.cuda()
        #     layer_params.append([h, c])
        # return layer_params

    def forward(self, input, input_lens):
        # initialize hidden
        batch_size = input.size(1)
        _h = self._init_hidden(batch_size)

        if torch.cuda.is_available():
            input = input.cuda()

        inputs = input
        output, _h = self._step(inputs, input_lens, _h)
        return output

    def sample(self, batch_size, max_len=140):
        # start_token = Variable(torch.zeros(batch_size).long())
        # start_token[:] = self.vocab.start_encoded()
        x = Variable(torch.zeros(1, batch_size).long())
        x[0][:] = self.vocab.start_encoded()

        _h = self._init_hidden(batch_size)
        res = [ ]
        finished = torch.zeros(batch_size).byte()
        if torch.cuda.is_available():
            finished = finished.cuda()
        for step in range(max_len):
            if torch.cuda.is_available():
                x = x.cuda()
            output, _h = self._step_pred(x, _h)
            prob = F.softmax(output, 2)
            prob = prob.view(batch_size, -1)
            x = torch.multinomial(prob, 1)
            res.append(x.view(-1, 1))

            x = Variable(x.data)
            END_samples = (x == self.vocab.start_encoded()).data
            if torch.cuda.is_available():
                END_samples = END_samples.cuda()
            finished = torch.ge(finished + END_samples, 1)
            if torch.prod(finished) == 1:
                break
            x = x.transpose(0, 1)

        res = torch.cat(res, 1)
        return res.data



    # def forward(self, input, hidden_curr):
    #     x = self._embedding(input)
    #     hidden_var = Variable(torch.zeros(hidden_curr.size()))
    #     for i, layer in enumerate(self._layers):
    #         hidden_var[i] = layer(x, hidden_curr[i])
    #         x = hidden_var[i]
    #     x = self._linear(x)
    #     return x, hidden_var

    # def init_hidden(self, batch_size):
    #     return Variable(torch.zeros(self.layer_num, batch_size, self.hidden_size))