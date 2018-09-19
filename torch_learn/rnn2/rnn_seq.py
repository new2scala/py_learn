
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RnnSeqNet(nn.Module):
    def __init__(self,
                 vocab, input_size, hidden_size,
                 cell_type='LSTM',
                 layer_num=3):
        super(RnnSeqNet, self).__init__()

        self.vocab = vocab
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num

        if cell_type == 'LSTM':
            self.is_lstm = True
        else:
            self.is_lstm = False

        self._embedding = nn.Embedding(len(vocab), input_size)

        if self.is_lstm:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=layer_num
            )
        else:
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=layer_num
            )

        self._linear = nn.Linear(hidden_size, len(vocab))

        if torch.cuda.is_available():
            self.cuda()
            if self.is_lstm:
                self.lstm.cuda()
            else:
                self.gru.cuda()
            self._linear.cuda()

    def _step(self, input, input_lens, pt):
        input_emb = self._embedding(input)
        packed = pack_padded_sequence(input_emb, input_lens, batch_first=False)

        if self.is_lstm:
            packed_out, pt1 = self.lstm(packed, pt)
        else:
            packed_out, pt1 = self.gru(packed, pt)
        output, _ = pad_packed_sequence(packed_out, batch_first=False)

        output = self._linear(output)
        return output, pt1

    def _step_pred(self, input, pt):
        if torch.cuda.is_available():
            input.cuda()
        input_emb = self._embedding(input)

        if self.is_lstm:
            output, pt1 = self.lstm(input_emb, pt)
        else:
            output, pt1 = self.gru(input_emb, pt)
        output = self._linear(output)
        return output, pt1

    def _init_hidden(self, batch_size):

        if self.is_lstm:
            h = torch.zeros(self.layer_num, batch_size, self.hidden_size, dtype=torch.float32)
            c = torch.zeros(self.layer_num, batch_size, self.hidden_size, dtype=torch.float32)
            if torch.cuda.is_available():
                h = h.cuda()
                c = c.cuda()
            p0 = (h, c)
        else:
            p0 = torch.zeros(self.layer_num, batch_size, self.hidden_size, dtype=torch.float32)
            if torch.cuda.is_available():
                p0 = p0.cuda()

        return p0

    def forward(self, input, input_lens):
        # initialize hidden
        batch_size = input.size(1)
        _p = self._init_hidden(batch_size)

        if torch.cuda.is_available():
            input = input.cuda()

        inputs = input
        output, _ = self._step(inputs, input_lens, _p)
        return output

    def sample(self, batch_size, max_len=140):
        # start_token = Variable(torch.zeros(batch_size).long())
        # start_token[:] = self.vocab.start_encoded()
        x = Variable(torch.zeros(1, batch_size).long())
        x[0][:] = self.vocab.start_encoded()

        _p = self._init_hidden(batch_size)
        res = [ ]
        finished = torch.zeros(batch_size).byte()
        if torch.cuda.is_available():
            finished = finished.cuda()
        for step in range(max_len):
            if torch.cuda.is_available():
                x = x.cuda()
            output, _p = self._step_pred(x, _p)
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
