
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F
from torch_learn.rnn2.vocab import Vocab

def trace_size(name, value):
    print('{} size: {}'.format(name, value.size()))

class LstmNet(nn.Module):
    def __init__(self,
                 vocab, input_size, hidden_size,
                 cell_type='LSTM'):
                 #layer_num=3):
        super(LstmNet, self).__init__()

        self.vocab = vocab
        self.input_size = input_size
        self.hidden_size = hidden_size

        self._embedding = nn.Embedding(len(vocab), input_size)
        self._lstm1 = nn.LSTMCell(input_size, hidden_size)
        self._lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self._lstm3 = nn.LSTMCell(hidden_size, hidden_size)
        self._linear = nn.Linear(hidden_size, len(vocab))

        if torch.cuda.is_available():
            self.cuda()
            self._lstm1.cuda()
            self._lstm2.cuda()
            self._lstm3.cuda()
            self._linear.cuda()


    def _step(self, input, layer_params):
        input = self._embedding(input)

        h0_t, c0_t = layer_params[0]
        h0_t, c0_t = self._lstm1(input, (h0_t, c0_t))
        h1_t, c1_t = layer_params[1]
        h1_t, c1_t = self._lstm2(h0_t, (h1_t, c1_t))
        h2_t, c2_t = layer_params[2]
        h2_t, c2_t = self._lstm2(h2_t, (h2_t, c2_t))
        output = self._linear(h2_t)
        new_params = [
            [h0_t, c0_t],
            [h1_t, c1_t],
            [h2_t, c2_t]
        ]
        return new_params, output

    def _init_hidden(self, batch_size):
        layer_params = []
        for i in range(3):
            h = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32)
            c = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32)
            # trace_size('h_%d'%i, h)
            # trace_size('c_%d'%i, c)
            if torch.cuda.is_available():
                h = h.cuda()
                c = c.cuda()
            layer_params.append([h, c])
        return layer_params

    def forward(self, input, input_lens):
        # initialize hidden
        #trace_size('input', input)
        batch_size = input.size(0)
        layer_params = self._init_hidden(batch_size)

        input_size1 = input.size(1)
        outputs = []
        if torch.cuda.is_available():
            input = input.cuda()
        input_emb = self._embedding(input)

        # input_padded = torch.nn.utils.rnn.pack_padded_sequence(
        #     input_emb, input_lens, batch_first=True)
        # print(input_padded.size())

        inputs = input.chunk(input_size1, dim=1)
        for _, input_t in enumerate(inputs):
            reshaped = input_t.view(-1)
            # trace_size('reshaped', reshaped)
            if torch.cuda.is_available():
                reshaped = reshaped.cuda()
            layer_params, output = self._step(reshaped, layer_params)
            # trace_size('output', output)
            # trace_size('layer_params[0][0]', layer_params[0][0])
            # trace_size('layer_params[0][1]', layer_params[0][1])
            output = F.log_softmax(output, 1)
            outputs.append(output)

        # print('outputs before stack {}'.format(len(outputs)))
        # print('\toutputs[0]: {}'.format(outputs[0].size()))
        outputs = torch.stack(outputs, 2)
        # trace_size('outputs after stack', outputs)
        outputs = outputs.squeeze(2)
        # trace_size('outputs after squeeze', outputs)
        return outputs

    def sample(self, batch_size, max_len=140):
        start_token = Variable(torch.zeros(batch_size).long())
        start_token[:] = self.vocab.start_encoded()
        x = start_token

        layer_params = self._init_hidden(batch_size)
        res = [ ]
        finished = torch.zeros(batch_size).byte()
        if torch.cuda.is_available():
            finished = finished.cuda()
        for step in range(max_len):
            if torch.cuda.is_available():
                x = x.cuda()
            layer_params, output = self._step(x, layer_params)
            prob = F.softmax(output, 1)
            x = torch.multinomial(prob, 1).view(-1)
            res.append(x.view(-1, 1))

            x = Variable(x.data)
            END_samples = (x == self.vocab.start_encoded()).data
            if torch.cuda.is_available():
                END_samples = END_samples.cuda()
            finished = torch.ge(finished + END_samples, 1)
            if torch.prod(finished) == 1:
                break
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