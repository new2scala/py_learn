
import torch
import torch.nn as nn
from torch.autograd import Variable

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

    def forward(self, input):
        # initialize hidden
        trace_size('input', input)
        input_size0 = input.size(0)
        layer_params = [ ]
        for i in range(3):
            h = torch.zeros(input_size0, self.hidden_size, dtype=torch.float32)
            c = torch.zeros(input_size0, self.hidden_size, dtype=torch.float32)
            # trace_size('h_%d'%i, h)
            # trace_size('c_%d'%i, c)
            if torch.cuda.is_available():
                h = h.cuda()
                c = c.cuda()
            layer_params.append([h,c])

        input_size1 = input.size(1)
        outputs = []

        for _, input_t in enumerate(input.chunk(input_size1, dim=1)):
            reshaped = input_t.view(-1)
            # trace_size('reshaped', reshaped)
            if torch.cuda.is_available():
                reshaped = reshaped.cuda()
            layer_params, output = self._step(reshaped, layer_params)
            # trace_size('output', output)
            # trace_size('layer_params[0][0]', layer_params[0][0])
            # trace_size('layer_params[0][1]', layer_params[0][1])
            outputs.append(output)

        # print('outputs before stack {}'.format(len(outputs)))
        # print('\toutputs[0]: {}'.format(outputs[0].size()))
        outputs = torch.stack(outputs, 1)
        # trace_size('outputs after stack', outputs)
        outputs = outputs.squeeze(2)
        # trace_size('outputs after squeeze', outputs)
        return outputs

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