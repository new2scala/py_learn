
import torch
import torch.nn as nn
from torch.autograd import Variable

class LstmNet(nn.Module):
    def __init__(self,
                 vocab, input_size, hidden_size,
                 cell_type='LSTM',
                 layer_num=2):
        super(LstmNet, self).__init__()

        self.vocab = vocab
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num

        self._embedding = nn.Embedding(vocab.vocab_size, input_size)
        self._layers = [ nn.LSTMCell(input_size, hidden_size) ]
        for i in range(1, layer_num):
            self._layers.append(
                nn.LSTMCell(hidden_size, hidden_size)
            )
        self._linear = nn.Linear(hidden_size, vocab.vocab_size)

        if torch.cuda.is_available():
            self.cuda()
            for layer in self._layers:
                layer.cuda()

    def forward(self, input, hidden_curr):
        x = self._embedding(input)
        hidden_var = Variable(torch.zeros(hidden_curr.size()))
        for i, layer in enumerate(self._layers):
            hidden_var[i] = layer(x, hidden_curr[i])
            x = hidden_var[i]
        x = self._linear(x)
        return x, hidden_var

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.layer_num, batch_size, self.hidden_size))