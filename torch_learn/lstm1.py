
import numpy as np
import torch

input_size, hidden_size, output_size = 1, 6, 1

epochs = 300
seq_len = 20

lr = 0.1

data_timesteps = np.linspace(2, 10, seq_len+1)
data = np.sin(data_timesteps)

# print(data)
data.resize((seq_len+1, 1))
# print(data)
#
# d2 = data[1:]
# print(d2)

import torch.nn as nn

import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size+hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size+hidden_size, output_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = F.tanh(self.i2o(combined))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


rnn = RNN(input_size, hidden_size, output_size)

# hidden = torch.zeros(1, hidden_size)
#
# for i in range(data.shape[0]):
#     input = torch.tensor([data[i]]).float()
#     output, next_hidden = rnn(input, hidden)
#     print(output)

criterion = nn.MSELoss()
l_rate = 0.01

import torch.optim as optim
opt = optim.SGD(rnn.parameters(), lr=l_rate)

x = data[:-1]
y = data[1:]
for ep in range(epochs):
    hidden = rnn.init_hidden()

    for i, xi in enumerate(x):
        rnn.zero_grad()
        input = torch.tensor([xi]).float()
        output, hidden = rnn(input, hidden)
        loss = criterion(output, torch.tensor([y[i]]).float())
        loss.backward()

        for p in rnn.parameters():
            p.data.add_(-l_rate * p.grad.data)

        print(loss.item())



# x = Variable(torch.Tensor(data[:-1]).float(), requires_grad=False)
# y = Variable(torch.Tensor(data[1:]).float(), requires_grad=False)
#
# w1 = torch.Tensor(input_size, hidden_size).float()
# import torch.nn.init as init
#
# init.normal(w1, 0.0, 0.4)
# w1 = Variable(w1, requires_grad=True)
#
# w2 = torch.Tensor(hidden_size, output_size).float()
# init.normal(w2, 0.0, 0.3)
# w2 = Variable(w2, requires_grad=True)
#
# import torch.nn.functional as F
# import torch.nn as nn
#
# def forward(input, context, w1, w2):
#     xh = torch.cat((input, context), 1)
#     context = F.tanh(nn.Linear(xh, w1))
#     out = nn.Linear()