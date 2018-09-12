
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

class Sequence(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(Sequence, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)

    def _step(self, input_data, h_t, c_t, h_t2, c_t2):
        h_t, c_t = self.lstm1(input_data, (h_t, c_t))
        h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        output = self.linear(h_t2)
        return h_t, c_t, h_t2, c_t2, output

    def forward(self, input, future=0):
        outputs = []
        input_size0 = input.size(0)

        h_t = torch.zeros(input_size0, self.hidden_size, dtype=torch.double)
        c_t = torch.zeros(input_size0, self.hidden_size, dtype=torch.double)
        h_t2 = torch.zeros(input_size0, self.hidden_size, dtype=torch.double)
        c_t2 = torch.zeros(input_size0, self.hidden_size, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t, h_t2, c_t2, output = self._step(input_t, h_t, c_t, h_t2, c_t2)
            outputs += [output]

        for i in range(future): # should we predict the future
            h_t, c_t, h_t2, c_t2, output = self._step(output, h_t, c_t, h_t2, c_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)

    torch.manual_seed(seed)

    data_split = 10
    data = torch.load('train.pt')
    input = torch.from_numpy(data[data_split:, :-1])
    target = torch.from_numpy(data[data_split:, 1:])

    test_start = 800
    test_input = torch.from_numpy(data[:data_split, test_start:-1])
    test_target = torch.from_numpy(data[:data_split, test_start+1:])

    seq = Sequence(1, 51, 1)
    seq.double()

    criterion = nn.MSELoss()
    opt = optim.LBFGS(seq.parameters(), lr = 0.8)

    # start training
    for i in range(8):
        print('step: %d'%(i))

        def clos():
            opt.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss: {}'.format(loss.item()))
            loss.backward()
            return loss

        opt.step(clos)

        # start predict
        with torch.no_grad():
            future = 500
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss: {}'.format(loss.item()))
            y = pred.detach().numpy()

        # visualize

        plt.figure(figsize=(30, 10))
        plt.title('Predict future values', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        size1 = input.size(1)-test_start
        def draw(yi, color):
            plt.plot(np.arange(size1), yi[:size1], color, linewidth=2.0)
            plt.plot(np.arange(size1, size1+future), yi[size1:], color+':', linewidth=2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')

        plt.show()

    print("done")
        #plt.close()
        # plt.savefig('predict%d.pdf'%i)
        # plt.close()