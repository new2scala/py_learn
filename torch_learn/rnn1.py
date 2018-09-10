
import torch
import torch.nn as nn

input_size, hidden_size = 3, 4
batch_size = 1
rnn = nn.RNN(input_size, hidden_size)
# **input** of shape `(seq_len, batch, input_size)`
# input = torch.tensor([
#     [
#         [0.1, 0.2, 0.3],
#     ],
#     [
#         [0.4, 0.5, 0.6]
#     ]
# ])
seq_len = 2
#input = torch.randn(seq_len, batch_size, input_size)
input = torch.tensor([
    [
        [0.1, 0.2, 0.3],
    ],
    [
        [0.4, 0.5, 0.6]
    ]
])
# h0 = torch.randn(1, batch_size, hidden_size)
h0 = torch.tensor(
    [[[0.7, 0.8, 0.9, 1.0]]]
)

output, hn = rnn(input, h0)
print(input)
print('---------')
print(h0)
print('---------')
print(output.data)
print('---------')
print(hn.data)