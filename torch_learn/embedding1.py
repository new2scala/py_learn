
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

torch.manual_seed(2)

word2idx = { 'hello': 0, "world": 1 }
vocab_size = len(word2idx)
print(vocab_size)
embedding_dim = 5
embeds = nn.Embedding(vocab_size, embedding_dim)
lookup_tensor = torch.tensor([word2idx['hello']], dtype=torch.long)
hello_embedding = embeds(lookup_tensor)
print(hello_embedding)
