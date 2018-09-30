
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

trigrams = [
    ([test_sentence[i], test_sentence[i+1]], test_sentence[i+2]) for i in range(len(test_sentence)-2)
]

print(trigrams[:3])

vocab = set(test_sentence)
print(vocab)

word_to_ix = { word: i for i, word in enumerate(vocab) }
idx_to_word = { word_to_ix[w]: w for w in vocab}

import torch.nn as nn
import torch.nn.functional as F
class NGramMod(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramMod, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        hidden_size = 128
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs).view(1, -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return out, log_probs


losses = []
loss_func = nn.NLLLoss()
model = NGramMod(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)

import torch.optim as optim
import torch
optimizer = optim.SGD(model.parameters(), lr=1e-3)

def test_emb(model):
    words = ['sum', 'my']
    idxs = torch.tensor([word_to_ix[w] for w in words], dtype=torch.long)
    _, prob = model(idxs)
    _, predicted = torch.max(prob, 1)
    output = idx_to_word[predicted.item()]
    print("   " + output)


for epoch in range(1000):
    total_loss = 0

    for context, target in trigrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        model.zero_grad()
        _, log_probs = model(context_idxs)

        tgt = torch.tensor([word_to_ix[target]], dtype=torch.long)
        loss = loss_func(log_probs, tgt)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 10 == 0:
        print("Loss at epoch {}: {}".format(epoch, total_loss))
        test_emb(model)
    losses.append(total_loss)

#print(losses)