
from torch_learn.rnn2.lstm_net import LstmNet
from torch_learn.rnn2.vocab import Vocab

from torch.utils.data import DataLoader

from torch_learn.rnn2.train_data_set import TrainDataset
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm

from rdkit import Chem

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

def dec_learning_rate(step, opt, dec_rate=0.05):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    prev_lrs = [ ]
    curr_lrs = [ ]
    for param_group in opt.param_groups:
        prev_lrs.append(param_group['lr'])
        param_group['lr'] *= (1 - dec_rate)
        curr_lrs.append(param_group['lr'])
    print('Step {}: learning rate decreased from {} to {}'.format(step, prev_lrs, curr_lrs))


def check_samples(samples, vocab):
    valid = 0
    valid_samples = [ ]
    for _, s in enumerate(samples.cpu().numpy()):
        smi = vocab.dec(s)
        if Chem.MolFromSmiles(smi):
            valid += 1
            valid_samples.append(smi)

    trace = "\n%d (%.2f%%) valid!"%(valid, valid*100.0/len(samples))
    for vs in valid_samples:
        trace += '\n\t%s'%vs
    tqdm.write(trace)

def batch_lens(vocab, batch):
    X_len = []
    for seq in batch:
        found = False
        for i, c in enumerate(seq):
            if c == vocab.end_encoded():
                X_len.append(i + 1 - 1) # -1: batch is complete, but input is batch[:-1]
                found = True
                break
        if not found:
            print("error")
    return X_len

import numpy as np

def perpare_pack_padding(input, input_lens):
    #print(input_emb.size())
    zipped = [(input_lens[i], input[i]) for i in range(len(input_lens))]
    zipped.sort(key = lambda tp: -tp[0])
    sz = input.size()
    mask = torch.ones(sz[0], sz[1]-1) # target sequence length = input length -1
    #print(zipped)
    for i, tp in enumerate(zipped):
        input_lens[i] = tp[0]
        mask[i][input_lens[i]:] = 0
        input[i] = tp[1]
    # res = pack_padded_sequence(input_emb, input_lens, batch_first=True)
    return input, input_lens, mask


def train_pass1():

    voc = Vocab('rnn2/tests/vocab.txt')

    lstm = LstmNet(
        vocab=voc,
        input_size=64,
        hidden_size=256
    )

    train_data = TrainDataset('rnn2/tests/train_data', voc)

    data = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
        drop_last=True,
        collate_fn=TrainDataset.normalize_batch
    )

    criterion = nn.NLLLoss(reduction='none')
    params = lstm.parameters()
    print(params)
    opt = optim.RMSprop(lstm.parameters(), lr=5e-3)

    for epoch in range(5):
        print('epoch: %d'%epoch)

        data_len = len(data)
        for step, batch in tqdm(enumerate(data), total=data_len):
            #
            # batch_long = batch.long()
            # print('batch size: {}'.format(batch_long.size()))
            dim1_size = batch.size(1)
            batch_input_lens = batch_lens(voc, batch)
            batch, batch_input_lens, batch_mask = perpare_pack_padding(batch, batch_input_lens)
            if torch.cuda.is_available():
                batch_mask = batch_mask.cuda()
            batch_input = batch.narrow(1, 0, dim1_size-1)
            batch_target = batch.narrow(1, 1, dim1_size-1)
            if torch.cuda.is_available():
                batch_target = batch_target.cuda()
            #batch_target = batch_target.transpose(0,1).transpose(1,2)

            def clos():
                opt.zero_grad()
                out = lstm(batch_input, batch_input_lens)
                # targets_ext = torch.zeros(out.size())
                # targets_reshaped = targets.view(-1, targets.size(1), 1)
                # targets_ext.scatter_(2, targets_reshaped, 1.0)
                # targets_ext = targets
                out = out.transpose(1,2)

                loss = criterion(out, batch_target)
                loss = loss * batch_mask
                # todo: mask out padding
                loss = loss.mean()
                loss.backward()
                if step % 50 == 0:
                    print('step {} loss: {}'.format(step, loss.item()))

                if step % 200 == 0:
                    dec_learning_rate(step, opt)
                #
                # if step % 200 == 199:
                    samples = lstm.sample(200)
                    check_samples(samples, voc)

            opt.step(clos)

            # if step % 50 == 0:
            #     print('-------------- {} loss: {}'.format(step, loss.item()))

if __name__ == '__main__':
    train_pass1()