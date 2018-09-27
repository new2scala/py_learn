
from torch_learn.rnn2.rnn_seq import RnnSeqNet
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

#torch.cuda.set_device(0)
# def dec_learning_rate(step, opt, dec_rate=0.03):
#     """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
#     prev_lrs = [ ]
#     curr_lrs = [ ]
#     for param_group in opt.param_groups:
#         prev_lrs.append(param_group['lr'])
#         param_group['lr'] *= (1 - dec_rate)
#         curr_lrs.append(param_group['lr'])
#     print('Step {}: learning rate decreased from {} to {}'.format(step, prev_lrs, curr_lrs))


def dec_learning_rate2(step, opt, curr_rate, min_lr):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    prev_lrs = [ ]
    curr_lrs = [ ]

    for param_group in opt.param_groups:
        prev_lrs.append(param_group['lr'])
        if curr_rate < 60:
            # param_group['lr'] *= (1 - dec_rate)
            #print('Learning rate unchanged!')
            param_group['lr'] *= 0.98
        elif curr_rate < 80:
            param_group['lr'] *= 0.96
        elif curr_rate < 90:
            param_group['lr'] *= 0.94
        else:
            param_group['lr'] *= 0.92

        lr = param_group['lr']
        if lr < min_lr:
            param_group['lr'] = min_lr
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

    valid_samples.sort()
    trace = ''
    factor = len(samples)/10
    for i, vs in enumerate(valid_samples):
        if i % factor == 0:
            trace += '\n\t_%s_'%vs
    rate = valid*100.0/len(samples)
    trace += "\n%d (%.2f%%) valid!"%(valid, rate)
    tqdm.write(trace)
    return rate


def batch_lens(vocab, batch):
    X_len = []
    for seq in batch.transpose(0, 1):
        found = False
        for i, c in enumerate(seq):
            if c == vocab.end_encoded():
                X_len.append(i + 1 - 1) # -1: batch is complete, but input is batch[:-1]
                found = True
                break
        if not found:
            print("error")
    return X_len


def perpare_pack_padding(input, input_lens):
    sz = input.size()
    zipped = [(input_lens[i], input[:,i]) for i in range(len(input_lens))]
    zipped.sort(key = lambda tp: -tp[0])
    mask = torch.ones(sz[0]-1, sz[1]) # target sequence length = input length -1

    for i, tp in enumerate(zipped):
        input_lens[i] = tp[0]
        if input_lens[i] < mask.size()[0]-1:
            mask[input_lens[i]:,i] = 0
        input[:,i] = tp[1]
    return input, input_lens, mask.byte()


def print_rates(rates):
    for ep, rates_ep in enumerate(rates):
        max_rate = max(rates_ep)
        avg = sum(rates_ep) / len(rates_ep)
        if ep < len(rates)-1:
            print('ep %d (max: %.1f%%, avg: %.1f%%): %s'%(ep, max_rate, avg, rates_ep))
        else:
            print('ep %d (max so far: %.1f%%, avg: %.1f%%): %s' % (ep, max_rate, avg, rates_ep))


def train_pass1():

    voc = Vocab('rnn2/tests/vocab.txt')

    rnn = RnnSeqNet(
        vocab=voc,
        input_size=128,
        hidden_size=512
    )

    train_data = TrainDataset('rnn2/tests/train_data', voc)

    data = DataLoader(
        train_data,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        collate_fn=TrainDataset.normalize_batch2
    )

    criterion = nn.CrossEntropyLoss(reduction='none')
    params = rnn.parameters()
    print(params)
    opt = optim.Adam(rnn.parameters(), lr=1.8e-3)

    rates = []

    for epoch in range(6):

        data_len = len(data)
        rates_ep = []
        rates.append(rates_ep)
        for step, batch in tqdm(enumerate(data), total=data_len):
            dim0_size = batch.size(0)
            batch_input_lens = batch_lens(voc, batch)
            batch, batch_input_lens, batch_mask = perpare_pack_padding(batch, batch_input_lens)
            batch_mask = batch_mask.byte()
            if torch.cuda.is_available():
                batch_mask = batch_mask.cuda()
            batch_input = batch.narrow(0, 0, dim0_size-1)
            batch_target = batch.narrow(0, 1, dim0_size-1)
            if torch.cuda.is_available():
                batch_target = batch_target.cuda()
                batch_input = batch_input.cuda()

            def clos():
                opt.zero_grad()
                #batch_size, batch_len = batch_input.size()[0], batch_input.size()[1]
                out = rnn(batch_input, batch_input_lens)

                out = out.permute(0,2,1)
                loss = criterion(out, batch_target)

                # loss = loss.sum(0)
                loss = loss.masked_select(batch_mask)
                # todo: mask out padding
                loss = loss.mean()
                loss.backward()
                if step % 50 == 0:
                    print('Epoch {} step {} loss: {}'.format(epoch, step, loss.item()))

                if step % 500 == 0:
                    rateX10 = []
                    for x in range(2):
                        samples = rnn.sample(200)
                        r = check_samples(samples, voc)
                        rateX10.append(r)
                    rate_avg = sum(rateX10) / len(rateX10)
                    rates_ep.append(rate_avg)
                    print_rates(rates)
                    #if step > 0:
                    dec_learning_rate2(step, opt, rate_avg, 1e-5)

            opt.step(clos)


if __name__ == '__main__':
    train_pass1()