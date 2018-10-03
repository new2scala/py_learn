import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import sys
import torch.optim as optim

HOST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def batch_lens(batch, end_token):
    X_len = []
    batch = batch.transpose(0, 1)
    for seq in batch:
        found = False
        for i, c in enumerate(seq):
            if c == end_token:
                X_len.append(i + 1)  # -1: batch is complete, but input is batch[:-1]
                found = True
                break
        if not found:
            raise RuntimeError('Error: end token {} not found!'.format(end_token))
    return X_len


def prepare_pack_padding(input_data, end_token, is_propagation=False):
    if is_propagation:
        input_data = input_data.transpose(0, 1)

    input_lens = batch_lens(
        batch=input_data,
        end_token=end_token
    )

    sz = input_data.size()
    zipped = [(input_lens[i], input_data[:, i]) for i in range(len(input_lens))]
    zipped.sort(key=lambda tp: -tp[0])
    mask = torch.ones(sz[0]-1, sz[1])  # target sequence length = input length -1

    sorted_input_data = torch.zeros(input_data.size()).long().to(HOST_DEVICE)
    for i, tp in enumerate(zipped):
        input_lens[i] = tp[0]
        if input_lens[i] < mask.size()[0]-1:
            mask[input_lens[i]:, i] = 0
        sorted_input_data[:, i] = tp[1]
        # input_data[:, i] = tp[1]
    return sorted_input_data, input_lens, mask.byte()


def sort_by_len(input_data, loss, end_token, is_propagation=False):
    if is_propagation:
        input_data = input_data.transpose(0, 1)

    input_lens = batch_lens(
        batch=input_data,
        end_token=end_token
    )

    sz = input_data.size()
    zipped = [(input_lens[i], loss[:, i]) for i in range(len(input_lens))]
    zipped.sort(key=lambda tp: -tp[0])
    mask = torch.ones(sz[0]-1, sz[1])  # target sequence length = input length -1

    for i, tp in enumerate(zipped):
        input_lens[i] = tp[0]
        if input_lens[i] < mask.size()[0]-1:
            mask[input_lens[i]:, i] = 0
        loss[:, i] = tp[1]
    return loss


class RnnSeqNet(nn.Module):
    def __init__(self,
                 dictionary, rev_dictionary, input_size, hidden_size, out_sz,
                 learning_rate,
                 criterion,
                 cell_type='LSTM',
                 layer_num=3):
        super(RnnSeqNet, self).__init__()

        self.dictionary = dictionary
        self.rev_dictionary = rev_dictionary
        self.criterion = criterion
        self.out_sz = out_sz

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num

        if cell_type == 'LSTM':
            self.is_lstm = True
        else:
            self.is_lstm = False

        vocab_size = len(dictionary)+1
        self.end_token = 0
        self._embedding = nn.Embedding(vocab_size, input_size)

        if self.is_lstm:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=layer_num
            )
        else:
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=layer_num
            )

        self._linear = nn.Linear(hidden_size, out_sz)
        self._softmax = nn.LogSoftmax(dim=2)

        # if torch.cuda.is_available():
        self.to(HOST_DEVICE)
        if self.is_lstm:
            self.lstm.to(HOST_DEVICE)
        else:
            self.gru.to(HOST_DEVICE)
        self._linear.to(HOST_DEVICE)

        self._optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def get_optimizer(self):
        """
        return optimizer of this rnn
        :return:
        """
        return self._optimizer

    def _step(self, input, input_lens, pt, pack_padding):
        input_emb = self._embedding(input)

        if pack_padding:
            in_data = pack_padded_sequence(input_emb, input_lens, batch_first=False)
        else:
            in_data = input_emb

        if self.is_lstm:
            out_data, pt1 = self.lstm(in_data, pt)
        else:
            out_data, pt1 = self.gru(in_data, pt)

        if pack_padding:
            out_data, _ = pad_packed_sequence(out_data, batch_first=False)

        output = self._linear(out_data)
        output = self._softmax(output)
        out_sz = output.size()
        last_out = Variable(torch.zeros(out_sz[1], out_sz[2]))
        for i, lens in enumerate(input_lens):
            last_out[i,:] = output[lens-2,i]
        return last_out, pt1

    # def _step_pred(self, input, pt):
    #     input_emb = self._embedding(input)
    #
    #     if self.is_lstm:
    #         output, pt1 = self.lstm(input_emb, pt)
    #     else:
    #         output, pt1 = self.gru(input_emb, pt)
    #     output = self._linear(output)
    #     return output, pt1

    def _init_hidden(self, batch_size):

        if self.is_lstm:
            h = torch.zeros(self.layer_num, batch_size, self.hidden_size, dtype=torch.float32).to(HOST_DEVICE)
            c = torch.zeros(self.layer_num, batch_size, self.hidden_size, dtype=torch.float32).to(HOST_DEVICE)
            p0 = (h, c)
        else:
            p0 = torch.zeros(self.layer_num, batch_size, self.hidden_size, dtype=torch.float32).to(HOST_DEVICE)

        return p0

    def propagation(self, input_data, back_propagation=True):
        """
        provide interface to PolicyGradient
        :param input:
        :param back_propagation:
        :return:
        """
        # TODO: in RL part, the input is from sample, perhaps
        # we can get the input_lens, batch_target and batch_mask
        # from input, need @Jiaji help to check

        input_data = input_data.transpose(0, 1)
        input_lens = batch_lens(
            batch=input_data,
            end_token=self.end_token
        )
        dim0_size = input_data.size(0)
        batch_input = input_data.narrow(0, 0, dim0_size - 1).to(HOST_DEVICE)
        batch_target = input_data.narrow(0, 1, dim0_size - 1).to(HOST_DEVICE)
        batch_mask = torch.zeros(batch_input.size())
        for i, len in enumerate(input_lens):
            batch_mask[:len, i] = 1.0
        batch_mask = batch_mask.to(HOST_DEVICE)

        return self._forward(
            batch_input=batch_input,
            batch_target=batch_target,
            input_lens=input_lens,
            batch_mask=batch_mask,
            is_propagation=True
        )

    def _forward(self, batch_input, batch_target, input_lens, batch_mask, is_propagation=False):
        # dim0_size = batch.size(0)
        # batch_mask = batch_mask.byte().to(HOST_DEVICE)
        # batch_input = batch.narrow(0, 0, dim0_size - 1).to(HOST_DEVICE)
        # batch_target = batch.narrow(0, 1, dim0_size - 1).to(HOST_DEVICE)

        batch_size = batch_input.size(1)
        _p = self._init_hidden(batch_size)

        inputs = batch_input.to(HOST_DEVICE)

        pack_padding = not is_propagation
        output, _ = self._step(inputs, input_lens, _p, pack_padding=pack_padding)

        out = output  # output.permute(0, 2, 1)
        loss = self.criterion(out, batch_target.view(-1))
        # if not is_propagation:
        #     loss = loss.masked_select(batch_mask)
        # else:
        #     loss = loss*batch_mask.float()
        #     loss = -loss.sum(0)

        return out, loss

    def forward(self, batch, is_propagation=False):
        # initialize hidden

        input_data = batch[0]
        target_cat = batch[1]
        input_data, input_lens, batch_mask = prepare_pack_padding(
            input_data=input_data,
            end_token=self.end_token,
            is_propagation=is_propagation
        )
        dim0_size = input_data.size(0)
        batch_mask = batch_mask.byte().to(HOST_DEVICE)
        batch_input = input_data  # batch.narrow(0, 0, dim0_size - 1).to(HOST_DEVICE)
        batch_target = target_cat # batch.narrow(0, 1, dim0_size - 1).to(HOST_DEVICE)

        return self._forward(
            batch_input=batch_input,
            batch_target=batch_target,
            input_lens=input_lens,
            batch_mask=batch_mask,
            is_propagation=is_propagation
        )

    def save_state(self, target_file):
        if self.is_lstm:
            torch.save(self.state_dict(), target_file)
        else:
            torch.save(self.state_dict(), target_file)

    def load_rnn_training_state(self, state_file):
        if self.is_lstm:
            self.load_state_dict(torch.load(state_file))
        else:
            self.load_state_dict(torch.load(state_file))

    def disable_gradients(self):
        """
        disable the gradients for rnn
        :return:
        """
        params = None
        if (self.is_lstm):
            params = self.lstm.parameters()
        else:
            params = self.gru.parameters()
        for parameter in params:
            parameter.requires_grad = False

    def enable_gradients(self):
        """
        disable the gradients for rnn
        :return:
        """
        params = None
        if (self.is_lstm):
            params = self.lstm.parameters()
        else:
            params = self.gru.parameters()
        for parameter in params:
            parameter.requires_grad = True

    # def sample(self, batch_size, start_token, end_token, max_len=140, return_loss=False):
    #     # start_token = Variable(torch.zeros(batch_size).long())
    #     # start_token[:] = self.vocab.start_encoded()
    #     x = Variable(torch.zeros(1, batch_size).long()).to(HOST_DEVICE)
    #     x[0][:] = start_token
    #
    #     _p = self._init_hidden(batch_size)
    #     res = []
    #     finished = torch.zeros(batch_size).byte().to(HOST_DEVICE)
    #
    #     # if return_loss:
    #     #     loss = torch.zeros(1, batch_size).to(HOST_DEVICE)
    #     #     loss_mask = torch.ones(1, batch_size).to(HOST_DEVICE)
    #     # if return_loss:
    #     #     self.disable_gradients()
    #
    #     for step in range(max_len):
    #         output, _p = self._step_pred(x, _p)
    #         prob = F.softmax(output, 2)
    #         prob = prob.view(batch_size, -1)
    #         x = torch.multinomial(prob, 1)
    #         res.append(x.view(-1, 1))
    #
    #         # if return_loss:
    #         #     loss_step = self.criterion(output.permute(0, 2, 1), x.permute(1, 0))
    #         #     loss_step = loss_step * loss_mask
    #         #     loss += loss_step
    #
    #         x = Variable(x.data)
    #         END_samples = (x == end_token).data.to(HOST_DEVICE)
    #         # if return_loss:
    #         #     new_loss_mask = loss_mask[0] * END_samples.view(-1).float()  # keep 0.0s
    #         #     loss_mask[0] = loss_mask[0] * (1.0-new_loss_mask)
    #         finished = torch.ge(finished + END_samples, 1)
    #         if torch.prod(finished) == 1:
    #             break
    #         x = x.transpose(0, 1)
    #
    #     end_found = torch.zeros(batch_size)
    #     for i, seq_step in enumerate(res):
    #         for j, token in enumerate(seq_step):
    #             if token == end_token:
    #                 end_found[j] = 1
    #             else:
    #                 if end_found[j] == 1:
    #                     seq_step[j] = end_token  # clean tokens after end_token
    #
    #     for i, found in enumerate(end_found):
    #         if found == 0:
    #             res[max_len-1][i] = end_token
    #
    #     # res = torch.cat(res, 1)
    #     if return_loss:
    #         # self.enable_gradients()
    #         # # self.load_rnn_training_state('../data/tmp.mod')
    #         mols = torch.cat(res, 1)
    #         starts = torch.zeros(mols.size(0), 1).long().to(HOST_DEVICE)
    #         starts[:] = start_token
    #         sample_batch = torch.cat((starts, mols), 1).to(HOST_DEVICE)
    #         loss = self.propagation(sample_batch, False)
    #
    #         # mols = torch.cat(res, 1)
    #         # loss = sort_by_len(
    #         #     input_data=mols,
    #         #     loss=loss,
    #         #     end_token=end_token,
    #         #     is_propagation=True
    #         # )
    #
    #         return res, loss
    #     else:
    #         return res, sys.float_info.max
