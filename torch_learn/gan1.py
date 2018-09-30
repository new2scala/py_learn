
import numpy as np
import torch


def get_distribution_sampler(mu, sigma):
    return lambda n: torch.tensor(
        np.random.normal(mu, sigma, (1, n))
    )


def test_sampler():
    sampler = get_distribution_sampler(mu=0.0, sigma=1.0)
    # r = 10
    print (sampler(10))  #[ sampler(5) for i in range(-r, r, 0.5) ]


def get_gen_input_sampler():
    return lambda m, n: torch.rand(m, n)


import torch.nn as nn
import torch.nn.functional as F


class GenNet(nn.Module):
    def __init__(self, in_sz, hd_sz, out_sz):
        super(GenNet, self).__init__()

        self._m1 = nn.Linear(in_sz, hd_sz)
        self._m2 = nn.Linear(hd_sz, hd_sz)
        self._m3 = nn.Linear(hd_sz, out_sz)

    def forward(self, in_data):
        x = F.elu(self._m1(in_data))
        x = F.sigmoid(self._m2(x))
        return self._m3(x)


class DiscNet(nn.Module):
    def __init__(self, in_sz, hd_sz, out_sz):
        super(DiscNet, self).__init__()
        self._m1 = nn.Linear(in_sz, hd_sz)
        self._m2 = nn.Linear(hd_sz, hd_sz)
        self._m3 = nn.Linear(hd_sz, out_sz)

    def forward(self, in_data):
        x = F.elu(self._m1(in_data))
        x = F.elu(self._m2(x))
        return F.sigmoid(self._m3(x))


data_mean, data_stddev = 4, 1.25

g_in_sz, g_hd_sz, g_out_sz = 1, 50, 1
d_in_sz, d_hd_sz, d_out_sz = 100, 50, 1  # in_sz: batch size

bat_sz = d_in_sz
d_lr = 2e-4
g_lr = 2e-4

opt_betas = (0.9, 0.999)
epochs = 30000
print_itv = 200

d_steps = 1
g_steps = 1


from torch.autograd import Variable


def decorate_with_diffs(data, exponent):
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast).double(), exponent)
    return torch.cat([data, diffs], 1).float()

name, preprocess, d_input_func = ("Raw data", lambda data: data, lambda x: x)
#(name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)

print("Using data [%s]" % name)


def stats(d):
    return [np.mean(d), np.std(d)]


d_sampler = get_distribution_sampler(data_mean, data_stddev)
g_sampler = get_gen_input_sampler()


G = GenNet(
    in_sz=g_in_sz,
    hd_sz=g_hd_sz,
    out_sz=g_out_sz
)

D = DiscNet(
    in_sz=d_in_sz,
    hd_sz=d_hd_sz,
    out_sz=d_out_sz
)


import torch.optim as optim
criterion = nn.BCELoss()
d_opt = optim.Adam(D.parameters(), lr=d_lr, betas=opt_betas)
g_opt = optim.Adam(G.parameters(), lr=g_lr, betas=opt_betas)

for ep in range(epochs):
    for d_idx in range(d_steps):
        # 1. train D on real + fake
        D.zero_grad()
        # 1A. train D on real
        d_real_data = Variable(d_sampler(d_in_sz))
        d_real_data = preprocess(d_real_data).float()
        d_real_decision = D(d_real_data)
        d_real_err = criterion(d_real_decision, torch.ones(1))
        d_real_err.backward()  # compute/store gradients, but don't change params
        # 1B. train D on fake
        d_gen_input = Variable(g_sampler(bat_sz, g_in_sz))
        d_fake_data = G(d_gen_input).detach()
        d_fake_decision = D(preprocess(d_fake_data).t())
        d_fake_err = criterion(d_fake_decision, torch.zeros(1))
        d_fake_err.backward()
        d_opt.step()

    for g_idx in range(g_steps):
        # 2. train G on D's response (but not train D)
        G.zero_grad()
        gen_input = Variable(g_sampler(bat_sz, g_in_sz))
        g_fake_data = G(gen_input)
        dg_fake_decision = D(preprocess(g_fake_data.t()))
        g_err = criterion(dg_fake_decision, torch.ones(1))
        g_err.backward()
        g_opt.step()

    if ep % print_itv == 0:
        print(
            '%s: D: %s/%s G: %s (real: %s, fake: %s)' % (
                ep, d_real_err.data[0], d_fake_err.data[0], g_err.data[0],
                stats(d_real_data.numpy()), stats(d_fake_data.numpy())
            )
        )