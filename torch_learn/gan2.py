
import torch
import torch.nn as nn

import pandas as pd
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torch import autograd
from torch.autograd import Variable

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
plt.ion()

HOST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FashionMNIST(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        fashion_df = pd.read_csv('../../data/fashion-mnist_train.csv')
        self.labels = fashion_df.label.values
        self.images = fashion_df.iloc[:, 1:].values.astype('uint8').reshape(-1, 28, 28)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = Image.fromarray(self.images[idx])

        if self.transform:
            img = self.transform(img)
        return img, label


# dataset = FashionMNIST()
# print(dataset[0][0])

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
)
dataset = FashionMNIST(transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)


class DiscNet(nn.Module):
    def __init__(self):
        super(DiscNet, self).__init__()
        self.label_emb = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()


class GenNet(nn.Module):
    def __init__(self):
        super(GenNet, self).__init__()

        self.label_emb = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

        self.to(HOST_DEVICE)

    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)


gen = GenNet().to(HOST_DEVICE)
disc = DiscNet().to(HOST_DEVICE)

criterion = nn.BCELoss()
disc_opt = torch.optim.Adam(disc.parameters(), lr=1e-4)
gen_opt = torch.optim.Adam(gen.parameters(), lr=1e-4)


def gen_train_step(batch_size, disc, gen, gen_opt, criterion):
    gen_opt.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).to(HOST_DEVICE)
    fake_labels = Variable(torch.LongTensor(np.random.randint(10, size=batch_size))).to(HOST_DEVICE)
    fake_images = gen(z, fake_labels)
    validity = disc(fake_images, fake_labels)
    gen_loss = criterion(validity, Variable(torch.ones(batch_size)).to(HOST_DEVICE))
    gen_loss.backward()
    gen_opt.step()
    return gen_loss.data[0]


def disc_train_step(batch_size, disc, gen, disc_opt, criterion, real_images, labels):
    disc_opt.zero_grad()

    real_validity = disc(real_images, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(HOST_DEVICE))

    z = Variable(torch.randn(batch_size, 100)).to(HOST_DEVICE)
    fake_labels = Variable(
        torch.LongTensor(np.random.randint(10, size=batch_size))
    ).to(HOST_DEVICE)
    fake_images = gen(z, fake_labels)
    fake_validity = disc(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(HOST_DEVICE))

    disc_loss = real_loss + fake_loss
    disc_loss.backward()
    disc_opt.step()
    return disc_loss.data[0]


num_epochs = 30
n_critic = 5
disp_step = 300

for ep in range(num_epochs):
    print('Starting epoch {} ...'.format(ep))

    for i, (images, labels) in enumerate(data_loader):
        real_images = Variable(images).to(HOST_DEVICE)
        labels = Variable(labels).to(HOST_DEVICE)
        gen.train()
        batch_size = real_images.size(0)
        disc_loss = disc_train_step(
            len(real_images), disc, gen, disc_opt, criterion,
            real_images, labels
        )
        gen_loss = gen_train_step(
            batch_size, disc, gen, gen_opt, criterion
        )

        gen.eval()

        if i % disp_step == 0:
            print('step {} - Gen loss: {}, Disc loss: {}'.format(i, gen_loss, disc_loss))

        if ep > 0 and ep % 10 == 0 and i > 0 and i % 900 == 0:


            z = Variable(torch.randn(9, 100)).to(HOST_DEVICE)
            labels = Variable(torch.LongTensor(np.arange(9))).to(HOST_DEVICE)
            sample_images = gen(z, labels).unsqueeze(1).data.cpu()
            grid = make_grid(sample_images, nrow=3, normalize=True).permute(1, 2, 0).numpy()
            plt.imshow(grid)
            plt.show()
