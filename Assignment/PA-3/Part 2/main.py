import argparse
import imageio
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets
from tqdm import tqdm


class Generator(nn.Module):
    r"""
    DCGAN generator

    Args:
        nz (int): Size of z latent vector (i.e. size of generator input, noise)
        ngf (int): Size of feature maps in generator
        nc (int): Number of channels in the training images
    """

    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc

        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        y = self.model(z)
        return y

    def record(self, index: int):
        fake_img = self(torch.randn(25, self.nz, 1, 1, device=DEVICE))
        fake_img = fake_img.view(fake_img.size(0), 1, 28, 28)
        save_image(fake_img[:25], 'images/{}.png'.format(index), nrow=5, normalize=True)


class Discriminator(nn.Module):
    r"""
        DCGAN Discriminator

        Args:
            nc (int): Number of channels in the training images
            ndf (int): Size of feature maps in discriminator
        """

    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf

        self.model = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        y = self.model(img).squeeze()
        return y


@torch.no_grad()
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# noinspection PyPep8Naming,PyUnboundLocalVariable
def train(dataloader, netG, netD, optimizerG, optimizerD):
    X, G_losses, D_losses = [], [], []
    criterion = nn.BCELoss()

    for epoch in range(OPT.n_epochs):
        for i, (real_img, _) in tqdm(enumerate(dataloader),
                                     total=len(dataloader),
                                     desc=f'Epoch: {epoch+1}/{OPT.n_epochs}',
                                     leave=False):
            batch_size = real_img.size(0)
            real_img = real_img.to(DEVICE)

            # Train Discriminator
            optimizerD.zero_grad()
            output = netD(real_img)
            label = torch.full((batch_size,), 1.0, device=DEVICE)  # real label
            lossD_real = criterion(output, label)
            lossD_real.backward()

            noise = torch.randn(batch_size, OPT.latent_dim, 1, 1, device=DEVICE)
            fake_img = netG(noise)
            label.fill_(0.0)  # fake label
            output = netD(fake_img.detach())
            lossD_fake = criterion(output, label)
            lossD_fake.backward()

            lossD = lossD_real + lossD_fake
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            label.fill_(1.0)
            output = netD(fake_img)
            lossG = criterion(output, label)
            lossG.backward()
            optimizerG.step()

        if epoch % OPT.eval_freq == 0:
            X.append(epoch)
            # Save Losses for plotting later
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())

        netG.record(epoch)
        
    if OPT.visual:
        plt.plot(X, G_losses, label='Generator Loss')
        plt.plot(X, D_losses, label='Discriminator Loss')
        plt.legend()
        plt.show()


# noinspection PyPep8Naming
def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=OPT.batch_size, shuffle=True)

    # Initialize models and optimizers
    netG = Generator(OPT.latent_dim, 128, 1).to(DEVICE)
    netD = Discriminator(1, 128).to(DEVICE)
    netG.apply(init_weights)
    netD.apply(init_weights)
    optimizer_G = optim.Adam(netG.parameters(), lr=OPT.lr)
    optimizer_D = optim.Adam(netD.parameters(), lr=OPT.lr)

    # Start training
    train(dataloader, netG, netD, optimizer_G, optimizer_D)

    img_list = [Image.open(f'./images/{i}.png') for i in range(OPT.n_epochs)]
    # noinspection PyTypeChecker
    imageio.mimsave('train.gif', img_list, fps=2)

    torch.save(netG.state_dict(), './model/mode.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='frequency to evaluate the model')
    parser.add_argument('--visual', action='store_true', default=False,
                        help='whether to display loss curve during traning process')
    OPT = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    main()
