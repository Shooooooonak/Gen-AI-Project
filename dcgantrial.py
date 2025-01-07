from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import pandas as pd
from PIL import Image

dataroot = os.path.abspath("/Users/shubham/Documents/College/GenAI/test")

workers = 4
batch_size = 1
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 8
ngpu = 1

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers = workers)
print(len(dataset))
test_indices = list(range(0,39800))
test_dataset = torch.utils.data.Subset(dataset, test_indices)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True, num_workers = workers)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
attributePath = "/Users/shubham/Documents/College/GenAI/list_attr_celeba.csv"
df = pd.read_csv(attributePath)

print(df.columns)

class Encoder(nn.Module):
    def __init__(self, ngpu):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3, ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf * 8, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf * 16, 100, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

generator = Generator(ngpu).to(device)
generator.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))
generator.eval()

encoder = Encoder(ngpu).to(device)
encoder.load_state_dict(torch.load('best_encoder.pth', map_location=torch.device('cpu')))
encoder.eval()



print(test_loader.dataset[1][0].shape)


encoderval1 = encoder(test_loader.dataset[188355-162771][0].unsqueeze(0).to(device))
encoderval2 = encoder(test_loader.dataset[162791-162771][0].unsqueeze(0).to(device))
encoderval3 = encoder(test_loader.dataset[1883-162771][0].unsqueeze(0).to(device))
generated1 = generator(encoderval1)
generated2 = generator(encoderval2)
generated3 = generator(encoderval3)
generated = generator(encoderval1 - encoderval2 + encoderval3)

to_pil = transforms.ToPILImage()
img = to_pil(generated1[0])

# Resize the image to make it bigger
width, height = img.size
new_size = (width * 4, height * 4) 
img_resized = img.resize(new_size, Image.BICUBIC)

# Show the resized image
img_resized.show()

img = to_pil(generated2[0])

# Resize the image to make it bigger
width, height = img.size
new_size = (width * 4, height * 4) 
img_resized = img.resize(new_size, Image.BICUBIC)

# Show the resized image
img_resized.show()

img = to_pil(generated3[0])

# Resize the image to make it bigger
width, height = img.size
new_size = (width * 4, height * 4) 
img_resized = img.resize(new_size, Image.BICUBIC)

# Show the resized image
img_resized.show()

img = to_pil(generated[0])

# Resize the image to make it bigger
width, height = img.size
new_size = (width * 4, height * 4) 
img_resized = img.resize(new_size, Image.BICUBIC)

# Show the resized image
img_resized.show()

