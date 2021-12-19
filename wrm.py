import torch
import torchvision
from model.net import Net
import torch.nn.functional as F
from torch import optim
import numpy as np
import time


# Check if GPU is corrected set up
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # dtype = torch.cuda.FloatTensor
else:
    raise Exception("GPU is not set up!")
IMG_ROWS, IMG_COLS = 28, 28


class WassersteinRobustModel:
    def __init__(self, net, dataset, gamma=1.3):
        self.net = net
        self.dataset = dataset

        gen_dataset =  np.random.uniform(size = dataset.data.shape)
        gen_dataset = np.expand_dims(gen_dataset, axis=1)
        self.gen_dataset = gen_dataset

        self.gamma = gamma

    def update(self, indices, lr_descent=0.001, lr_ascent=0.001):
        net = self.net
        net.to(DEVICE)

        optimizer_descent = optim.SGD(net.parameters(), lr=lr_descent)
        optimizer_descent.zero_grad()

        data = self.dataset.data[indices]
        data = data.unsqueeze(1)
        data = data.float().to(DEVICE)
        target = self.dataset.targets[indices].to(DEVICE)

        gen_data = torch.from_numpy(self.gen_dataset[indices]).float().to(DEVICE)
        gen_data.requires_grad = True
        optimizer_ascent = optim.SGD([gen_data], lr=lr_ascent)
        optimizer_ascent.zero_grad()

        output = net(gen_data)
        loss_descent = F.nll_loss(output, target)
        loss_descent.backward()
        optimizer_descent.step()

        output = net(gen_data)
        loss_ascent = - F.nll_loss(output, target) + self.gamma * torch.sum((data - gen_data) ** 2)
        loss_ascent.backward()
        optimizer_ascent.step()


net = Net()
# net.load_state_dict(torch.load("tmp.pth"))
transform = torchvision.transforms.Compose([
    #transforms.Grayscale(3),
    torchvision.transforms.ToTensor()
])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
batch_size = 100
seq = np.arange(len(trainset.targets))
np.random.shuffle(seq)
sample_indeces_list = np.split(seq, len(trainset.targets) // batch_size)

model = WassersteinRobustModel(net, trainset)
for i in range(10):
    for index in sample_indeces_list:
        model.update(index)

print("Done.")