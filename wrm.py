from utils import *
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


class WassersteinRobustModel:
    def __init__(self, net, dataset, optimizer, gamma=1.3, **kwargs):
        self.net = net
        self.net.to(DEVICE)

        self.dataset = dataset

        gen_dataset =  np.random.uniform(size = dataset.data.shape)
        gen_dataset = np.expand_dims(gen_dataset, axis=1)
        self.gen_dataset = gen_dataset

        self.gamma = gamma
        self.optimizer_descent = optimizer(self.net.parameters(), **kwargs)

    def update(self, indices, num_ascent_steps=1, lr_ascent=0.001):
        net = self.net
        net.to(DEVICE)

        optimizer_descent = self.optimizer_descent
        optimizer_descent.zero_grad()

        data = self.dataset.data[indices]
        data = data.unsqueeze(1)
        data = data.float().to(DEVICE)
        target = self.dataset.targets[indices].to(DEVICE)

        gen_data = torch.from_numpy(self.gen_dataset[indices]).float().to(DEVICE)
        gen_data.requires_grad = True

        # Stage 1 - One step gradient descent. Freeze the adversarial data `y`; do descent on net params.
        output = net(gen_data)
        loss_descent = F.nll_loss(output, target)
        loss_descent.backward()
        optimizer_descent.step()

        # Stage 2 - Multiple steps gradient ascent. Freeze net params; do ascent on the adversarial data `y`
        for param in net.parameters():
            param.requires_grad = False
        gen_data = torch.from_numpy(self.gen_dataset[indices]).float().to(DEVICE)
        gen_data.requires_grad = True
        for step in range(num_ascent_steps):
            output = net(gen_data)
            loss_ascent = - F.nll_loss(output, target) + self.gamma * torch.sum((data - gen_data) ** 2)
            loss_ascent.backward()
            gen_data.data = gen_data.data + lr_ascent * gen_data.grad.data
            gen_data.data = l1proximal(gen_data.data, 0.0001)
            gen_data.grad.data.zero_()
        self.gen_dataset[indices] = np.copy(gen_data.data.cpu())


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