from model.net import Net
import torch
import torchvision
import numpy as np
from wrm import WRM, MomentumWRM
from utils import *
import time
import argparse

torch.manual_seed(2)
np.random.seed(3)

NUM_EPOCHS = 100

# TODO: (4) Conduct a hyper-parameter exploration. Draw some plots. Write a technical report.


def main():
    net = Net()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=60000,
                                              shuffle=True, num_workers=0)
    trainset = next(iter(trainloader))

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                             shuffle=False, num_workers=0)


    # Optimizer parameters
    batch_size = 1000
    lr_ascent = 10e-3
    lr_descent = 10e-3
    num_ascent_steps = 1

    # model = WRM(net, trainset[0], trainset[1], attacker_type="L2",
    #                               lr_descent=lr_descent, lr_ascent=lr_ascent)
    model = MomentumWRM(net, trainset[0], trainset[1], attacker_type="L2",
                lr_descent = lr_descent, lr_ascent = lr_ascent)

    accuracy = []
    for i in range(NUM_EPOCHS):
        sample_indeces_list = get_indices(batch_size, len(trainset[0]))
        for index in sample_indeces_list:
            model.update(index, num_ascent_steps=num_ascent_steps)

        num_correct = 0
        num_total = 0
        for data in testloader:
            _num_correct, _num_total = model.evaluate(data[0], data[1])
            num_correct,  num_total = num_correct + _num_correct, num_total + _num_total
        accuracy.append(num_correct/num_total)
        print(i, accuracy[-1])
    print("Done.")


if __name__ == "__main__":
    main()
