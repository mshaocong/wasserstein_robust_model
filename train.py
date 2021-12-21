from model.net import Net
import torch
import torchvision
import numpy as np
from wrm import WassersteinRobustModel
from utils import *
import time
torch.manual_seed(2)
np.random.seed(3)

NUM_EPOCHS = 200




def main():
    net = Net()
    transform = torchvision.transforms.Compose([
        # transforms.Grayscale(3),
        torchvision.transforms.ToTensor()
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                             shuffle=False, num_workers=0)

    # Generate the indices
    batch_size = 1000

    # Optimizer parameters
    opt_kwargs = {"lr" : 0.001}

    model = WassersteinRobustModel(net, trainset, optimizer=torch.optim.SGD, **opt_kwargs)
    accuracy = []
    for i in range(NUM_EPOCHS):
        sample_indeces_list = get_indices(batch_size, len(trainset.targets))
        for index in sample_indeces_list:
            model.update(index, lr_ascent=0.001)

        num_correct = 0
        num_total = 0
        for data in testloader:
            _num_correct, _num_total = model.evaluate(data)
            num_correct,  num_total = num_correct + _num_correct, num_total + _num_total
        accuracy.append(num_correct/num_total)
        print(i, accuracy[-1])
    print("Done.")


if __name__ == "__main__":
    main()
