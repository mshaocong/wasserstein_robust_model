from wrm import attacker
from wrm.model.net import Net
import torch
import torchvision
import numpy as np
from wrm.utils import *
import matplotlib.pyplot as plt


def test():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=60000,
                                              shuffle=True, num_workers=0)
    trainset = next(iter(trainloader))


    # Sample a batch of data. Show their labels.
    batch_size = 5
    indices = np.random.randint(0, 60000, size=batch_size).astype(int)

    print("Sampled labels are ", trainset[1][indices].tolist())

    # Load Pre-trained classifier. Show the classification.
    net = Net()
    net.load_state_dict(torch.load("net98acc.weights"))
    net.cuda()

    data = trainset[0][indices].cuda()
    labels = trainset[1][indices].cuda()

    outputs = net(data)
    _, predicted = torch.max(outputs.data, 1)
    print("Predicted labels are ", predicted.tolist())

    # Apply Attacking
    # att = attacker.AttackerL2(reg_strength = 0.01)
    att = attacker.AttackerFGSM()
    # adv_images, loss_hist = att.attack(images=data, true_labels=labels, target_net=net, lr = 0.1, num_steps = 1000)
    adv_images, loss_hist = att.attack(images=data, target_label=8, target_net=net, lr = 0.1, num_steps=10)

    # Test results
    if loss_hist is not None:
        plt.plot(loss_hist, c="b")
        plt.show()

    outputs = net(adv_images)
    _, predicted = torch.max(outputs.data, 1)
    print("Post-attacking predictions are ", predicted.tolist())

    for i in range(5):
        f, axarr = plt.subplots(1, 2)
        images1 = data[i].cpu().detach().numpy().reshape((28, 28))
        axarr[0].imshow(images1)

        images2 = adv_images[i].cpu().detach().numpy().reshape((28, 28))
        axarr[1].imshow(images2)
        plt.show()


if __name__ == "__main__":
    test()