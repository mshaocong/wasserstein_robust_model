from utils import *
import torch
import torch.nn.functional as F
import numpy as np
import time



# Check if GPU is corrected set up
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    raise Exception("GPU is not set up!")


class WRM:
    def __init__(self, net, data, labels, attacker_type="L2", lr_descent=10e-3, lr_ascent=10e-3, gamma=1.3):
        self.net = net
        self.net.to(DEVICE)

        self.data = data
        self.targets = labels
        gen_dataset =  np.random.standard_normal(size = data.shape) + np.copy(data)
        self.gen_dataset = gen_dataset

        self.gamma = gamma
        self.optim_descent = torch.optim.SGD(self.net.parameters(), lr=lr_descent)

        self.lr_descent = lr_descent
        self.lr_ascent = lr_ascent
        self.attacker_type = attacker_type


    def update(self, indices, num_ascent_steps=1):
        lr_ascent = self.lr_ascent

        net = self.net
        net.to(DEVICE)

        optim_descent = self.optim_descent
        optim_descent.zero_grad()

        data = self.data[indices]
        # data = data.unsqueeze(1)
        # data = data/255.0
        data = data.float().to(DEVICE)
        target = self.targets[indices].to(DEVICE)

        # Stage 1 - One step gradient descent. Freeze the adversarial data `y`; do descent on net params.
        gen_data = torch.from_numpy(self.gen_dataset[indices]).float().to(DEVICE)
        gen_data.requires_grad = False
        for param in net.parameters():
            param.requires_grad = True
        output = net(gen_data)
        loss_descent = F.nll_loss(output, target)
        loss_descent.backward()
        optim_descent.step()

        # Stage 2 - Multiple steps gradient ascent. Freeze net params; do ascent on the adversarial data `y`
        gen_data = torch.from_numpy(self.gen_dataset[indices]).float().to(DEVICE)
        gen_data.requires_grad = True
        for param in net.parameters():
            param.requires_grad = False
        for step in range(num_ascent_steps):
            # when the attacker is not specified, WRM degenerates to normal classification task
            if self.attacker_type is None:
                break

            output = net(gen_data)
            if self.attacker_type == "L2":
                # loss_ascent = F.nll_loss(output, target) - self.gamma * torch.sum((data - gen_data) ** 2)
                loss_ascent = F.nll_loss(output, target) - self.gamma * torch.linalg.vector_norm(data - gen_data, ord=2)
            elif self.attacker_type == "L1":
                loss_ascent = F.nll_loss(output, target) - self.gamma * torch.linalg.vector_norm(data - gen_data, ord=1)
            else:
                raise "Attacker type "+str(self.attacker_type)+" is not supported."
            loss_ascent.backward()
            gen_data.data = gen_data.data + lr_ascent * gen_data.grad.data
            # gen_data.data = l1proximal(gen_data.data, 0.0001)
            gen_data.grad.data.zero_()
        self.gen_dataset[indices] = np.copy(gen_data.data.cpu())

    def evaluate(self, data, labels):
        data = data.float().to(DEVICE)
        labels = labels.float().to(DEVICE)
        outputs = self.net(data)
        _, predicted = torch.max(outputs.data, 1)
        return (predicted == labels).sum().item(), labels.size(0)


class MomentumWRM(WRM):
    def __init__(self, net, data, labels, attacker_type="L2", lr_descent=10e-3, lr_ascent=10e-3, momentum_descent=0.1,
                 momentum_ascent=0.1, gamma=1.3):
        super().__init__(net, data, labels, attacker_type, lr_descent, lr_ascent, gamma)
        self.optim_descent = torch.optim.SGD(self.net.parameters(), lr=lr_descent, momentum=momentum_descent)
        self.momentum_descent = momentum_descent
        self.momentum_ascent = momentum_ascent

        self.cache = np.copy(self.gen_dataset)

    def update(self, indices, num_ascent_steps=1):
        raise NotImplementedError