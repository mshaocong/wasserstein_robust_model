import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from wrm.cls_loss import *
from wrm.utils import *

# TODO:
#   1. Implementation of `AttackerCW`
#   2. Test the implementation of `AcceleratedAttacker`
#   3. Re-check the implementation of `FGSM`
#   4. Re-write the WRM using the `AcceleratedAttacker`
#   5. Repeat the experiment of WRM paper
#   6. Implement all mentioned classification losses in the attacking paper

# Check if GPU is correctly set up
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    raise Exception("GPU is not set up!")


class AttackerBase(object):
    def __init__(self, cls_loss, dist, reg_strength=1.0):
        self.cls_loss = cls_loss
        self.dist = dist
        self.reg_strength = reg_strength

    def _update(self, x, lr):
        with torch.no_grad():
            x.data = x.data - lr * x.grad.data
            x.data = torch.clamp(x.data, min=0.0, max=1.0)

    def _loss(self, true_labels, adv_images, images, target_net, target_label):
        output = target_net(adv_images)
        if true_labels is not None:
            loss = -self.cls_loss(output, true_labels) + self.reg_strength * self.dist(adv_images, images)
        elif target_label is not None:
            batch_size = adv_images.shape[0]
            if type(target_label) == int:
                labels = torch.from_numpy(np.array([target_label] * batch_size)).long().to(DEVICE)
            else:
                raise ValueError("Target label must be int.")
            loss = self.cls_loss(output, labels) + self.reg_strength * self.dist(adv_images, images)
        else:
            raise ValueError("Label should be specified.")
        return loss

    def attack(self, images, target_net, true_labels=None, target_label=None, lr=0.1, num_steps=100, adv_images=None):
        if true_labels is None and target_label is None:
            raise ValueError("Label should be specified.")

        if true_labels is not None:
            true_labels = true_labels.to(DEVICE)
        target_net = target_net.to(DEVICE)

        if adv_images is None:
            adv_images = images.clone().to(DEVICE)
        images.requires_grad = False
        adv_images.requires_grad = True

        for param in target_net.parameters():
            param.requires_grad = False
        # optim = torch.optim.SGD(adv_data, lr=self.lr)
        loss_record = []
        for step in range(num_steps):
            loss = self._loss(true_labels, adv_images, images, target_net, target_label)
            loss.backward()
            self._update(adv_images, lr)

            images.requires_grad = False
            adv_images.requires_grad = True
            for param in target_net.parameters():
                param.requires_grad = False
            # gen_data.data = l1proximal(gen_data.data, 0.0001)
            adv_images.grad.data.zero_()
            loss_record.append(loss.item())

        return adv_images, loss_record


class AttackerL2(AttackerBase):
    def __init__(self, reg_strength=1.0):
        def cls_loss(output, labels):
            output = F.softmax(output, dim=1)
            return F.cross_entropy(output, labels)

        def dist(images, adv_images):
            return torch.linalg.vector_norm(images - adv_images, ord=2)

        super().__init__(cls_loss=cls_loss, dist=dist, reg_strength=reg_strength)

    def _update(self, x, lr):
        with torch.no_grad():
            x.data = x.data - lr * x.grad.data
            x.data = torch.clamp(x.data, min=0.0, max=1.0)


class AttackerFGSM(AttackerBase):
    def __init__(self, epsilon=0.1):
        def cls_loss(output, labels):
            output = F.softmax(output, dim=1)
            return F.cross_entropy(output, labels)

        def dist(images, adv_images):
            return torch.linalg.vector_norm(images - adv_images, ord=2)

        super().__init__(cls_loss=cls_loss, dist=dist, reg_strength=0.0)
        self.epsilon = epsilon

    def _update(self, x, lr):
        with torch.no_grad():
            x.data = x.data - torch.clamp(lr * torch.sign(x.grad.data), min=-self.epsilon, max=self.epsilon)
            x.data = torch.clamp(x.data, min=0.0, max=1.0)


class AttackerCW(AttackerBase):
    """
    Implementation of C-W attack. Originally presented in the paper Carlini, Nicholas, and David Wagner. "Towards
    evaluating the robustness of neural networks." 2017 ieee symposium on security and privacy (sp). IEEE, 2017.
    """

    def __init__(self, reg_strength, type="L0"):
        def cls_loss(output, labels):
            output = F.softmax(output, dim=1)
            return F.cross_entropy(output, labels)

        def dist(images, adv_images):
            return torch.linalg.vector_norm(images - adv_images, ord=2)

        super().__init__(cls_loss=cls_loss, dist=dist, reg_strength=reg_strength)


class AcceleratedAttacker(AttackerBase):
    """
    This implementation aims to support the momentum acceleration. Chen, Ziyi, Shaocong Ma, and Yi Zhou. "Accelerated
    Proximal Alternating Gradient-Descent-Ascent for Nonconvex Minimax Machine Learning." arXiv preprint
    arXiv:2112.11663 (2021).
    """

    def __init__(self, reg_strength, momentum, regularizer="L1"):
        def cls_loss(output, labels):
            output = F.log_softmax(output, dim=1)
            return F.nll_loss(output, labels)

        self.regularizer = regularizer
        self.momentum = momentum
        self.cache = {}

        super().__init__(cls_loss=cls_loss, dist=None, reg_strength=reg_strength)

    def _proximal(self, x):
        if self.regularizer == "L1":
            return l1proximal(x, self.reg_strength)
        elif self.regularizer == "L2":
            return l2proximal(x, self.reg_strength)
        else:
            raise ValueError("This type of regularizer has not been supported.")

    def _update(self, x, lr):
        if "past_data" in self.cache.keys():
            with torch.no_grad():
                x.data = self._proximal(x.data - lr * x.grad.data)
                x.data = torch.clamp(x.data, min=0.0, max=1.0)
            self.cache["past_data"] = x.data
        else:
            with torch.no_grad():
                x.data = self._proximal(x.data - lr * x.grad.data - self.momentum * (x.data - self.cache["past_data"]))
                x.data = torch.clamp(x.data, min=0.0, max=1.0)
            self.cache["past_data"] = x.data

    def _loss(self, true_labels, adv_images, images, target_net, target_label):
        output = target_net(adv_images)
        if true_labels is not None:
            loss = -self.cls_loss(output, true_labels)
        elif target_label is not None:
            batch_size = adv_images.shape[0]
            if type(target_label) == int:
                labels = torch.from_numpy(np.array([target_label] * batch_size)).long().to(DEVICE)
            else:
                raise ValueError("Target label must be int.")
            loss = self.cls_loss(output, labels)
        else:
            raise ValueError("Label should be specified.")
        return loss
