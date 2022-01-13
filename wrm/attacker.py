import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from wrm.cls_loss import *

# Check if GPU is corrected set up
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    raise Exception("GPU is not set up!")

class AttackerBase(object):
    def __init__(self, cls_loss, dist, reg_strength = 1.0):
        self.cls_loss = cls_loss
        self.dist = dist
        self.reg_strength = reg_strength

    def _update(self, x, lr):
        with torch.no_grad():
            x.data = x.data - lr * x.grad.data
            x.data = torch.clamp(x.data, min=0.0, max=1.0)

    def attack(self, images, target_net, true_labels = None, target_label = None, lr = 0.1, num_steps = 100, adv_images = None):
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
            loss.backward()

            self._update(adv_images, lr)
            # with torch.no_grad():
            #    adv_images.data = adv_images.data - lr * adv_images.grad.data
            #    adv_images.data = torch.clamp(adv_images.data, min=0.0, max=1.0)

            images.requires_grad = False
            adv_images.requires_grad = True
            for param in target_net.parameters():
                param.requires_grad = False
            # gen_data.data = l1proximal(gen_data.data, 0.0001)
            adv_images.grad.data.zero_()
            loss_record.append(loss.item())

        return adv_images, loss_record




class AttackerL2(AttackerBase):
    def __init__(self, reg_strength = 1.0):
        def cls_loss(output, labels):
            return F.nll_loss(output, labels)
            # return F.cross_entropy(output, labels)

        def dist(images, adv_images):
            return torch.linalg.vector_norm(images - adv_images, ord=2)

        super().__init__(cls_loss=cls_loss, dist=dist, reg_strength=reg_strength)



class AttackerFGSM(AttackerBase):
    def __init__(self):
        def cls_loss(output, labels):
            return F.nll_loss(output, labels)

        def dist(images, adv_images):
            return torch.linalg.vector_norm(images - adv_images, ord=2)

        super().__init__(cls_loss=cls_loss, dist=dist, reg_strength=0.0)


    def _update(self, x, lr):
        with torch.no_grad():
            x.data = x.data - lr * torch.sign(x.grad.data)
            x.data = torch.clamp(x.data, min=0.0, max=1.0)

class AttackerCW(AttackerBase):
    '''
    Implementation of C-W attack. Originally presented in the paper
    Carlini, Nicholas, and David Wagner. "Towards evaluating the robustness of neural networks." 2017 ieee symposium on security and privacy (sp). IEEE, 2017.
    '''
    def __init__(self, lr_ascent, type="L0"):
        super().__init__(lr_ascent)

    def attack(self, images, target_net):
        raise NotImplementedError

