from model.net import Net
import torch
import torchvision
import numpy as np
from wrm import NonWRM, WRM, MomentumWRM
from utils import *
import pickle
import argparse



def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    testset = next(iter(testloader))

    model = WRM(Net(), trainset[0], trainset[1], attacker_type=args.attacker,
                lr_descent=args.lr_descent, lr_ascent=args.lr_ascent)
    accelerated_model = MomentumWRM(Net(), trainset[0], trainset[1], attacker_type=args.attacker,
                                    momentum_descent=args.momentum_descent, momentum_ascent=args.momentum_ascent,
                                    lr_descent=args.lr_descent, lr_ascent=args.lr_ascent)

    model_accuracy = {"vanilla": [], "momentum": [], "non-wrm":[]}
    for i in range(args.epochs):
        sample_indeces_list = get_indices(args.batch_size, len(trainset[0]))
        for index in sample_indeces_list:
            model.update(index, num_ascent_steps=args.num_ascent_steps)
            accelerated_model.update(index, num_ascent_steps=args.num_ascent_steps)

        num_correct, num_total = model.evaluate(testset[0], testset[1])
        model_accuracy["vanilla"].append(num_correct / num_total)

        num_correct, num_total = accelerated_model.evaluate(testset[0], testset[1])
        model_accuracy["momentum"].append(num_correct / num_total)

    with open('accuracy_' + args.id + '.pickle', 'wb') as handle:
        pickle.dump(model_accuracy, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wasserstein Robust Model')
    parser.add_argument('--id', type=str, default='MNIST', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=114514, help='Random seed')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr_ascent', type=float, default=10e-3,
                        help='Learning rate for maximization')
    parser.add_argument('--lr_descent', type=float, default=10e-3,
                        help='Learning rate for minimization')
    parser.add_argument('--momentum_ascent', type=float, default=0.75,
                        help='Momentum for maximization')
    parser.add_argument('--momentum_descent', type=float, default=0.25,
                        help='Momentum for minimization')
    parser.add_argument('--gamma', type=float, default=1.3,
                        help='Regularization strength')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
    parser.add_argument('--num_ascent_steps', type=int, default=1, help='Number of gradient ascent updates')
    parser.add_argument('--attacker', type=str, default='L2', help='Type of adversarial attacker')

    parse_args = parser.parse_args()

    main(args=parse_args)
