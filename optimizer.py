import torch
import torchvision
from model.net import Net
import torch.nn.functional as F
import numpy as np
import time
# Reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Global constants
IMG_ROWS, IMG_COLS = 28, 28

# Check if GPU is corrected set up
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dtype = torch.cuda.FloatTensor
else:
    raise Exception("GPU is not set up!")


def l1proximal(x, _lambda):
    return torch.sign(x) * torch.maximum( torch.abs(x) - _lambda, torch.zeros_like(x) )

def l2proximal(x, _lambda):
    return torch.maximum(1.0 - _lambda/torch.norm(x) ,torch.zeros_like(torch.norm(x))) * x

def gda(batch_size = 2, lr1 = 0.01, lr2 = 0.01, gamma = 1.3, num_epochs = 100, momentum1 = 0.01, momentum2 = 0.01):
    # Stage 0. Initialization
    transform = torchvision.transforms.Compose([
        #transforms.Grayscale(3),
        torchvision.transforms.ToTensor()
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    train_loss = []
    accuracy = []
    y_total = [ np.random.randn(batch_size, 1, IMG_ROWS, IMG_COLS) for _ in range(60000//batch_size+1)] # used to store y_t
    y_cache = [ np.copy(_) for _ in y_total] # used to store y_t-1

    net = Net()
    net.to(device)
    net.load_state_dict(torch.load("tmp-ini.pth"))
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                             shuffle=False, num_workers=0)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.float().to(device)
            labels.float().to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy.append(correct/total)
    print(accuracy)

    seq = np.arange(len(trainset.targets))
    np.random.shuffle(seq)
    sample_indeces_list = np.split(seq, len(trainset.targets) // batch_size)

    torch.save(net.state_dict(), "tmp.pth")
    torch.save(net.state_dict(), "tmp-cache.pth")
    for i in range(num_epochs):
        print("The %d-th epoch starts." % i)
        start_time = time.time()
        est_loss = 0.0

        for j in range(len(sample_indeces_list)):
            data = trainset.data[sample_indeces_list[j]]
            data = data.unsqueeze(1)
            data = data.float().to(device)
            # data = data/255.0
            target = trainset.targets[sample_indeces_list[j]].to(device)

            # Stage 1 - One step gradient descent. Freeze the parameters of neural network; do descent on y
            net = Net()
            net.load_state_dict(torch.load("tmp.pth"))
            net.to(device)
            y = torch.from_numpy(y_total[j]).float().to(device)
            y.requires_grad = False
            output = net(y)
            loss = F.nll_loss(output, target)
            for param in net.parameters():
                param.requires_grad = True
            net.zero_grad()
            loss.backward()

            tmp_net = Net()
            tmp_net.load_state_dict(torch.load("tmp-cache.pth"))
            with torch.no_grad():
                for p, p_past in zip(net.parameters(), tmp_net.parameters()):
                    new_val = p - lr1*p.grad + momentum1 * (p - p_past)
                    p_past.copy_(p)
                    p.copy_(l2proximal(new_val, 0.0001))
            torch.save(tmp_net.state_dict(), "tmp-cache.pth")
            torch.save(net.state_dict(), "tmp.pth")

            # train_loss.append(np.copy(loss.item()))

            # Stage 2 - One step gradient ascent. Freeze y; do ascent on parameter x

            net = Net()
            net.load_state_dict(torch.load("tmp.pth"))
            net.to(device)
            y = torch.from_numpy(y_total[j]).float().to(device)
            y.requires_grad = True
            y_past = torch.from_numpy(y_cache[j]).float().to(device)
            y_past.requires_grad = False
            y_cache[j] = np.copy(y.data.cpu())
            output = net(y)
            for param in net.parameters():
                param.requires_grad = False
            loss1 = F.nll_loss(output, target)
            loss2 = torch.sum((data - y) ** 2) #/batch_size  #/IMG_ROWS/IMG_COLS
            loss = loss1 - gamma * loss2
            loss.backward()
            y.data = y.data + lr2 * y.grad.data + momentum2 * (y.data - y_past.data)
            y.data = l1proximal(y.data, 0.0001)
            y.grad.data.zero_()
            y_total[j] = np.copy(y.data.cpu())

            # Evaluate the estimated maxmium
            est_net = Net()
            est_net.load_state_dict(torch.load("tmp.pth"))
            est_net.to(device)
            est_y = torch.from_numpy(y_total[j]).float().to(device)
            est_y.requires_grad = True
            for param in est_net.parameters():
                param.requires_grad = False
            for _ in range(100):
                est_output = net(est_y)
                loss1 = F.nll_loss(est_output, target)
                loss2 = torch.sum((data - est_y) ** 2)
                loss = loss1 - gamma * loss2
                loss.backward()
                est_y.data = est_y.data + 0.1 * est_y.grad.data
                est_y.grad.data.zero_()
            est_loss += loss.item()
        train_loss.append(est_loss )

        # Evaluate Test Accuracy
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                                 shuffle=False, num_workers=0)
        test_net = Net().to(device)
        test_net.load_state_dict(torch.load("tmp.pth"))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.float().to(device)
                labels.float().to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy.append(correct / total)

        print("Epoch ends. Time spent:", time.time() - start_time)
    return train_loss, accuracy

def gda2(batch_size = 2, lr1 = 0.01, lr2 = 0.01, lr3 = 0.01, gamma = 1.3, num_epochs = 100):
    # Stage 0. Initialization
    transform = torchvision.transforms.Compose([
        #transforms.Grayscale(3),
        torchvision.transforms.ToTensor()
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    accuracy = []
    train_loss = []
    y_total = [ np.random.randn(batch_size, 1, IMG_ROWS, IMG_COLS) for _ in range(60000//batch_size+1)]
    y_tilde_total = [ np.copy(_) for _ in y_total]

    net = Net()
    net.to(device)
    net.load_state_dict(torch.load("tmp-ini.pth"))
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                             shuffle=False, num_workers=0)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.float().to(device)
            labels.float().to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy.append(correct/total)
    print(accuracy)

    seq = np.arange(len(trainset.targets))
    np.random.shuffle(seq)
    sample_indeces_list = np.split(seq, len(trainset.targets) // batch_size)
    torch.save(net.state_dict(), "tmp.pth")
    torch.save(net.state_dict(), "tmp-cache.pth")
    for i in range(num_epochs):
        print("The %d-th epoch starts." % i)
        est_loss = 0.0
        start_time = time.time()

        for j in range(len(sample_indeces_list)):
            data = trainset.data[sample_indeces_list[j]]
            data = data.unsqueeze(1)
            data = data.float().to(device)
            target = trainset.targets[sample_indeces_list[j]].to(device)

            # State 0 - One step gradient ascent on x.
            net = Net()
            net.load_state_dict(torch.load("tmp.pth"))
            net.to(device)
            y = torch.from_numpy(y_total[j]).float().to(device)
            y.requires_grad = True
            y_tilde = torch.from_numpy(y_tilde_total[j]).float().to(device)
            y_tilde.requires_grad = False
            output = net(y)
            for param in net.parameters():
                param.requires_grad = False
            loss1 = F.nll_loss(output, target)
            loss2 = torch.sum((data - y) ** 2)
            loss = loss1 - gamma * loss2
            loss.backward()
            y.data = y_tilde.data + lr1 * y.grad.data
            y.data = l1proximal(y.data, 0.0001)
            y.grad.data.zero_()
            y_total[j] = np.copy(y.data.cpu())

            # Stage 1 - One step gradient descent. Freeze the parameters of neural network; do descent on y
            net = Net()
            net.load_state_dict(torch.load("tmp.pth"))
            net.to(device)
            y = torch.from_numpy(y_total[j]).float().to(device)
            y.requires_grad = False
            output = net(y)
            loss = F.nll_loss(output, target)
            for param in net.parameters():
                param.requires_grad = True
            net.zero_grad()
            loss.backward()

            tmp_net = Net()
            tmp_net.load_state_dict(torch.load("tmp-cache.pth"))
            with torch.no_grad():
                for p, p_past in zip(net.parameters(), tmp_net.parameters()):
                    new_val = p - lr2*p.grad
                    p_past.copy_(p)
                    p.copy_(l2proximal(new_val, 0.0001))
            torch.save(tmp_net.state_dict(), "tmp-cache.pth")
            torch.save(net.state_dict(), "tmp.pth")


            # Stage 2 - One step gradient ascent. Freeze y; do ascent on parameter x
            net = Net()
            net.load_state_dict(torch.load("tmp.pth"))
            net.to(device)
            y1 = torch.from_numpy(y_total[j]).float().to(device)
            y1.requires_grad = True
            y2 = torch.from_numpy(y_total[j]).float().to(device)
            y2.requires_grad = True
            y_tilde = torch.from_numpy(y_tilde_total[j]).float().to(device)
            y_tilde.requires_grad = False
            output = net(y1)

            tmp_net = Net()
            tmp_net.load_state_dict(torch.load("tmp-cache.pth"))
            tmp_output = tmp_net(y2)
            for param in net.parameters():
                param.requires_grad = False
            loss1 = F.nll_loss(output, target)
            loss2 = torch.sum((data - y1) ** 2)
            loss = loss1 - gamma * loss2
            loss.backward()

            tmp_loss1 = F.nll_loss(tmp_output, target)
            tmp_loss2 = torch.sum((data - y2) ** 2)
            tmp_loss = tmp_loss1 - gamma * tmp_loss2
            tmp_loss.backward()

            y_tilde.data = y_tilde.data + lr3 * ( y1.grad.data - y2.grad.data + (y1.data - y_tilde.data)/lr1 )
            y1.grad.data.zero_()
            y2.grad.data.zero_()
            y_tilde_total[j] = np.copy(y_tilde.data.cpu())

            # Evaluate the estimated maxmium
            est_net = Net()
            est_net.load_state_dict(torch.load("tmp.pth"))
            est_net.to(device)
            est_y = torch.from_numpy(y_total[j]).float().to(device)
            est_y.requires_grad = True
            for param in est_net.parameters():
                param.requires_grad = False
            for _ in range(100):
                est_output = net(est_y)
                loss1 = F.nll_loss(est_output, target)
                loss2 = torch.sum((data - est_y) ** 2)
                loss = loss1 - gamma * loss2
                loss.backward()
                est_y.data = est_y.data + 0.1 * est_y.grad.data
                est_y.grad.data.zero_()
            est_loss += loss.item()
        train_loss.append(est_loss )
        # Evaluate Test Accuracy
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                                 shuffle=False, num_workers=0)
        test_net = Net().to(device)
        test_net.load_state_dict(torch.load("tmp.pth"))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.float().to(device)
                labels.float().to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy.append(correct / total)

        # Evaluate the estimated maxmium
        print("Epoch ends. Time spent:", time.time() - start_time)
    return train_loss, accuracy

def gda3(batch_size = 2, lr1 = 0.01, lr2 = 0.01, gamma = 1.3, num_epochs = 100, momentum1 = 0.01, momentum2 = 0.01):
    # Stage 0. Initialization
    transform = torchvision.transforms.Compose([
        #transforms.Grayscale(3),
        torchvision.transforms.ToTensor()
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    train_loss = []
    accuracy = []
    y_total = [ np.random.randn(batch_size, 1, IMG_ROWS, IMG_COLS) for _ in range(60000//batch_size+1)] # used to store y_t
    y_cache = [ np.copy(_) for _ in y_total] # used to store y_t-1

    net = Net()
    net.to(device)
    net.load_state_dict(torch.load("tmp-ini.pth"))
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                             shuffle=False, num_workers=0)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.float().to(device)
            labels.float().to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy.append(correct/total)
    print(accuracy)

    seq = np.arange(len(trainset.targets))
    np.random.shuffle(seq)
    sample_indeces_list = np.split(seq, len(trainset.targets) // batch_size)

    torch.save(net.state_dict(), "tmp.pth")
    torch.save(net.state_dict(), "tmp-cache.pth")
    for i in range(num_epochs):
        print("The %d-th epoch starts." % i)
        start_time = time.time()
        est_loss = 0.0

        for j in range(len(sample_indeces_list)):
            data = trainset.data[sample_indeces_list[j]]
            data = data.unsqueeze(1)
            data = data.float().to(device)
            # data = data/255.0
            target = trainset.targets[sample_indeces_list[j]].to(device)

            # Stage 1 - One step gradient descent. Freeze the parameters of neural network; do descent on y
            net = Net()
            net.load_state_dict(torch.load("tmp.pth"))
            net.to(device)
            y = torch.from_numpy(y_total[j]).float().to(device)
            y.requires_grad = False
            output = net(y)
            loss = F.nll_loss(output, target)
            for param in net.parameters():
                param.requires_grad = True
            net.zero_grad()
            loss.backward()

            tmp_net = Net()
            tmp_net.load_state_dict(torch.load("tmp-cache.pth"))
            with torch.no_grad():
                for p, p_past in zip(net.parameters(), tmp_net.parameters()):
                    new_val = p - lr1*p.grad + momentum1 * (p - p_past)
                    p_past.copy_(p)
                    p.copy_(l2proximal(new_val, 0.0001))
            torch.save(tmp_net.state_dict(), "tmp-cache.pth") # parameter no update
            torch.save(net.state_dict(), "tmp.pth") # updated parameter

            # train_loss.append(np.copy(loss.item()))

            # Stage 2 - One step gradient ascent. Freeze y; do ascent on parameter x

            net = Net()
            net.load_state_dict(torch.load("tmp-cache.pth"))
            net.to(device)
            y = torch.from_numpy(y_total[j]).float().to(device)
            y.requires_grad = True
            y_past = torch.from_numpy(y_cache[j]).float().to(device)
            y_past.requires_grad = False
            y_cache[j] = np.copy(y.data.cpu())
            output = net(y)
            for param in net.parameters():
                param.requires_grad = False
            loss1 = F.nll_loss(output, target)
            loss2 = torch.sum((data - y) ** 2) #/batch_size  #/IMG_ROWS/IMG_COLS
            loss = loss1 - gamma * loss2
            loss.backward()
            y.data = y.data + lr2 * y.grad.data + momentum2 * (y.data - y_past.data)
            y.data = l1proximal(y.data, 0.0001)
            y.grad.data.zero_()
            y_total[j] = np.copy(y.data.cpu())

            # Evaluate the estimated maxmium
            est_net = Net()
            est_net.load_state_dict(torch.load("tmp.pth"))
            est_net.to(device)
            est_y = torch.from_numpy(y_total[j]).float().to(device)
            est_y.requires_grad = True
            for param in est_net.parameters():
                param.requires_grad = False
            for _ in range(100):
                est_output = net(est_y)
                loss1 = F.nll_loss(est_output, target)
                loss2 = torch.sum((data - est_y) ** 2)
                loss = loss1 - gamma * loss2
                loss.backward()
                est_y.data = est_y.data + 0.1 * est_y.grad.data
                est_y.grad.data.zero_()
            est_loss += loss.item()
        train_loss.append(est_loss)

        # Evaluate Test Accuracy
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                                 shuffle=False, num_workers=0)
        test_net = Net().to(device)
        test_net.load_state_dict(torch.load("tmp.pth"))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.float().to(device)
                labels.float().to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy.append(correct / total)

        print("Epoch ends. Time spent:", time.time() - start_time)
    return train_loss, accuracy