import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class MyTransform(object):
    def __init__(self):
        pass

    def __call__(self, img):
        ne = np.array(torchvision.transforms.ToPILImage()(img))
        ne1 = np.vstack((ne[0:16], ne[2:18], ne[3:19], ne[4:20], ne[5:21], ne[6:22], ne[7:23], ne[8:24], ne[9:25],
                         ne[10:26], ne[11:27], ne[12:28], ne[13:29], ne[15:31]))
        ne2 = np.hstack((ne1[:, 0:16], ne1[:, 2:18], ne1[:, 3:19], ne1[:, 4:20], ne1[:, 5:21], ne1[:, 6:22],
                         ne1[:, 7:23], ne1[:, 8:24], ne1[:, 9:25], ne1[:, 10:26], ne1[:, 11:27], ne1[:, 12:28],
                         ne1[:, 13:29], ne1[:, 15:31]))
        return Image.fromarray(ne2)


def tran(epoch):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    scheduler.step()
    print('[%d] loss: %.3f' % (epoch + 1, running_loss))


def test():
    net.eval()
    all_counter = 0
    correct_counter = 0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        out = net(inputs)
        out = out.detach().cpu().argmax(1)
        t = labels.cpu()
        for m in range(len(t)):
            all_counter += 1
            if t[m] == out[m]:
                correct_counter += 1

    print(correct_counter, all_counter, correct_counter / all_counter)
    return (correct_counter / all_counter)


if __name__ == '__main__':

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DTYPE = torch.float32

    BATCH_SIZE = 100
    epochs = 40
    lr = 0.005
    momentum = 0.9

    transform1 = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        MyTransform(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.ConvertImageDtype(DTYPE)
    ])

    transform2 = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        MyTransform(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.ConvertImageDtype(DTYPE)
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform1)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=6)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=6)

    net = torch.load("checkpoint/vit_base_patch16_224.pth")
    net.head = torch.nn.Linear(net.head.in_features, out_features=10, bias=True)
    net = net.train().to(device=DEVICE, dtype=DTYPE)

    optimizer = torch.optim.SGD(net.head.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    loss_func = torch.nn.CrossEntropyLoss()

    correctRate = 0
    for i in range(epochs):
        tran(i)
        r = test()
        if (r > correctRate):
            correctRate = r
            print("best: ", r, " in NO: ", i)
            torch.save(net.cpu(), "checkpoint/trans_vit2.pth")
            net = net.to(DEVICE)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)
    loss_func = torch.nn.CrossEntropyLoss()

    correctRate = 0
    for i in range(epochs):
        tran(i)
        r = test()
        if (r > correctRate):
            correctRate = r
            print("best: ", r, " in NO: ", i)
            torch.save(net.cpu(), "checkpoint/trans_vit2.pth")
            net = net.to(DEVICE)
