import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np


def main():
    DEVICE = torch.device("cuda")

    random_affine = torchvision.transforms.RandomAffine(degrees=10,
                                                        scale=(0.9, 1.1),
                                                        translate=(0.1, 0.1),
                                                        interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

    transform1 = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform2 = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 128

    training_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform1)
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = torchvision.models.quantization.resnet50(weights=None, progress=True, quantize=False)
    # net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(32, 32), stride=(2, 2), padding=(3, 3), bias=False)
    net.train()
    net = net.to(DEVICE)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(60):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs = inputs.to(DEVICE)  # Send input to GPU or CPU
            labels = labels.to(DEVICE)
            outputs = net(inputs)  # Predict
            loss = criterion(outputs, labels)  # Score / Evaluate
            loss.backward()  # Determine how each parameter effected the loss
            optimizer.step()  # Update parameters

            # print statistics
            running_loss += loss.item()
            if i % 200 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.6f templost: %.6f' %
                      (epoch + 1, i + 1, running_loss / 200, loss.item()))
                running_loss = 0.0

    print('Finished Training')

    correct_counter = 0
    all_counter = 0
    timess = 0

    net.eval()

    for i, data in enumerate(train_loader, 0):
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

        if timess > 10:
            break
        else:
            timess += 1

    print(correct_counter, all_counter, correct_counter / all_counter)

    for i, data in enumerate(test_loader, 0):
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

        if timess > 10:
            break
        else:
            timess += 1

    print(correct_counter, all_counter, correct_counter / all_counter)

    torch.save(net.to("cpu"), 'checkpoint/resnet1.pth')


if __name__ == '__main__':
    main()
