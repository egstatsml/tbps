#!/usr/bin/env python3

import torch

import torchvision
import torchvision.transforms as transforms



if __name__ == '__main__':
    print('==> Preparing data...')
    transform_train = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data',
                                             train=True,
                                             download=True,
                                             transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              pin_memory=True,
                                              shuffle=True,
                                              num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset,
                                             shuffle=False,
                                             num_workers=2)
