import csv
import os
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image

def get_transform(opt, train=True):
    transforms_list = []
    if train:
        transforms_list.append(transforms.RandomCrop(32, padding=4))
        if opt.dataset == "cifar10":
            transforms_list.append(transforms.RandomHorizontalFlip())

    transforms_list.append(transforms.ToTensor())

    return transforms.Compose(transforms_list)

def get_dataloader(opt, train=True):
    transform = get_transform(opt, train)
    if opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=True
    )
    return dataloader
