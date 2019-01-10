from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
import numpy as np
import gzip
import random
from PIL import Image


class RandomCycleIter:
    """Randomly iterate element in each cycle
    Example:
        >>> rand_cyc_iter = RandomCycleIter([1, 2, 3])
        >>> [next(rand_cyc_iter) for _ in range(10)]
        [2, 1, 3, 2, 3, 1, 1, 2, 3, 2]
    """
    def __init__(self, data):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1

    def next(self):
        self.i += 1
        if self.i == self.length:
            self.i = 0
            random.shuffle(self.data_list)
        return self.data_list[self.i]


class ClassAwareMNIST(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            train_dataset = datasets.MNIST('~/data', train=True, download=download)
            self.imgs, self.labels = train_dataset.train_data.numpy(), train_dataset.train_labels.numpy()
        else:
            test_dataset = datasets.MNIST('~/data', train=False, download=download)
            self.imgs, self.labels = test_dataset.test_data.numpy(), test_dataset.test_labels.numpy()

        n_classes = len(np.unique(self.labels))
        self.class_iter = RandomCycleIter(range(n_classes))
        class_data_list = [list() for _ in range(n_classes)]
        for i, label in enumerate(self.labels):
            class_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in class_data_list]

    def __getitem__(self, index):
        target = self.class_iter.next()
        img1 = self.imgs[self.data_iter_list[target].next()]
        img2 = self.imgs[self.data_iter_list[target].next()]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img1 = Image.fromarray(img1, mode='L')
        img2 = Image.fromarray(img2, mode='L')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, target

    def __len__(self):
        return len(self.imgs) // 2
