#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 23:19:45 2023

@author: vaishnavijanakiraman
"""

from torchvision import datasets

class CIFAR10_ds(datasets.CIFAR10):
    def __init__(self, root=".", train=True, download=True, transform= None):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label