import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class CircleDataset(Dataset):
    def __init__(self, length=1000, height=140, width=480, transform=None, target_transform=None):
        self.length = length
        self.height = height
        self.width = width
        self.transform = transform 
        self.target_transform = target_transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = np.full((self.height, self.width), 0, dtype=np.uint8)
        image = image.astype(np.float32) / 255.0
        noise = np.random.normal(loc=0.2, scale=0.3, size=image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)

        color = random.randint(100, 255)
        radius = random.randint(20, 60)
        center_x = random.randint(radius, self.width - radius)
        center_y = random.randint(radius, self.height - radius)
        cv2.circle(image, (center_x, center_y), radius, color, thickness=-1)

        print(image.shape)
        image = torch.tensor(np.expand_dims(image, axis=0), dtype=torch.float32) # (140, 480) - > [1, 140, 480]
        print(image.shape)
        label = torch.tensor(1, dtype=torch.long) # good practice to make it one value tensor, setting all of them to 1 cuz there are all correct images

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

dataset = CircleDataset(length=10000)
img1, label1 = dataset[0]
img2, label2 = dataset[1]


