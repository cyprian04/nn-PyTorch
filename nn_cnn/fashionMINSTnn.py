import os
import cv2
import torch
import random
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda



class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential( # getting out textures, feateures etc (convolution)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # -> (32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                               # -> (32, 14, 14)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),# -> (64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                                # -> (64, 7, 7)
        )
        self.fc_layer = nn.Sequential( # decision making (clasification based after conv)
            nn.Flatten(),                                                        # -> (64*7*7)
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        logits = self.conv_layer(x)
        logits = self.fc_layer(logits)
        return logits


train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
    )
# dont need target_transform to do one-hot encoding, CrossEntropyLoss will do it by itself and then dims are ok
test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
    )

train_dataloader = DataLoader(train_dataset, 32, shuffle=True)
test_dataloader = DataLoader(test_dataset, 32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

loss_fn = nn.CrossEntropyLoss()
model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(10):
    # === Learning phase ===
    for images, target_labels in train_dataloader:
        images = images.to(device)
        target_labels = target_labels.to(device)

        predicted_labels = model(images)
        loss = loss_fn(predicted_labels, target_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")
    
    # === Evaluation after each epoch ===
    model.eval()
    correct = 0 
    total = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            accuracy = 100 * correct / total
            print(f"Test Accuracy: {accuracy:.2f}%")
    model.train()

input("Press enter to continue...")
os.system('cls')

model_path = os.path.join("saved_models", "fashionMNIST_model.pth")
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")