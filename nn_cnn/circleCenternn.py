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
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # (1, H, W) -> (32, H, W)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # (32, H/2, W/2)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # -> (64, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # -> (64, H/4, W/4)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),# -> (128, H/4, W/4)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # -> (128, H/8, W/8)
            nn.AdaptiveAvgPool2d((4, 4))                 # zapewni (128, 4, 4)
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),                # -> (128 * 4 * 4) = 2048
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 2)            # regresja: (x, y)
        )

    def forward(self, x):
            x = self.conv_layer(x)
            x = self.fc_layer(x)
            return x

class CircleDataset(Dataset):
    def __init__(self, length=1024, height=140, width=480, transform=None, target_transform=None):
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

        image = torch.tensor(np.expand_dims(image, axis=0), dtype=torch.float32)
        label = torch.tensor((center_x, center_y), dtype=torch.float32) 

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

dataset = CircleDataset(length=1024)
train_dataloader = DataLoader(dataset, 32, shuffle=True)
test_dataloader = DataLoader(dataset, 32, shuffle=True)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

loss_fn = nn.MSELoss()
model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    # === Learning phase #
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
    total_loss = 0.0
    total_distance = 0.0
    count = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)

            distances = torch.sqrt(torch.sum((outputs - labels) ** 2, dim=1))  # shape: (batch,)
            total_distance += distances.sum().item()
            count += images.size(0)

    avg_mse = total_loss / count
    avg_pixel_error = total_distance / count
    print(f"Eval -> MSE: {avg_mse:.4f}, Avg Pixel Error: {avg_pixel_error:.2f} px")
    model.train()

input("Press enter to continue...")
os.system('cls')

model_path = os.path.join("saved_models", "circleCenter_model.pth")
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
