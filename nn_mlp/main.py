import os
import torch
import random
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 15),
            nn.ReLU(),
            nn.Linear(15, 15),
            nn.ReLU(),
            nn.Linear(15, 10) 
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

label_map = {
    0:'Fire', 
    1:'Water', 
    2:'Grass', 
    3:'Electric', 
    4:'Psychic', 
    5:'Ghost', 
    6:'Dark', 
    7:'Fighting', 
    8:'Bug', 
    9:'Normal'}

def generate_pokemon_stat_and_label():
    label = random.randint(0,9)
    if label_map[label] == 'Fire':
        stats = [random.randint(40, 80), random.randint(60, 100), random.randint(30, 60), random.randint(70, 120), random.randint(60, 100)]
    elif label_map[label] == 'Water':
        stats = [random.randint(50, 100), random.randint(50, 90), random.randint(60, 100), random.randint(50, 90), random.randint(70, 110)]
    elif label_map[label] == 'Grass':
        stats = [random.randint(50, 90), random.randint(60, 90), random.randint(50, 80), random.randint(40, 80), random.randint(70, 100)]
    elif label_map[label] == 'Electric':
        stats = [random.randint(30, 60), random.randint(60, 90), random.randint(40, 70), random.randint(80, 130), random.randint(70, 100)]
    elif label_map[label] == 'Psychic':
        stats = [random.randint(40, 70), random.randint(40, 70), random.randint(40, 70), random.randint(50, 90), random.randint(100, 130)]
    elif label_map[label] == 'Ghost':
        stats = [random.randint(40, 70), random.randint(50, 80), random.randint(60, 90), random.randint(50, 80), random.randint(80, 110)]
    elif label_map[label] == 'Dark':
        stats = [random.randint(60, 90), random.randint(80, 100), random.randint(60, 90), random.randint(70, 100), random.randint(50, 80)]
    elif label_map[label] == 'Fighting':
        stats = [random.randint(60, 90), random.randint(90, 130), random.randint(70, 100), random.randint(60, 90), random.randint(30, 60)]
    elif label_map[label] == 'Bug':
        stats = [random.randint(40, 70), random.randint(40, 70), random.randint(40, 70), random.randint(60, 90), random.randint(40, 70)]
    else:  # Normal
        stats = [random.randint(60, 90), random.randint(60, 90), random.randint(60, 90), random.randint(60, 90), random.randint(60, 90)]
    
    return torch.tensor(stats, dtype=torch.float32), torch.tensor(label, dtype=torch.long) 

class PokemonDataset(Dataset):
    def __init__(self, length=1024):
        self.data = [generate_pokemon_stat_and_label() for _ in range(length)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
dataset = PokemonDataset(length=1024)

train_dataloader = DataLoader(dataset, 64, shuffle=True)
test_dataloader = DataLoader(dataset, 64, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


loss_fn = nn.CrossEntropyLoss()
model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1024):
    # === Learning phase ===
    for stats, target_labels in train_dataloader:
        stats = stats.to(device)
        target_labels = target_labels.to(device)

        predicted_labels = model(stats)
        loss = loss_fn(predicted_labels, target_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

    # === Evaluation after each epoch ===
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(): # turned of gradient calc — only need while learning, saving RAM and is faster (less strain on GPU)
        for stats, labels in test_dataloader:
            stats = stats.to(device)
            labels = labels.to(device)
            outputs = model(stats)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    model.train() # after eval we go back to traning mode, because in eval section we check how well the model works


input("Press enter to continue...")
os.system('cls')

model_path = os.path.join("saved_models", "pokemon_model.pth")
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
