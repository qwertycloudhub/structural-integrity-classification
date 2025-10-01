# -------------------------------
# Bearing Condition Classifier
# Complete EDA + Training + Testing
# -------------------------------

import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# -------------------------------
# Step 1: Paths and device
# -------------------------------
data_dir = os.path.expanduser(
    "~/Downloads/data/Bearing Condition State Classifier/original"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# Step 2: Data transforms
# -------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------------------
# Step 3: Load datasets
# -------------------------------
full_train_dataset = datasets.ImageFolder(os.path.join(data_dir, "Train"), transform=train_transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "Test"), transform=test_transform)

# Split training into train + val (80/20)
val_size = int(0.2 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = full_train_dataset.classes
print("Classes (folder names):", class_names)
print("Class to idx mapping:", full_train_dataset.class_to_idx)

# -------------------------------
# Step 4: Basic EDA plots
# -------------------------------
# Count per class
train_targets = [s[1] for s in full_train_dataset.samples]
counter = Counter(train_targets)
plt.bar([class_names[i] for i in counter.keys()], counter.values())
plt.title("Train Set Class Distribution")
plt.show()


# Sample images
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title: plt.title(title)
    plt.axis('off')
    plt.show()


# Show 1 sample per class
shown = set()
for img, label in full_train_dataset:
    if label not in shown:
        imshow(img, title=f"Class: {class_names[label]}")
        shown.add(label)
    if len(shown) == len(class_names):
        break

# -------------------------------
# Step 5: Model setup
# -------------------------------
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# -------------------------------
# Step 6: Training loop
# -------------------------------
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    best_acc = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Training
        model.train()
        running_loss, running_corrects = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    print("Training complete. Best val acc:", best_acc.item())


# Run training
train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10)

# -------------------------------
# Step 7: Test evaluation
# -------------------------------
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
correct = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)

print("Test Accuracy:", correct.double() / len(test_dataset))

# -------------------------------
# Step 8: Optional: Map classes to 1-4
# -------------------------------
name_to_number = {name: i + 1 for i, name in enumerate(class_names)}
print("Class to number mapping:", name_to_number)