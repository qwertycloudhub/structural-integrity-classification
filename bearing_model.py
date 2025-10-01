# -------------------------------
# Bearing Condition Classifier
# Full EDA + Training + Testing + Plots
# Includes per-class accuracy & misclassified images
# -------------------------------

import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------------------------------
# Step 1: Paths and device
# -------------------------------
data_dir = os.path.expanduser(
    "~/Downloads/data/Bearing Condition State Classifier/original"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Directory to save plots
plot_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plot_dir, exist_ok=True)

# -------------------------------
# Step 2: Data transforms
# -------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
# Class distribution
train_targets = [s[1] for s in full_train_dataset.samples]
counter = Counter(train_targets)
plt.figure()
plt.bar([class_names[i] for i in counter.keys()], counter.values())
plt.title("Train Set Class Distribution")
plt.savefig(os.path.join(plot_dir, "train_class_distribution.png"))
plt.show()
plt.close()

# Sample images
def imshow(inp, title=None, filename=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title: plt.title(title)
    plt.axis('off')
    if filename:
        plt.savefig(os.path.join(plot_dir, filename))
    plt.show()
    plt.close()

shown = set()
for img, label in full_train_dataset:
    if label not in shown:
        imshow(img, title=f"Class: {class_names[label]}", filename=f"class_{class_names[label]}.png")
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
# Step 6: Training loop with history tracking
# -------------------------------
train_losses = []
val_accuracies = []

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=20):
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
        train_losses.append(train_loss)

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
        val_accuracies.append(val_acc.item())

        print(f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    print("Training complete. Best val acc:", best_acc.item())

# Train the model
train_model(model, criterion, optimizer, train_loader, val_loader, epochs=20)

# -------------------------------
# Step 7: Plot training curves
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_accuracies, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Progress")
plt.legend()
plt.savefig(os.path.join(plot_dir, "training_progress.png"))
plt.show()
plt.close()

# -------------------------------
# Step 8: Test evaluation
# -------------------------------
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
correct = 0
all_preds = []
all_labels = []
misclassified_images = []
misclassified_labels = []
misclassified_preds = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)

        # Collect for analysis
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Misclassified
        mis_mask = preds != labels
        misclassified_images.extend(inputs[mis_mask].cpu())
        misclassified_labels.extend(labels[mis_mask].cpu())
        misclassified_preds.extend(preds[mis_mask].cpu())

print("Test Accuracy:", correct.double() / len(test_dataset))

# -------------------------------
# Step 9: Sample predictions
# -------------------------------
images, labels = next(iter(test_loader))
images, labels = images[:5], labels[:5]
outputs = model(images.to(device))
_, preds = torch.max(outputs, 1)
images = images.cpu()
preds = preds.cpu()
labels = labels.cpu()

for i in range(5):
    imshow(images[i], title=f"True: {class_names[labels[i]]}, Pred: {class_names[preds[i]]}", filename=f"sample_pred_{i}.png")

# -------------------------------
# Step 10: Confusion matrix
# -------------------------------
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"))
plt.show()
plt.close()

# -------------------------------
# Step 11: Per-class accuracy plot
# -------------------------------
class_correct = [0]*len(class_names)
class_total = [0]*len(class_names)

for label, pred in zip(all_labels, all_preds):
    class_total[label] += 1
    if label == pred:
        class_correct[label] += 1

per_class_acc = [c/t if t>0 else 0 for c, t in zip(class_correct, class_total)]
plt.figure()
plt.bar(class_names, per_class_acc)
plt.ylim([0, 1])
plt.title("Per-Class Accuracy")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(plot_dir, "per_class_accuracy.png"))
plt.show()
plt.close()

# -------------------------------
# Step 12: Visualize misclassified samples
# -------------------------------
num_to_show = min(10, len(misclassified_images))
for i in range(num_to_show):
    imshow(
        misclassified_images[i],
        title=f"True: {class_names[misclassified_labels[i]]}, Pred: {class_names[misclassified_preds[i]]}",
        filename=f"misclassified_{i}.png"
    )

# Save model after training
