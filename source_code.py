import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import os

# ------------------- Configuration ------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "D:\DeepLearning\cancer\Lung_and_Colon_Cancer" #you can change the dataset path
batch_size = 32
num_epochs = 30
learning_rate = 0.001
image_size = 224
num_classes = 5  # 5 classes in the dataset

# ------------------- Data Transformations ------------------- #
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ------------------- Load Dataset ------------------- #
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes
print(f"Loaded dataset with {len(class_names)} classes: {class_names}")

# ------------------- Stratified Split ------------------- #
def stratified_split(dataset, test_size=0.2):
    labels = [label for _, label in dataset]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

    for train_idx, val_idx in sss.split(np.zeros(len(labels)), labels):
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

    return train_dataset, val_dataset

# Apply stratified split
train_dataset, val_dataset = stratified_split(dataset, test_size=0.2)
print(f"Training size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ------------------- MLP-Mixer Block ------------------- #
class MixerBlock(nn.Module):
    def __init__(self, num_patches, hidden_dim, token_dim, channel_dim, dropout=0.1):
        super(MixerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.mlp_token = nn.Sequential(
            nn.Linear(num_patches, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_patches),
            nn.Dropout(dropout)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.mlp_channel = nn.Sequential(
            nn.Linear(hidden_dim, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        y = self.layer_norm1(x)
        y = y.transpose(1, 2)
        y = self.mlp_token(y)
        y = y.transpose(1, 2)
        x = x + y
        y = self.layer_norm2(x)
        y = self.mlp_channel(y)
        return x + y

# ------------------- MLP-Mixer Model ------------------- #
class MLPMixer(nn.Module):
    def __init__(self, num_classes, num_blocks=8, num_patches=196, hidden_dim=512, token_dim=256, channel_dim=2048):
        super(MLPMixer, self).__init__()
        self.patch_embedding = nn.Conv2d(3, hidden_dim, kernel_size=16, stride=16)
        self.mixer_blocks = nn.Sequential(*[MixerBlock(num_patches, hidden_dim, token_dim, channel_dim) for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.mixer_blocks(x)
        x = self.layer_norm(x.mean(dim=1))
        x = self.mlp_head(x)
        return x

# ------------------- Training and Evaluation ------------------- #
def train_and_evaluate(model, train_loader, val_loader, class_names):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        # ------------------- Training Phase ------------------- #
        model.train()
        correct, total, train_loss = 0, 0, 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # ------------------- Validation Phase ------------------- #
        model.eval()
        correct, total, val_loss = 0, 0, 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        print(f"\nEpoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Save Model
    torch.save(model.state_dict(), "mixer_mlp_cancer_v2.pth")
    print("✅ Model saved successfully as 'mixer_mlp_cancer_v2.pth'")

    # Plot Loss and Accuracy
    plot_metrics(train_losses, val_losses, train_accs, val_accs)

    # Generate Classification Report and Confusion Matrix
    generate_classification_report(model, val_loader, class_names)

# ------------------- Plot Training Metrics ------------------- #
def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, marker='o', label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accs, marker='o', label="Train Acc")
    plt.plot(range(1, num_epochs + 1), val_accs, marker='o', label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.grid(True)
    plt.legend()

    plt.savefig("training_metrics_v2.png")
    print("📊 Saved training and validation graphs as 'training_metrics_v2.png'")
    plt.show()

# ------------------- Generate Classification Report ------------------- #
def generate_classification_report(model, val_loader, class_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:\n", report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_v2.png")
    print("📊 Saved confusion matrix as 'confusion_matrix_v2.png'")
    plt.show()

# ------------------- Main Execution ------------------- #
def main():
    model = MLPMixer(num_classes=num_classes).to(device)
    train_and_evaluate(model, train_loader, val_loader, class_names)

if __name__ == "__main__":
    main()
