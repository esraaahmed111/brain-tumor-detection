# Import libraries
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import warnings

# Configuration
warnings.filterwarnings('ignore')

CONFIG = {
    "zip_path":   "archive.zip",
    "data_dir":   "data/brain_tumor",
    "model_name": "resnet50",
    "img_size":   224,
    "batch_size": 16,
    "epochs":     15,
    "lr":         1e-4,
    "device":     "cuda" if torch.cuda.is_available() else "cpu",
    "save_path":  "best_brain_model.pth",
    "seed":       42,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
os.makedirs("outputs", exist_ok=True)


# Dataset
class BrainTumorDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# Data Preparation
def prepare_data(config):
    # Extract zip if needed
    if not os.path.exists(config["data_dir"]):
        print(f"Extracting {config['zip_path']}...")
        with zipfile.ZipFile(config["zip_path"], 'r') as zf:
            zf.extractall(config["data_dir"])

    # Discover images
    data_path   = Path(config["data_dir"])
    unique_dirs = sorted([
        d for d in data_path.rglob('*')
        if d.is_dir() and any(f.suffix.lower() in {'.jpg', '.png'} for f in d.iterdir())
    ])
    class_names = [d.name for d in unique_dirs]

    samples = []
    for idx, d in enumerate(unique_dirs):
        for f in d.iterdir():
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                samples.append((str(f), idx))

    # Split data
    train_s, val_s = train_test_split(
        samples, test_size=0.2,
        random_state=config["seed"],
        stratify=[s[1] for s in samples]
    )

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(BrainTumorDataset(train_s, train_tf), batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(BrainTumorDataset(val_s,   val_tf),   batch_size=config["batch_size"])

    return train_loader, val_loader, class_names


# Training
def build_and_train():
    train_loader, val_loader, class_names = prepare_data(CONFIG)
    print(f"Device: {CONFIG['device']} | Classes: {class_names}")

    # Build model
    model    = models.resnet50(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model    = model.to(CONFIG["device"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    history  = {"loss": [], "acc": []}
    best_acc = 0

    # Training loop
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(CONFIG["device"]), labels.to(CONFIG["device"])
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(CONFIG["device"]), labels.to(CONFIG["device"])
                preds = model(imgs).argmax(1)
                correct     += (preds == labels).sum().item()
                total       += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = correct / total
        history["loss"].append(train_loss / len(train_loader))
        history["acc"].append(val_acc)

        print(f"Epoch {epoch}/{CONFIG['epochs']} | Loss: {history['loss'][-1]:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CONFIG["save_path"])

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.close()

    # Training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], color='steelblue', lw=2)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.grid(alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history["acc"], color='tomato', lw=2)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/training_curves.png', dpi=150)
    plt.close()

    print(f"Done. Best Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    build_and_train()
