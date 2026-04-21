#Import libraries
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
    "zip_path": "archive.zip",           # Path to your uploaded zip
    "data_dir": "data/brain_tumor",      # Where data will be extracted
    "model_name": "resnet50",            # resnet50 | efficientnet
    "img_size": 224,
    "batch_size": 16,
    "epochs": 15,
    "lr": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "best_brain_model.pth",
    "seed": 42,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
os.makedirs("outputs", exist_ok=True)

# Dataset
class BrainTumorDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label

def prepare_data(config):
    # Extract zip if folder doesn't exist
    if not os.path.exists(config["data_dir"]):
        print(f"Extracting {config['zip_path']}...")
        with zipfile.ZipFile(config["zip_path"], 'r') as zf:
            zf.extractall(config["data_dir"])
    
    # Discover Images
    data_path = Path(config["data_dir"])
    unique_dirs = sorted([d for d in data_path.rglob('*') if d.is_dir() and any(f.suffix.lower() in {'.jpg', '.png'} for f in d.iterdir())])
    class_names = [d.name for d in unique_dirs]
    
    samples = []
    for idx, d in enumerate(unique_dirs):
        for f in d.iterdir():
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                samples.append((str(f), idx))
    
    # Split Data
    train_s, val_s = train_test_split(samples, test_size=0.2, random_state=config["seed"], stratify=[s[1] for s in samples])
    
    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(BrainTumorDataset(train_s, train_tf), batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(BrainTumorDataset(val_s, val_tf), batch_size=config["batch_size"])
    
    return train_loader, val_loader, class_names

# MODEL
def build_and_train():
    # Load Data
    train_loader, val_loader, class_names = prepare_data(CONFIG)
    print(f"Device: {CONFIG['device']} | Classes: {class_names}")

    # Build Model
    model = models.resnet50(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(CONFIG["device"])

    # Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    
    history = {"loss": [], "acc": []}
    best_acc = 0

    # Loop
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
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(CONFIG["device"]), labels.to(CONFIG["device"])
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        history["loss"].append(train_loss / len(train_loader))
        history["acc"].append(val_acc)
        
        print(f"Epoch {epoch}/{CONFIG['epochs']} - Loss: {history['loss'][-1]:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CONFIG["save_path"])

    # Final Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history["loss"]); plt.title("Training Loss")
    plt.subplot(1, 2, 2); plt.plot(history["acc"]); plt.title("Validation Accuracy")
    plt.show()
    
    print(f"Done! Best Accuracy: {best_acc:.4f}")

build_and_train()
