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
warnings.filterwarnings('ignore')

CONFIG = {
    "zip_path": "archive.zip",
    "data_dir": "data/brain_tumor",
    "model_name": "resnet50",       # resnet50 | efficientnet | googlenet
    "num_classes": 2,
    "img_size": 224,
    "batch_size": 16,
    "epochs": 20,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "outputs/best_model.pth",
    "seed": 42,
}

def extract_dataset(zip_path, target_dir):
    if not os.path.exists(target_dir):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_dir)
    else:
        print(f"Dataset already extracted at {target_dir}")

def discover_images(data_dir):
    data_path = Path(data_dir)
    class_dirs = sorted([d for d in data_path.rglob('*') if d.is_dir()])
    seen, unique_dirs = set(), []
    for d in class_dirs:
        if d.name not in seen and d.name not in ('', '.'):
            seen.add(d.name)
            unique_dirs.append(d)
    label_map = {d.name.lower(): idx for idx, d in enumerate(unique_dirs)}
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    samples = []
    for d in unique_dirs:
        lbl = label_map[d.name.lower()]
        for f in d.iterdir():
            if f.suffix.lower() in exts:
                samples.append((str(f), lbl))
    class_names = [d.name for d in unique_dirs]
    print(f"Found {len(samples)} images across {len(class_names)} classes: {class_names}")
    return samples, class_names

class BrainTumorDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def get_transforms(img_size, augment=False):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def build_model(model_name, num_classes, pretrained=True):
    w = "DEFAULT" if pretrained else None
    if model_name == "resnet50":
        model = models.resnet50(weights=w)
        model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, num_classes))
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights=w)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))
    elif model_name == "googlenet":
        model = models.inception_v3(weights=w, aux_logits=True)
        model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, num_classes))
        if model.AuxLogits:
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

def train_one_epoch(model, loader, criterion, optimizer, device, is_inception=False):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if is_inception:
            outputs, aux = model(imgs)
            loss = criterion(outputs, labels) + 0.4 * criterion(aux, labels)
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        total_loss += criterion(outputs, labels).item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels

def plot_history(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_losses, label='Train', color='steelblue', lw=2)
    ax1.plot(val_losses,   label='Val',   color='tomato',    lw=2)
    ax1.set_title('Loss'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(train_accs,   label='Train', color='steelblue', lw=2)
    ax2.plot(val_accs,     label='Val',   color='tomato',    lw=2)
    ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/training_history.png', dpi=150)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.close()

def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    os.makedirs("outputs", exist_ok=True)
    device = CONFIG["device"]
    print(f"Device: {device}")

    extract_dataset(CONFIG["zip_path"], CONFIG["data_dir"])
    samples, class_names = discover_images(CONFIG["data_dir"])
    CONFIG["num_classes"] = len(class_names)

    train_s, val_s = train_test_split(samples, test_size=0.2,
                                      random_state=CONFIG["seed"],
                                      stratify=[s[1] for s in samples])
    print(f"Train: {len(train_s)}  Val: {len(val_s)}")

    train_ds = BrainTumorDataset(train_s, get_transforms(CONFIG["img_size"], augment=True))
    val_ds   = BrainTumorDataset(val_s,   get_transforms(CONFIG["img_size"], augment=False))
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    is_inception = CONFIG["model_name"] == "googlenet"
    model = build_model(CONFIG["model_name"], CONFIG["num_classes"]).to(device)
    print(f"Model: {CONFIG['model_name']}  Classes: {class_names}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    best_val_acc = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, CONFIG["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, is_inception)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);    val_accs.append(vl_acc)
        print(f"Epoch {epoch:02d}  Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}  |  Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'model_name': CONFIG["model_name"],
                'num_classes': CONFIG["num_classes"],
                'img_size': CONFIG["img_size"],
            }, CONFIG["save_path"])
            print(f"  ✓ Saved (val_acc={best_val_acc:.4f})")

    _, _, all_preds, all_labels = evaluate(model, val_loader, criterion, device)
    print("\n" + classification_report(all_labels, all_preds, target_names=class_names))
    plot_history(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(all_labels, all_preds, class_names)
    print(f"\nDone! Best Val Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
