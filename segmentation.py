#Import libraries
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from pathlib import Path
import random
from models.unet import UNet, DiceBCELoss 

# Configuration
SEG_CONFIG = {
    "data_dir":   "data/lgg_segmentation/kaggle_3m",
    "img_size":   256,
    "batch_size": 8,
    "epochs":     30,
    "lr":         1e-4,
    "device":     "cuda" if torch.cuda.is_available() else "cpu",
    "save_path":  "outputs/best_unet.pth",
    "seed":       42,
}

# Reproducibility
random.seed(SEG_CONFIG["seed"])
np.random.seed(SEG_CONFIG["seed"])
torch.manual_seed(SEG_CONFIG["seed"])
os.makedirs("outputs", exist_ok=True)

# Dataset
class SegDataset(Dataset):
    def __init__(self, img_paths, mask_paths, img_size=256, augment=False):
        self.imgs = img_paths
        self.masks = mask_paths
        self.augment = augment
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img  = Image.open(self.imgs[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])
        if self.augment:
            if random.random() > 0.5:
                img, mask = transforms.functional.hflip(img), transforms.functional.hflip(mask)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                img = transforms.functional.rotate(img, angle)
                mask = transforms.functional.rotate(mask, angle)
        
        mask_tensor = (self.mask_tf(mask) > 0.5).float()
        return self.img_tf(img), mask_tensor

def dice_score(logits, masks, threshold=0.5, smooth=1.0):
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * masks).sum(dim=(2, 3))
    return ((2 * inter + smooth) / (preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) + smooth)).mean().item()

# Training
def run_segmentation_training():
    device = SEG_CONFIG["device"]
    
    # Data Discovery
    mask_paths = sorted(glob.glob(f'{SEG_CONFIG["data_dir"]}/**/*_mask.tif', recursive=True))
    img_paths  = [p.replace('_mask.tif', '.tif') for p in mask_paths]
    
    # Patient-level split (important for medical imaging)
    patients = list(set(Path(p).parent.name for p in img_paths))
    random.shuffle(patients)
    n_val = max(1, int(len(patients) * 0.2))
    val_patients = set(patients[:n_val])

    tr_imgs = [p for p in img_paths if Path(p).parent.name not in val_patients]
    tr_masks = [p for p in mask_paths if Path(p).parent.name not in val_patients]
    vl_imgs = [p for p in img_paths if Path(p).parent.name in val_patients]
    vl_masks = [p for p in mask_paths if Path(p).parent.name in val_patients]

    train_ds = SegDataset(tr_imgs, tr_masks, SEG_CONFIG["img_size"], augment=True)
    val_ds = SegDataset(vl_imgs, vl_masks, SEG_CONFIG["img_size"], augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=SEG_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=SEG_CONFIG["batch_size"])

    # Initialize Model
    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=SEG_CONFIG["lr"])
    
    print(f"Starting Segmentation Training on {device}...")
    best_dice = 0

    for epoch in range(1, SEG_CONFIG["epochs"] + 1):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward(); optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                val_dice += dice_score(model(imgs), masks)
        
        avg_dice = val_dice / len(val_loader)
        print(f"Epoch {epoch} | Loss: {train_loss/len(train_loader):.4f} | Val Dice: {avg_dice:.4f}")

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), SEG_CONFIG["save_path"])

    # Results
    model.load_state_dict(torch.load(SEG_CONFIG["save_path"]))
    model.eval()
    
    # Reverse normalization for plotting
    inv_norm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(3):
        img_t, mask_t = val_ds[random.randint(0, len(val_ds)-1)]
        with torch.no_grad():
            pred = torch.sigmoid(model(img_t.unsqueeze(0).to(device)))
        
        axes[i, 0].imshow(inv_norm(img_t).permute(1, 2, 0).clamp(0,1))
        axes[i, 0].set_title("MRI")
        axes[i, 1].imshow(mask_t.squeeze(), cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(pred.cpu().squeeze() > 0.5, cmap='jet')
        axes[i, 2].set_title("U-Net Prediction")
    plt.tight_layout()
    plt.show()

run_segmentation_training()
