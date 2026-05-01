# Import libraries
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
        self.imgs    = img_paths
        self.masks   = mask_paths
        self.augment = augment
        mean, std    = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.img_tf  = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img  = Image.open(self.imgs[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])
        if self.augment:
            if random.random() > 0.5:
                img  = transforms.functional.hflip(img)
                mask = transforms.functional.hflip(mask)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                img   = transforms.functional.rotate(img,  angle)
                mask  = transforms.functional.rotate(mask, angle)
        mask_tensor = (self.mask_tf(mask) > 0.5).float()
        return self.img_tf(img), mask_tensor


# Metrics
def dice_score(logits, masks, threshold=0.5, smooth=1.0):
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * masks).sum(dim=(2, 3))
    return ((2 * inter + smooth) / (preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) + smooth)).mean().item()

def iou_score(logits, masks, threshold=0.5, smooth=1.0):
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * masks).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) - inter
    return ((inter + smooth) / (union + smooth)).mean().item()


# Training
def run_segmentation_training():
    device = SEG_CONFIG["device"]

    # Data discovery
    mask_paths = sorted(glob.glob(f'{SEG_CONFIG["data_dir"]}/**/*_mask.tif', recursive=True))
    img_paths  = [p.replace('_mask.tif', '.tif') for p in mask_paths]

    missing = [p for p in img_paths if not os.path.exists(p)]
    if missing:
        print(f"Warning: {len(missing)} missing images. Check data_dir.")
        return

    print(f"Found {len(img_paths)} image-mask pairs")

    # Patient-level split
    patients     = list(set(Path(p).parent.name for p in img_paths))
    random.shuffle(patients)
    n_val        = max(1, int(len(patients) * 0.2))
    val_patients = set(patients[:n_val])

    tr_imgs  = [p for p in img_paths  if Path(p).parent.name not in val_patients]
    tr_masks = [p for p in mask_paths if Path(p).parent.name not in val_patients]
    vl_imgs  = [p for p in img_paths  if Path(p).parent.name in val_patients]
    vl_masks = [p for p in mask_paths if Path(p).parent.name in val_patients]

    print(f"Train: {len(tr_imgs)}  Val: {len(vl_imgs)}")

    # Dataloaders
    sz           = SEG_CONFIG["img_size"]
    n_workers    = 0 if os.name == 'nt' else 2
    train_ds     = SegDataset(tr_imgs, tr_masks, sz, augment=True)
    val_ds       = SegDataset(vl_imgs, vl_masks, sz, augment=False)
    train_loader = DataLoader(train_ds, batch_size=SEG_CONFIG["batch_size"], shuffle=True,  num_workers=n_workers)
    val_loader   = DataLoader(val_ds,   batch_size=SEG_CONFIG["batch_size"], shuffle=False, num_workers=n_workers)

    # Model
    model     = UNet(n_channels=3, n_classes=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=SEG_CONFIG["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_dice  = 0
    seg_history = {"loss": [], "dice": [], "iou": []}

    print(f"Starting Segmentation Training on {device}...\n")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Dice':>9}  {'Val IoU':>8}")
    print("-" * 40)

    # Training loop
    for epoch in range(1, SEG_CONFIG["epochs"] + 1):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_dice, val_iou = 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits   = model(imgs)
                val_dice += dice_score(logits, masks)
                val_iou  += iou_score(logits,  masks)

        avg_loss = train_loss / len(train_loader)
        avg_dice = val_dice  / len(val_loader)
        avg_iou  = val_iou   / len(val_loader)
        scheduler.step(avg_loss)

        seg_history["loss"].append(avg_loss)
        seg_history["dice"].append(avg_dice)
        seg_history["iou"].append(avg_iou)

        print(f"{epoch:>6}  {avg_loss:>10.4f}  {avg_dice:>9.4f}  {avg_iou:>8.4f}")

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), SEG_CONFIG["save_path"])

    # Training curves
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(seg_history["loss"], color='steelblue', lw=2)
    plt.title("Train Loss"); plt.xlabel("Epoch"); plt.grid(alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.plot(seg_history["dice"], color='seagreen', lw=2)
    plt.title("Val Dice"); plt.xlabel("Epoch"); plt.grid(alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.plot(seg_history["iou"], color='purple', lw=2)
    plt.title("Val IoU"); plt.xlabel("Epoch"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/segmentation_curves.png", dpi=150)
    plt.close()

    # Predictions visualization
    model.load_state_dict(torch.load(SEG_CONFIG["save_path"]))
    model.eval()
    inv_norm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for i in range(3):
        img_t, mask_t = val_ds[random.randint(0, len(val_ds) - 1)]
        with torch.no_grad():
            logit = model(img_t.unsqueeze(0).to(device))
        prob = torch.sigmoid(logit).cpu().squeeze().numpy()
        pred = (prob > 0.5).astype(np.uint8)
        gt   = mask_t.squeeze().numpy()
        img_d = inv_norm(img_t).permute(1, 2, 0).clamp(0, 1).numpy()

        axes[i, 0].imshow(img_d);              axes[i, 0].set_title("MRI Scan")
        axes[i, 1].imshow(gt,   cmap='gray');  axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(pred, cmap='gray');  axes[i, 2].set_title("Prediction")

        ov = (img_d * 255).astype(np.uint8).copy()
        ov[(pred == 1) & (gt == 1)] = [0,   200, 0]
        ov[(pred == 1) & (gt == 0)] = [200, 0,   0]
        ov[(pred == 0) & (gt == 1)] = [0,   0, 200]
        axes[i, 3].imshow(ov); axes[i, 3].set_title("TP=green FP=red FN=blue")
        for ax in axes[i]: ax.axis('off')

    plt.suptitle("Segmentation Results", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/segmentation_results.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nDone. Best Val Dice: {best_dice:.4f}")


if __name__ == "__main__":
    run_segmentation_training()
