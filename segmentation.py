import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from pathlib import Path
import random

from models.unet import UNet, DiceBCELoss

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

torch.manual_seed(SEG_CONFIG["seed"])
np.random.seed(SEG_CONFIG["seed"])
random.seed(SEG_CONFIG["seed"])


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
        img  = self.img_tf(img)
        mask = (self.mask_tf(mask) > 0.5).float()
        return img, mask


def dice_score(logits, masks, threshold=0.5, smooth=1.0):
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * masks).sum(dim=(2, 3))
    return ((2 * inter + smooth) / (preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) + smooth)).mean().item()

def iou_score(logits, masks, threshold=0.5, smooth=1.0):
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * masks).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) - inter
    return ((inter + smooth) / (union + smooth)).mean().item()


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, dice_sum, iou_sum = 0, 0, 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, masks).item()
        dice_sum   += dice_score(logits, masks)
        iou_sum    += iou_score(logits,  masks)
    n = len(loader)
    return total_loss / n, dice_sum / n, iou_sum / n


def visualize_predictions(model, dataset, device, n=4):
    model.eval()
    inv_norm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225])
    fig, axes = plt.subplots(n, 4, figsize=(18, 4 * n))
    idxs = random.sample(range(len(dataset)), n)
    for row, idx in enumerate(idxs):
        img_t, mask_t = dataset[idx]
        with torch.no_grad():
            logit = model(img_t.unsqueeze(0).to(device))
        prob = torch.sigmoid(logit).cpu().squeeze().numpy()
        pred = (prob > 0.5).astype(np.uint8)
        gt   = mask_t.squeeze().numpy()
        img_d = inv_norm(img_t).permute(1, 2, 0).clamp(0, 1).numpy()

        axes[row, 0].imshow(img_d);             axes[row, 0].set_title('MRI Scan')
        axes[row, 1].imshow(gt,   cmap='gray'); axes[row, 1].set_title('Ground Truth')
        axes[row, 2].imshow(pred, cmap='gray'); axes[row, 2].set_title('Prediction')

        ov = (img_d * 255).astype(np.uint8).copy()
        ov[(pred == 1) & (gt == 1)] = [0,   200, 0]
        ov[(pred == 1) & (gt == 0)] = [200, 0,   0]
        ov[(pred == 0) & (gt == 1)] = [0,   0, 200]
        axes[row, 3].imshow(ov); axes[row, 3].set_title('TP=green FP=red FN=blue')
        for ax in axes[row]: ax.axis('off')

    plt.suptitle('U-Net Segmentation Results', fontsize=13)
    plt.tight_layout()
    plt.savefig('outputs/unet_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Predictions saved → outputs/unet_predictions.png")


def main():
    os.makedirs("outputs", exist_ok=True)
    device = SEG_CONFIG["device"]
    print(f"Device: {device}")

    mask_paths = sorted(glob.glob(f'{SEG_CONFIG["data_dir"]}/**/*_mask.tif', recursive=True))
    img_paths  = [p.replace('_mask.tif', '.tif') for p in mask_paths]
    missing    = [p for p in img_paths if not os.path.exists(p)]
    if missing:
        print(f" {len(missing)} missing images — check data_dir")
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

    sz = SEG_CONFIG["img_size"]
    train_ds = SegDataset(tr_imgs, tr_masks, sz, augment=True)
    val_ds   = SegDataset(vl_imgs, vl_masks, sz, augment=False)
    n_workers = 0 if os.name == 'nt' else 2
    train_loader = DataLoader(train_ds, batch_size=SEG_CONFIG["batch_size"], shuffle=True,  num_workers=n_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=SEG_CONFIG["batch_size"], shuffle=False, num_workers=n_workers, pin_memory=True)

    model     = UNet(n_channels=3, n_classes=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=SEG_CONFIG["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_dice = 0.0
    print(f"\nTraining U-Net for {SEG_CONFIG['epochs']} epochs...\n")
    print(f"{'Epoch':>6}  {'Tr Loss':>9}  {'Vl Loss':>9}  {'Dice':>7}  {'IoU':>7}")
    print('─' * 48)

    for epoch in range(1, SEG_CONFIG["epochs"] + 1):
        tr_l          = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_l, vl_d, vl_iou = val_epoch(model, val_loader, criterion, device)
        scheduler.step(vl_l)
        flag = ''
        if vl_d > best_dice:
            best_dice = vl_d
            torch.save(model.state_dict(), SEG_CONFIG["save_path"])
            flag = '  ✓ saved'
        print(f"{epoch:>6}  {tr_l:>9.4f}  {vl_l:>9.4f}  {vl_d:>7.4f}  {vl_iou:>7.4f}{flag}")

    visualize_predictions(model, val_ds, device)
    print(f"\nDone! Best Val Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
