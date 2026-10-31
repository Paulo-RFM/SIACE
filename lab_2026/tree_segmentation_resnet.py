import os
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.losses import DiceLoss

from torch.cuda.amp import autocast, GradScaler
from datetime import timedelta
import time

# ==============================
# CONFIGURAÇÕES
# ==============================

base_image_tiles_dir = "/home/paulomaciel/proj/imagens/"
base_mask_tiles_dir  = "/home/paulomaciel/proj/masks/"
checkpoint_path      = "/home/paulomaciel/proj/lab_2026/unet_binary_tree_detector.pth"

img_size = (512, 512)
batch_size = 6
epochs = 40
patience = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# UTILS
# ==============================

def list_all_files(directory, extension='.png'):
    result = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(extension):
                result.append(os.path.join(root, f))
    return sorted(result)

def rgb_to_binary(mask_rgb):
    """
    Converte qualquer pixel que não seja preto (0,0,0) em 1 (árvore).
    """
    mask = np.any(mask_rgb != [0, 0, 0], axis=-1)
    return mask.astype(np.uint8)

# ==============================
# DATASET
# ==============================

class TreeBinaryDataset(Dataset):
    def __init__(self, image_files, mask_files, img_size, train=True):
        self.image_files = image_files
        self.mask_files = mask_files
        self.img_size = img_size
        self.train = train

        # Aumentações para imagem
        self.color_jitter = transforms.ColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.25
        )

        self.crop = transforms.RandomResizedCrop(
            img_size,
            scale=(0.4, 1.0),     # zoom forte para destacar árvores pequenas
            ratio=(1.0, 1.0)
        )

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        mask = Image.open(self.mask_files[idx]).convert("RGB")

        # Resize inicial
        img = img.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)

        # Augmentations consistentes
        if self.train:
            # Crop sincronizado
            seed = np.random.randint(0, 100000)
            random.seed(seed)
            img = self.crop(img)
            random.seed(seed)
            mask = self.crop(mask)

            # Flips
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

            # Color jitter só na imagem
            img = self.color_jitter(img)

        # Converter máscara RGB → binária
        mask_np = rgb_to_binary(np.array(mask))
        mask_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)

        # Normalizar imagem
        img = self.to_tensor(img)
        img = self.normalize(img)

        return img, mask_tensor  # máscara 1 canal (0/1)

# ==============================
# CARREGAR ARQUIVOS
# ==============================

image_files = list_all_files(base_image_tiles_dir, '.tif')
mask_files  = list_all_files(base_mask_tiles_dir , '.png')

print("Imagens:", len(image_files))
print("Máscaras:", len(mask_files))
assert len(image_files) == len(mask_files)

image_train, image_val, mask_train, mask_val = train_test_split(
    image_files, mask_files, test_size=0.3, random_state=42
)

train_dataset = TreeBinaryDataset(image_train, mask_train, img_size, train=True)
val_dataset   = TreeBinaryDataset(image_val  , mask_val  , img_size, train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ==============================
# MODELO
# ==============================

model = Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1   # ← binário
).to(device)

# LOSS BINÁRIA
bce = nn.BCEWithLogitsLoss()
dice = DiceLoss(mode="binary", from_logits=True)

def loss_fn(pred, target):
    return 0.5 * bce(pred, target) + 0.5 * dice(pred, target)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

scaler = GradScaler()

# ==============================
# MÉTRICAS
# ==============================

def bin_metrics(pred, target):
    """
    pred: logits
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    TP = (pred * target).sum().item()
    FP = (pred * (1 - target)).sum().item()
    FN = ((1 - pred) * target).sum().item()

    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 0 if (precision + recall == 0) else 2 * precision * recall / (precision + recall)
    iou = TP / max(TP + FP + FN, 1)

    acc = (pred == target).float().mean().item()

    TN = ((1 - pred) * (1 - target)).sum().item()

    return acc, precision, recall, f1, iou, TP, FP, FN, TN

# ==============================
# TREINAMENTO
# ==============================

best_iou = -1
no_improve = 0
history = []
metrics_csv = os.path.join(os.path.dirname(checkpoint_path), "unet_metrics.csv")
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)

        optimizer.zero_grad()
        with autocast():
            pred = model(img)
            loss = loss_fn(pred, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -----------------------------
    # VALIDAÇÃO
    # -----------------------------
    model.eval()
    val_loss = 0
    TP_total = FP_total = FN_total = TN_total = 0

    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            with autocast():
                pred = model(img)
                loss = loss_fn(pred, mask)

            val_loss += loss.item()

            acc, pre, rec, f1, iou, tp, fp, fn, tn = bin_metrics(pred, mask)
            TP_total += tp
            FP_total += fp
            FN_total += fn
            TN_total += tn

    val_loss /= len(val_loader)
    pixel_count = TP_total + TN_total + FP_total + FN_total
    acc_m = (TP_total + TN_total) / max(pixel_count, 1)
    pre_m = TP_total / max(TP_total + FP_total, 1)
    rec_m = TP_total / max(TP_total + FN_total, 1)
    f1_m = 0 if pre_m + rec_m == 0 else 2 * pre_m * rec_m / (pre_m + rec_m)
    iou_m = TP_total / max(TP_total + FP_total + FN_total, 1)

    scheduler.step(val_loss)

    print(f"\n=== Época {epoch+1}/{epochs} ===")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Acc: {acc_m:.4f}  Prec: {pre_m:.4f}  Rec: {rec_m:.4f}  F1: {f1_m:.4f}  IoU: {iou_m:.4f}")
    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "accuracy": acc_m,
        "precision": pre_m,
        "recall": rec_m,
        "f1": f1_m,
        "iou": iou_m,
        "tp": TP_total,
        "fp": FP_total,
        "fn": FN_total,
        "tn": TN_total,
    })
    pd.DataFrame(history).to_csv(metrics_csv, index=False)
    print(f"CSV atualizado: {metrics_csv}")

    # Salvar o melhor modelo baseado no IoU
    if iou_m > best_iou:
        best_iou = iou_m
        no_improve = 0
        torch.save(model.state_dict(), checkpoint_path)
        print("✔ Modelo salvo (melhor IoU)")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("⛔ Early Stopping")
            break

print("\nTreinamento finalizado!")
print("Melhor IoU:", best_iou)
print("Modelo salvo em:", checkpoint_path)
