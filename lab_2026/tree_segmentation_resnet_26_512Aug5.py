import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.losses import DiceLoss

from torch.cuda.amp import autocast, GradScaler

# ==============================
# CONFIG
# ==============================

base_image_tiles_dir = "/home/paulomaciel/proj/imagens/"
base_mask_tiles_dir  = "/home/paulomaciel/proj/masks/"

pseudo_img_dir = "/home/paulomaciel/proj/pseudo_dataset/images"
pseudo_mask_dir = "/home/paulomaciel/proj/pseudo_dataset/masks"

checkpoint_path = "/home/paulomaciel/proj/lab_2026/unet_binary_tree_detector512_softAug2.pth"

img_size = (512,512)
batch_size = 6
epochs = 10
patience = 8
pseudo_ratio = 0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# UTILS
# ==============================

def list_all_files(directory, extension):
    result = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(extension):
                result.append(os.path.join(root, f))
    return sorted(result)

def build_pairs(image_files, mask_files):
    img_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
    mask_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in mask_files}

    common = sorted(set(img_dict.keys()) & set(mask_dict.keys()))

    images = [img_dict[k] for k in common]
    masks  = [mask_dict[k] for k in common]

    return images, masks

def rgb_to_binary(mask_rgb):
    mask = np.any(mask_rgb != [0, 0, 0], axis=-1)
    return mask.astype(np.uint8)

# ==============================
# DATASET
# ==============================

class TreeBinaryDataset(Dataset):
    def __init__(self, image_files, mask_files, img_size, train=True):
        self.image_files = list(image_files)
        self.mask_files = list(mask_files)
        self.img_size = img_size
        self.train = train

        self.crop = transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0))
        self.color_jitter = transforms.ColorJitter(0.1, 0.1, 0.1)
        self.blur = transforms.GaussianBlur(kernel_size=3)

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

        if self.train:
            seed = np.random.randint(0, 100000)

            random.seed(seed)
            img = self.crop(img)
            random.seed(seed)
            mask = self.crop(mask)

            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])
                img = img.rotate(angle)
                mask = mask.rotate(angle)

            img = self.color_jitter(img)

            if random.random() < 0.3:
                img = self.blur(img)

        else:
            img = img.resize(self.img_size, Image.BILINEAR)
            mask = mask.resize(self.img_size, Image.NEAREST)

        mask_np = rgb_to_binary(np.array(mask))
        mask_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)

        img = self.to_tensor(img)
        img = self.normalize(img)

        return img, mask_tensor

# ==============================
# LOAD DATA
# ==============================

image_files = list_all_files(base_image_tiles_dir, '.tif')
mask_files  = list_all_files(base_mask_tiles_dir , '.png')
image_files, mask_files = build_pairs(image_files, mask_files)

pseudo_images = list_all_files(pseudo_img_dir, '.tif')
pseudo_masks  = list_all_files(pseudo_mask_dir, '.png')
pseudo_images, pseudo_masks = build_pairs(pseudo_images, pseudo_masks)

print("Real:", len(image_files))
print("Pseudo:", len(pseudo_images))

# split real
image_train, image_val, mask_train, mask_val = train_test_split(
    image_files, mask_files, test_size=0.3, random_state=42
)

# ==============================
# MIX REAL + PSEUDO
# ==============================

mixed_images = list(image_train)
mixed_masks  = list(mask_train)

if len(pseudo_images) > 0:
    n_real = len(image_train)
    n_pseudo = int(n_real * pseudo_ratio)

    n_pseudo = min(n_pseudo, len(pseudo_images))

    pseudo_idx = np.random.choice(len(pseudo_images), n_pseudo, replace=False)

    for i in pseudo_idx:
        mixed_images.append(pseudo_images[i])
        mixed_masks.append(pseudo_masks[i])

print(f"Treino total: {len(mixed_images)}")

combined = list(zip(mixed_images, mixed_masks))
random.shuffle(combined)
mixed_images, mixed_masks = zip(*combined)

# ==============================
# DATALOADER
# ==============================

train_dataset = TreeBinaryDataset(mixed_images, mixed_masks, img_size, train=True)
val_dataset   = TreeBinaryDataset(image_val, mask_val, img_size, train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ==============================
# MODEL
# ==============================

model = Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(device)

# carregar checkpoint se existir
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print("Checkpoint carregado")

bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2]).to(device))
dice = DiceLoss(mode="binary", from_logits=True, smooth=1.0)

def loss_fn(pred, target):
    return 0.4 * bce(pred, target) + 0.6 * dice(pred, target)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

scaler = GradScaler(enabled=(device.type == "cuda"))

# ==============================
# TRAIN
# ==============================

best_iou = -1
no_improve = 0
history = []

for epoch in range(epochs):

    model.train()
    train_loss = 0

    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)

        optimizer.zero_grad()

        with autocast(enabled=(device.type == "cuda")):
            pred = model(img)
            loss = loss_fn(pred, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ===== VALIDATION =====
    model.eval()

    TP = FP = FN = TN = 0

    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)

            pred = torch.sigmoid(model(img))
            pred = (pred > 0.5).float()

            TP += ((pred == 1) & (mask == 1)).sum().item()
            FP += ((pred == 1) & (mask == 0)).sum().item()
            FN += ((pred == 0) & (mask == 1)).sum().item()
            TN += ((pred == 0) & (mask == 0)).sum().item()

    iou = TP / max(TP + FP + FN, 1)
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    accuracy = (TP + TN) / max(TP + TN + FP + FN, 1)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"Accuracy: {accuracy:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": TP,
        "fp": FP,
        "fn": FN,
        "tn": TN,
        "threshold": 0.5,
    })
    pd.DataFrame(history).to_csv(
        os.path.join(os.path.dirname(checkpoint_path), "unet_metrics_512_softAug2.csv"),
        index=False,
    )

    if iou > best_iou:
        best_iou = iou
        no_improve = 0
        torch.save(model.state_dict(), checkpoint_path)
        print("✔ Melhor modelo salvo")
    else:
        no_improve += 1
        print(f"Sem melhora: {no_improve}/{patience}")

        if no_improve >= patience:
            print("🛑 Early stopping")
            break