import os
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from segmentation_models_pytorch import UnetPlusPlus
from segmentation_models_pytorch.losses import DiceLoss

from sklearn.model_selection import train_test_split

# ==============================
# CONFIG
# ==============================

base_image_tiles_dir = "/home/paulomaciel/proj/imagens/"
base_mask_tiles_dir  = "/home/paulomaciel/proj/masks/"

csv_summary = "/home/paulomaciel/proj/lab_2026/benchmark_unetpp_summary.csv"
csv_history = "/home/paulomaciel/proj/lab_2026/benchmark_unetpp_history.csv"

img_size = (512,512)
batch_size = 6
epochs = 30
patience = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoders = [
    "mobilenet_v2",
    "resnet34",
    "efficientnet-b0"
]

# ==============================
# UTILS
# ==============================

def list_all_files(directory, ext):
    files = []
    for root, _, filenames in os.walk(directory):
        for f in filenames:
            if f.lower().endswith(ext):
                files.append(os.path.join(root, f))
    return sorted(files)

def rgb_to_binary(mask_rgb):
    return np.any(mask_rgb != [0,0,0], axis=-1).astype(np.uint8)

# ==============================
# DATASET
# ==============================

class TreeDataset(Dataset):
    def __init__(self, imgs, masks, train=True):
        self.imgs = imgs
        self.masks = masks
        self.train = train

        self.crop = transforms.RandomResizedCrop(img_size, scale=(0.8,1.0))
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = Image.open(self.imgs[i]).convert("RGB")
        mask = Image.open(self.masks[i]).convert("RGB")

        if self.train:
            seed = np.random.randint(99999)

            random.seed(seed)
            img = self.crop(img)

            random.seed(seed)
            mask = self.crop(mask)

            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        else:
            img = img.resize(img_size)
            mask = mask.resize(img_size)

        mask = rgb_to_binary(np.array(mask))
        mask = torch.tensor(mask).float().unsqueeze(0)

        img = self.to_tensor(img)
        img = self.norm(img)

        return img, mask

# ==============================
# METRICS
# ==============================

def compute_metrics(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    TP = (pred * target).sum().item()
    FP = (pred * (1-target)).sum().item()
    FN = ((1-pred) * target).sum().item()
    TN = ((1-pred)*(1-target)).sum().item()

    iou = TP / max(TP+FP+FN,1)
    precision = TP / max(TP+FP,1)
    recall = TP / max(TP+FN,1)
    f1 = 2*precision*recall/max(precision+recall,1)

    return iou, precision, recall, f1

# ==============================
# LOAD DATA
# ==============================

images = list_all_files(base_image_tiles_dir, ".tif")
masks  = list_all_files(base_mask_tiles_dir , ".png")

train_img, val_img, train_mask, val_mask = train_test_split(
    images, masks, test_size=0.3, random_state=42
)

train_loader = DataLoader(TreeDataset(train_img, train_mask, True),
                          batch_size=batch_size, shuffle=True, num_workers=4)

val_loader   = DataLoader(TreeDataset(val_img, val_mask, False),
                          batch_size=batch_size, shuffle=False, num_workers=4)

# ==============================
# LOSS
# ==============================

bce = nn.BCEWithLogitsLoss()
dice = DiceLoss(mode="binary", from_logits=True)

def loss_fn(pred, target):
    return 0.5*bce(pred,target) + 0.5*dice(pred,target)

# ==============================
# LISTAS DE RESULTADO
# ==============================

summary_results = []
history = []

# ==============================
# BENCHMARK
# ==============================

for encoder in encoders:

    print(f"\n============================")
    print(f"TESTANDO: {encoder}")
    print(f"============================")

    model = UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    best_iou = 0
    no_improve = 0

    for epoch in range(epochs):

        start_time = time.time()

        # ===== TREINO =====
        model.train()
        train_loss = 0

        loop = tqdm(train_loader, desc=f"{encoder} | Epoch {epoch+1}")

        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            pred = model(imgs)
            loss = loss_fn(pred, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        # ===== VAL =====
        model.eval()

        iou_total = prec_total = rec_total = f1_total = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)

                pred = model(imgs)
                iou, p, r, f1 = compute_metrics(pred, masks)

                iou_total += iou
                prec_total += p
                rec_total += r
                f1_total += f1

        n = len(val_loader)
        iou = iou_total / n
        precision = prec_total / n
        recall = rec_total / n
        f1 = f1_total / n

        elapsed = time.time() - start_time

        print(f"\nIoU: {iou:.4f} | F1: {f1:.4f} | Tempo: {elapsed:.1f}s")

        # ===== HISTÓRICO =====
        history.append({
            "encoder": encoder,
            "epoch": epoch+1,
            "train_loss": train_loss,
            "iou": iou,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "time_sec": elapsed
        })

        pd.DataFrame(history).to_csv(csv_history, index=False)

        # ===== MELHOR MODELO =====
        if iou > best_iou:
            best_iou = iou
            no_improve = 0

            torch.save(model.state_dict(),
                       f"/home/paulomaciel/proj/lab_2026/unetpp_{encoder}_best.pth")

            print("✔ Melhor modelo salvo")

        else:
            no_improve += 1

            if no_improve >= patience:
                print("🛑 Early stopping")
                break

    # ===== RESUMO POR ENCODER =====
    summary_results.append({
        "encoder": encoder,
        "best_iou": best_iou,
        "final_f1": f1,
        "epochs_ran": epoch+1
    })

    pd.DataFrame(summary_results).to_csv(csv_summary, index=False)

# ==============================
# FINAL
# ==============================

print("\nBenchmark finalizado!")
print(pd.DataFrame(summary_results))
