import os
import random
import time
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.losses import DiceLoss

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================

base_image_tiles_dir = "/home/paulomaciel/proj/imagens/"
base_mask_tiles_dir  = "/home/paulomaciel/proj/masks/"

output_csv = "/home/paulomaciel/proj/benchmark_unet.csv"
model_dir  = "/home/paulomaciel/proj/models_benchmark/"
os.makedirs(model_dir, exist_ok=True)

img_size = (512,512)
batch_size = 6
epochs = 20
patience = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# GRID
# ==============================

encoders = ["resnet34", "resnet50", "mobilenet_v2", "efficientnet-b0"]
lrs = [1e-4, 5e-5]

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
    return [img_dict[k] for k in common], [mask_dict[k] for k in common]

def rgb_to_binary(mask_rgb):
    return np.any(mask_rgb != [0,0,0], axis=-1).astype(np.uint8)

def get_gpu_usage():
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**3
        max_mem = torch.cuda.max_memory_allocated() / 1024**3
        return f"{mem:.2f}GB / {max_mem:.2f}GB"
    return "CPU"

# ==============================
# DATASET
# ==============================

class TreeBinaryDataset(Dataset):
    def __init__(self, image_files, mask_files, img_size, train=True):
        self.image_files = image_files
        self.mask_files = mask_files
        self.img_size = img_size
        self.train = train

        self.crop = transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        mask = Image.open(self.mask_files[idx]).convert("RGB")

        if self.train:
            seed = np.random.randint(0,10000)
            random.seed(seed)
            img = self.crop(img)
            random.seed(seed)
            mask = self.crop(mask)
        else:
            img = img.resize(self.img_size)
            mask = mask.resize(self.img_size)

        mask = rgb_to_binary(np.array(mask))
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        img = self.normalize(self.to_tensor(img))
        return img, mask

# ==============================
# DADOS
# ==============================

images = list_all_files(base_image_tiles_dir, ".tif")
masks  = list_all_files(base_mask_tiles_dir , ".png")

images, masks = build_pairs(images, masks)

img_train, img_val, mask_train, mask_val = train_test_split(
    images, masks, test_size=0.3, random_state=42
)

val_dataset = TreeBinaryDataset(img_val, mask_val, img_size, train=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ==============================
# BENCHMARK
# ==============================

results = []

for encoder in encoders:
    for lr in lrs:

        print(f"\n🚀 EXPERIMENTO: {encoder} | LR={lr}")

        train_dataset = TreeBinaryDataset(img_train, mask_train, img_size, train=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        model = Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        ).to(device)

        bce = nn.BCEWithLogitsLoss()
        dice = DiceLoss(mode="binary", from_logits=True)

        def loss_fn(pred, target):
            return 0.5*bce(pred,target) + 0.5*dice(pred,target)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scaler = GradScaler()

        best_iou = 0
        best_threshold = 0
        no_improve = 0

        thresholds = [0.4,0.5,0.6,0.7]

        for epoch in range(epochs):

            start_time = time.time()

            # ================= TREINO =================
            model.train()
            train_loss = 0

            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

            for img, mask in loop:
                img, mask = img.to(device), mask.to(device)

                optimizer.zero_grad()

                with autocast(enabled=(device.type=="cuda")):
                    pred = model(img)
                    loss = loss_fn(pred, mask)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            train_loss /= len(train_loader)

            # ================= VAL =================
            model.eval()

            best_iou_epoch = 0
            best_t_epoch = 0

            with torch.no_grad():
                for t in thresholds:

                    TP=FP=FN=0

                    for img, mask in val_loader:
                        img, mask = img.to(device), mask.to(device)

                        pred = torch.sigmoid(model(img))
                        pred = (pred > t).float()

                        TP += ((pred==1)&(mask==1)).sum().item()
                        FP += ((pred==1)&(mask==0)).sum().item()
                        FN += ((pred==0)&(mask==1)).sum().item()

                    iou = TP / max(TP+FP+FN,1)

                    if iou > best_iou_epoch:
                        best_iou_epoch = iou
                        best_t_epoch = t

            # ================= COMPARAÇÃO =================
            if best_iou_epoch > best_iou:
                best_iou = best_iou_epoch
                best_threshold = best_t_epoch
                no_improve = 0

                model_path = os.path.join(
                    model_dir,
                    f"{encoder}_lr{lr:.0e}_iou{best_iou:.3f}.pth"
                )

                torch.save(model.state_dict(), model_path)
                print(f"✔ Novo melhor modelo salvo: {model_path}")

            else:
                no_improve += 1

            epoch_time = time.time() - start_time

            print(f"\nEpoch {epoch+1}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Best IoU: {best_iou:.4f} (t={best_threshold})")
            print(f"Tempo: {epoch_time:.1f}s")
            print(f"GPU: {get_gpu_usage()}")
            print(f"Sem melhora: {no_improve}/{patience}")

            if no_improve >= patience:
                print("🛑 Early stopping")
                break

        # ================= RESULTADO FINAL =================
        results.append({
            "encoder": encoder,
            "lr": lr,
            "best_iou": best_iou,
            "best_threshold": best_threshold
        })

        pd.DataFrame(results).to_csv(output_csv, index=False)

print("\n✅ Benchmark finalizado!")
