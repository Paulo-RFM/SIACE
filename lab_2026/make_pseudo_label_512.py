import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from segmentation_models_pytorch import Unet
from torch.cuda.amp import autocast

# ==============================
# CONFIG
# ==============================

input_dir  = "/home/paulomaciel/proj/novas_imagens/"
output_img_dir  = "/home/paulomaciel/proj/pseudo_dataset/images/"
output_mask_dir = "/home/paulomaciel/proj/pseudo_dataset/masks/"
model_path = "/home/paulomaciel/proj/unet_binary_tree_detector512_semAug.pth"

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

batch_size = 8
img_size = (512, 512)

threshold = 0.8
min_area_ratio = 0.01
max_area_ratio = 0.6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

# ==============================
# DATASET
# ==============================

class InferenceDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, path


# ==============================
# LOAD MODEL
# ==============================

model = Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# ==============================
# LOAD IMAGES
# ==============================

image_paths = [
    os.path.join(input_dir, f)
    for f in os.listdir(input_dir)
    if f.endswith(".tif") or f.endswith(".png")
]

dataset = InferenceDataset(image_paths)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ==============================
# INFERÊNCIA
# ==============================

kept = 0
discarded = 0

with torch.no_grad():
    for imgs, paths in tqdm(loader):

        imgs = imgs.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            preds = model(imgs)
            probs = torch.sigmoid(preds)

        probs = probs.cpu().numpy()

        for i in range(len(paths)):
            prob = probs[i, 0]

            mask_bin = (prob > threshold).astype(np.uint8)
            area_ratio = mask_bin.mean()

            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                discarded += 1
                continue

            filename = os.path.basename(paths[i])

            # ✅ RECARREGA imagem original (corrigido)
            original_img = Image.open(paths[i]).convert("RGB")
            original_img.save(os.path.join(output_img_dir, filename))

            # máscara
            mask_img = (mask_bin * 255).astype(np.uint8)
            Image.fromarray(mask_img).save(
                os.path.join(output_mask_dir, filename.replace(".tif", ".png"))
            )

            kept += 1

print("\nFinalizado!")
print("Mantidas:", kept)
print("Descartadas:", discarded)
