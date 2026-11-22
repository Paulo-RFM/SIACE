import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from segmentation_models_pytorch import Unet
from torch.cuda.amp import autocast

# ==============================
# CONFIGURAÇÕES
# ==============================

images_dir = "/home/paulomaciel/proj/tiles_512/"     # diretório das imagens originais
output_dir = "/home/paulomaciel/proj/masks_bin_pred/"  # onde salvar as máscaras previstas
checkpoint = "/home/paulomaciel/proj/unet_binary_tree_detector.pth"

img_size = (512, 512)   # mesmo tamanho usado no treinamento
threshold = 0.5          # limiar para transformar sigmoid → máscara binária

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(output_dir, exist_ok=True)

# ==============================
# MODELO
# ==============================

print("Carregando modelo binário...")

model = Unet(
    encoder_name="resnet34",
    encoder_weights=None,     # não importa para inferência
    in_channels=3,
    classes=1
).to(device)

model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

print("Modelo carregado com sucesso.")

# ==============================
# TRANSFORMAÇÕES
# ==============================

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# FUNÇÃO DE PREDIÇÃO
# ==============================

@torch.no_grad()
def predict_mask(image_path):
    img = Image.open(image_path).convert("RGB")
    img_resized = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

    with autocast():
        pred = model(img_resized)  # logits
        prob = torch.sigmoid(pred)  # [1,1,H,W]

    mask = (prob.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255
    return mask

# ==============================
# PROCESSAMENTO DE TODAS AS IMAGENS
# ==============================

image_paths = sorted([p for p in Path(images_dir).glob("*") if p.suffix.lower() in [".tif", ".png", ".jpg"]])

print(f"Encontradas {len(image_paths)} imagens.")

for img_path in image_paths:
    print("Processando:", img_path.name)

    mask = predict_mask(img_path)
    mask_img = Image.fromarray(mask.astype(np.uint8))

    save_path = os.path.join(output_dir, img_path.stem + ".png")
    mask_img.save(save_path)

print("\n Finalizado!")
print("Máscaras salvas em:", output_dir)
