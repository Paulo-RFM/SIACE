import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage import morphology, segmentation, feature, measure
from scipy import ndimage as ndi

from segmentation_models_pytorch import Unet
from efficientnet_pytorch import EfficientNet

# ======================================================
# CONFIGURAÇÕES
# ======================================================

tiles_dir = "/home/paulomaciel/proj/tiles_512/"
out_dir   = "/home/paulomaciel/proj/simple_predictions/"
os.makedirs(out_dir, exist_ok=True)

seg_model_path = "/home/paulomaciel/proj/unet_binary_tree_detector.pth"
cls_model_path = "/home/paulomaciel/proj/species_model/efficientnet_b0_best.pth"

seg_img_size = (512, 512)
cls_img_size = 512

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

MASK_THRESHOLD = 0.25
MIN_OBJ = 8
PEAK_FOOT = (7, 7)

# ======================================================
# CORES POR ESPÉCIE
# ======================================================

color_map = {
    0: (0, 0, 0),          # fundo
    1: (255, 0, 0),        # espécie 1
    2: (0, 255, 0),        # espécie 2
    3: (0, 0, 255),        # espécie 3
    4: (255, 255, 0),      # espécie 4
    5: (255, 0, 255),      # espécie 5
    6: (0, 255, 255)       # espécie 6
}

# ======================================================
# MODELOS
# ======================================================

# UNet binário
seg_model = Unet(
    encoder_name="resnet34", 
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)
seg_model.load_state_dict(torch.load(seg_model_path, map_location=device))
seg_model.eval()

# EfficientNet
num_classes = 6
cls_model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=num_classes).to(device)
cls_model.load_state_dict(torch.load(cls_model_path, map_location=device))
cls_model.eval()

# ======================================================
# TRANSFORMS
# ======================================================

seg_transform = transforms.Compose([
    transforms.Resize(seg_img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

cls_transform = transforms.Compose([
    transforms.Resize((cls_img_size, cls_img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

softmax = torch.nn.Softmax(dim=1)

# ======================================================
# FUNÇÃO PRINCIPAL
# ======================================================

def process_tile(path):
    base = path.stem
    img = np.array(Image.open(path).convert("RGB"))

    H, W = img.shape[:2]

    # -----------------------------------------------
    # 1) Segmentação binária
    # -----------------------------------------------
    inp = seg_transform(Image.fromarray(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        pr = torch.sigmoid(seg_model(inp)).squeeze().cpu().numpy()

    pr = cv2.resize(pr, (W, H))
    mask_bin = (pr > MASK_THRESHOLD).astype(np.uint8)


    # -----------------------------------------------
    # 2) Watershed — versão corrigida
    # -----------------------------------------------
    

    # limpar
    clean = morphology.remove_small_objects(mask_bin.astype(bool), MIN_OBJ).astype(np.uint8)

    # erosão leve para separar copas grudadas
    clean = cv2.erode(clean, np.ones((3,3), np.uint8), iterations=1)

    dist = ndi.distance_transform_edt(clean)

    # máscara de picos
    peak_mask = feature.peak_local_max(
        dist,
        footprint=np.ones(PEAK_FOOT),
        labels=clean
    )

    # coordenadas dos picos
    coords = np.argwhere(peak_mask)

    # marcadores
    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i

    # fallback caso markers = 0
    if np.max(markers) == 0:
        labeled, _ = ndi.label(clean)
    else:
        marked = ndi.label(markers)[0]
        labeled = segmentation.watershed(-dist, marked, mask=clean)

    inst = labeled

    # -----------------------------------------------
    # 3) Classificar cada copa
    # -----------------------------------------------
    mask_species = np.zeros((H, W), dtype=np.uint8)

    n_inst = inst.max()
    for i in range(1, n_inst+1):
        region = (inst == i)
        ys, xs = np.where(region)

        if len(ys) < 20:
            continue

        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        crop = img[y0:y1+1, x0:x1+1]
        crop_tensor = cls_transform(Image.fromarray(crop)).unsqueeze(0).to(device)

        with torch.no_grad():
            out = cls_model(crop_tensor)
            probs = softmax(out).cpu().numpy()[0]
            pred = int(np.argmax(probs)) + 1  # converte 0..5 → 1..6

        mask_species[region] = pred

    # ------------------------------
    # SALVAR MÁSCARA NUMÉRICA
    # ------------------------------
    Image.fromarray(mask_species.astype(np.uint8)).save(
        f"{out_dir}/{base}_mask_species.png"
    )

    # ------------------------------
    # SALVAR MÁSCARA COLORIDA
    # ------------------------------
    mask_color = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_value, color in color_map.items():
        mask_color[mask_species == cls_value] = color

    Image.fromarray(mask_color).save(
        f"{out_dir}/{base}_mask_species_color.png"
    )

    print("Tile processed:", base)

# ======================================================
# LOOP
# ======================================================

tiles = sorted(Path(tiles_dir).glob("*.tif"))

for t in tiles:
    process_tile(t)

print("\nFinalizado! Máscaras salvas em:", out_dir)
