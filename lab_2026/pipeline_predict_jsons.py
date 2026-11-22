#!/usr/bin/env python3
"""
Pipeline: gera 1 JSON estilo LabelMe por tile.
- Polígonos são extraídos com cv2.findContours (compatível com LabelMe)
- group_id = espécie predita (1..6)
- confidence incluída
- NÃO usa rasterio nem GDAL
"""

import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import torch.nn.functional as F

from skimage import morphology, segmentation, feature, measure
from scipy import ndimage as ndi
import cv2

from segmentation_models_pytorch import Unet
from efficientnet_pytorch import EfficientNet

# ---------------------------
# CONFIGURAÇÕES
# ---------------------------
BASE = "/home/paulomaciel/proj/"
tiles_dir = BASE + "tiles_512/"
out_json_dir = BASE + "pred_jsons/"
os.makedirs(out_json_dir, exist_ok=True)

seg_model_path = BASE + "unet_binary_tree_detector.pth"
cls_model_path = BASE + "species_model/efficientnet_b0_best.pth"

SEG_IMG_SIZE = (512, 512)
CLS_IMG_SIZE = 512
MASK_THRESHOLD = 0.25
MIN_OBJ_PX = 8
ERODE_KERNEL = (3, 3)
PEAK_FOOTPRINT = (7, 7)
MIN_INSTANCE_PX = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------
# LOAD MODELS
# ---------------------------
def safe_load_state(model, path):
    state = torch.load(path, map_location=device)
    try:
        model.load_state_dict(state)
        print(f"Loaded {path} (strict=True)")
    except:
        model.load_state_dict(state, strict=False)
        print(f"Loaded {path} (strict=False fallback)")

# UNet
seg_model = Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)
safe_load_state(seg_model, seg_model_path)
seg_model.eval()

# EfficientNet
NUM_CLASSES = 6
cls_model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=NUM_CLASSES).to(device)
safe_load_state(cls_model, cls_model_path)
cls_model.eval()

softmax = torch.nn.Softmax(dim=1)

# transforms
seg_transform = transforms.Compose([
    transforms.Resize(SEG_IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
cls_transform = transforms.Compose([
    transforms.Resize((CLS_IMG_SIZE, CLS_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ---------------------------
# Função para extrair polígonos via OpenCV
# ---------------------------
def mask_to_polygons_cv(mask):
    """
    Converte máscara binária em lista de polígonos (cada polígono = lista de [x,y]).
    Igual ao que LabelMe usa.

    mask: uint8 (0/1)
    """
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        poly = []
        for p in cnt:
            x, y = p[0]
            poly.append([float(x), float(y)])
        polygons.append(poly)

    return polygons

# ---------------------------
# PIPELINE POR TILE
# ---------------------------
tile_paths = sorted(Path(tiles_dir).glob("*.tif"))
print("Found tiles:", len(tile_paths))

for tile_path in tile_paths:
    base = tile_path.stem
    print(f"\nProcessing tile: {base}")

    img = np.array(Image.open(tile_path).convert("RGB"))
    H, W = img.shape[:2]

    # --- 1) Segmentation ---
    inp = seg_transform(Image.fromarray(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        prob_small = torch.sigmoid(seg_model(inp)).squeeze().cpu().numpy()

    prob = cv2.resize(prob_small, (W, H), interpolation=cv2.INTER_LINEAR)
    mask_bin = (prob > MASK_THRESHOLD).astype(np.uint8)

    # limpeza
    clean = morphology.remove_small_objects(mask_bin.astype(bool), min_size=MIN_OBJ_PX)
    clean = morphology.remove_small_holes(clean, area_threshold=MIN_OBJ_PX*2)
    clean = clean.astype(np.uint8)

    # tile vazio
    if clean.sum() == 0:
        json_obj = {
            "version": "5.2.1",
            "flags": {},
            "shapes": [],
            "imagePath": f"{base}.tif",
            "imageHeight": H,
            "imageWidth": W
        }
        out_path = os.path.join(out_json_dir, f"{base}_pred.json")
        with open(out_path, "w") as f:
            json.dump(json_obj, f, indent=2)
        print(f" -> empty tile, saved {out_path}")
        continue

    clean = cv2.erode(clean, np.ones(ERODE_KERNEL, np.uint8), iterations=1)

    # --- 2) Watershed ---
    dist = ndi.distance_transform_edt(clean)
    peak_mask = feature.peak_local_max(dist, footprint=np.ones(PEAK_FOOTPRINT), labels=clean)
    coords = np.argwhere(peak_mask)

    if coords.shape[0] == 0:
        markers, _ = ndi.label(clean)
    else:
        markers = np.zeros_like(dist, dtype=np.int32)
        for i, (r, c) in enumerate(coords, 1):
            markers[r, c] = i
        markers, _ = ndi.label(markers)

    inst = segmentation.watershed(-dist, markers, mask=clean)

    # remover instâncias pequenas
    inst_mask_filtered = np.zeros_like(inst, dtype=np.int32)
    new_id = 1
    for p in measure.regionprops(inst):
        if p.area >= MIN_INSTANCE_PX:
            inst_mask_filtered[inst == p.label] = new_id
            new_id += 1

    n_instances = inst_mask_filtered.max()
    print(f" -> instances: {n_instances}")

    # --- 3) Polígonos + Classificação ---
    shapes_list = []

    for inst_id in range(1, n_instances+1):
        m = (inst_mask_filtered == inst_id).astype(np.uint8)
        if m.sum() == 0:
            continue

        # polígonos via opencv
        polys = mask_to_polygons_cv(m)
        if not polys:
            continue

        ys, xs = np.where(m)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        crop = img[y0:y1+1, x0:x1+1]
        crop_pil = Image.fromarray(crop)
        inp_cls = cls_transform(crop_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            out = cls_model(inp_cls)
            probv = softmax(out).cpu().numpy()[0]
            pred = int(np.argmax(probv)) + 1
            conf = float(probv.max())

        for poly in polys:
            shapes_list.append({
                "label": "tree",
                "points": poly,
                "group_id": pred,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "confidence": conf
            })

    # --- 4) Salvar JSON ---
    json_obj = {
        "version": "5.2.1",
        "flags": {},
        "shapes": shapes_list,
        "imagePath": f"{base}.tif",
        "imageHeight": H,
        "imageWidth": W,
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }

    out_path = os.path.join(out_json_dir, f"{base}_pred.json")
    with open(out_path, "w") as f:
        json.dump(json_obj, f, indent=2)

    print(" -> saved JSON:", out_path)

print("\nAll tiles processed.")
print("JSONs saved in:", out_json_dir)
