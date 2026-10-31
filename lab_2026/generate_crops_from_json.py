import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd

# ======================================
# CONFIGURAÇÕES
# ======================================

base = "/home/paulomaciel/proj/"

images_dir = base + "imagens/"
labels_dir = base + "labels/"
output_dir = base + "species_crops/"

os.makedirs(output_dir, exist_ok=True)

CROP_SIZE = 512
PADDING = 20   # pixels para ampliar bounding box


# ======================================
# FUNÇÕES
# ======================================

def load_image(path):
    return np.array(Image.open(path).convert("RGB"))

def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def safe_crop(img, x0, y0, x1, y1):
    H, W = img.shape[:2]
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(W-1, int(x1))
    y1 = min(H-1, int(y1))
    return img[y0:y1+1, x0:x1+1]

# ======================================
# EXECUÇÃO
# ======================================

rows = []

json_files = sorted(Path(labels_dir).glob("*.json"))
print("Arquivos JSON encontrados:", len(json_files))

for json_path in json_files:
    basename = json_path.stem   # ex: OrtoRP-3cm_005_020
    print("Processando:", basename)

    image_path = Path(images_dir) / (basename + ".tif")
    if not image_path.exists():
        print("Imagem não encontrada:", image_path)
        continue

    img = load_image(image_path)

    with open(json_path, 'r') as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    print(" → Árvores detectadas no JSON:", len(shapes))

    for idx, shape in enumerate(shapes, start=1):
        if shape.get("label") != "tree":
            continue

        species = shape.get("group_id", None)
        if species is None:
            continue

        points = shape.get("points", [])
        if len(points) < 3:
            continue

        # bounding box
        x0, y0, x1, y1 = polygon_to_bbox(points)

        # aplicar padding
        x0 -= PADDING
        y0 -= PADDING
        x1 += PADDING
        y1 += PADDING

        # recortar imagem
        crop = safe_crop(img, x0, y0, x1, y1)
        crop = np.array(Image.fromarray(crop).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR))

        # nome do arquivo
        crop_name = f"{basename}_tree_{idx:04d}.png"
        crop_path = str(Path(output_dir) / crop_name)

        Image.fromarray(crop).save(crop_path)

        rows.append({
            "image": basename,
            "tree_id": idx,
            "crop": crop_path,
            "species": species
        })

# salvar CSV
csv_path = Path(output_dir) / "species_labels.csv"
pd.DataFrame(rows).to_csv(csv_path, index=False)

print("\n==== FINALIZADO ====")
print("Total de árvores processadas:", len(rows))
print("CSV salvo em:", csv_path)
