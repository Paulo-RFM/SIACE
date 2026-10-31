import os
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd

# ============================
# CONFIGURAÇÕES
# ============================

images_dir = "/home/paulomaciel/proj/imagens/"
instances_dir = "/home/paulomaciel/proj/masks_instances/"
species_dir = "/home/paulomaciel/proj/masks_distin/"

out_dir = "/home/paulomaciel/proj/species_crops/"
os.makedirs(out_dir, exist_ok=True)

crop_size = 512
padding = 16

color_to_species = {
    (0,0,0): 0,
    (255,0,0): 1,
    (0,255,0): 2,
    (0,0,255): 3,
    (255,255,0): 4,
    (255,0,255): 5,
    (0,255,255): 6
}

# ============================
# FUNÇÕES
# ============================

def find_corresponding_file(base, dir_path):
    """
    Encontra arquivo correspondente independentemente da extensão:
    base = "IMG_001"
    Procura base + (".png", ".jpg", ".jpeg", ".tif")
    """
    extensions = [".png", ".jpg", ".jpeg", ".tif"]
    for ext in extensions:
        p = Path(dir_path) / f"{base}{ext}"
        if p.exists():
            return p
    return None

def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

def load_species_mask(path):
    return np.array(Image.open(path).convert("RGB"))

def load_inst_mask(path):
    return np.array(Image.open(path), dtype=np.int32)

def get_species_label(seg_crop, cmap):
    pixels = seg_crop.reshape(-1,3)
    counts = {label:0 for label in cmap.values()}
    for color, label in cmap.items():
        c = np.array(color)
        counts[label] = np.sum(np.all(pixels == c, axis=1))
    return max(counts, key=counts.get)

def safe_crop(arr, y0, y1, x0, x1):
    H, W = arr.shape[:2]
    return arr[max(0,y0):min(H,y1)+1, max(0,x0):min(W,x1)+1]

# ============================
# EXECUÇÃO
# ============================

rows = []
instance_files = sorted(Path(instances_dir).glob("*_instances.png"))

print("Máscaras de instância encontradas:", len(instance_files))

for inst_path in instance_files:
    base = inst_path.stem.replace("_instances","")
    print(f"\nProcessando: {base}")

    # localizar imagem RGB real
    img_path = find_corresponding_file(base, images_dir)
    if img_path is None:
        print("❌ ERRO: imagem RGB não encontrada:", base)
        continue
    pause()
    # localizar máscara de espécie real
    species_path = find_corresponding_file(base, species_dir)
    if species_path is None:
        print("❌ ERRO: máscara de espécie não encontrada:", base)
        continue

    img = load_rgb(img_path)
    inst = load_inst_mask(inst_path)
    species_mask = load_species_mask(species_path)

    n_instances = inst.max()
    print("→ Instâncias:", n_instances)

    if n_instances == 0:
        print("⚠ Nenhuma instância encontrada, pulando...")
        continue

    for inst_id in range(1, n_instances+1):
        mask_inst = (inst == inst_id).astype(np.uint8)
        if mask_inst.sum() == 0:
            continue

        ys, xs = np.where(mask_inst)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        y0 -= padding; x0 -= padding
        y1 += padding; x1 += padding

        crop_img = safe_crop(img, y0, y1, x0, x1)
        crop_species = safe_crop(species_mask, y0, y1, x0, x1)

        crop_img_resized = np.array(Image.fromarray(crop_img).resize((crop_size, crop_size), Image.BILINEAR))
        crop_species_resized = np.array(Image.fromarray(crop_species).resize((crop_size, crop_size), Image.NEAREST))

        label = get_species_label(crop_species_resized, color_to_species)

        crop_name = f"{base}_tree_{inst_id:04d}.png"
        crop_path = str(Path(out_dir) / crop_name)
        Image.fromarray(crop_img_resized).save(crop_path)

        rows.append({
            "image": base,
            "instance_id": inst_id,
            "crop_path": crop_path,
            "label": label
        })

# salvar CSV
csv_path = Path(out_dir) / "species_labels.csv"
pd.DataFrame(rows).to_csv(csv_path, index=False)

print("\n✔ Finalizado!")
print("Total de crops:", len(rows))
print("CSV salvo em:", csv_path)
