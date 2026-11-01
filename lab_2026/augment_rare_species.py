import os
import random
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance

# ======================================
# CONFIGURAÇÕES
# ======================================

csv_path   = "/home/paulomaciel/proj/species_crops/species_labels.csv"
input_dir  = "/home/paulomaciel/proj/species_crops/"
output_dir = "/home/paulomaciel/proj/species_crops_balanced/"

os.makedirs(output_dir, exist_ok=True)

# quantidade desejada por classe APÓS balanceamento
target_counts = {
    1: 1200,
    2: 800,
    3: 800,
    4: 800,
    5: 800,
    6: 800
}

# ======================================
# AUGMENTATIONS
# ======================================

def augment(img):
    """Aplica uma augmentation aleatória."""
    choice = random.choice(["flip_h", "flip_v", "rotate", "color", "crop"])

    if choice == "flip_h":
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    elif choice == "flip_v":
        return img.transpose(Image.FLIP_TOP_BOTTOM)

    elif choice == "rotate":
        angle = random.choice([90, 180, 270])
        return img.rotate(angle, expand=False)

    elif choice == "color":
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(random.uniform(0.5, 1.5))

    elif choice == "crop":
        w, h = img.size
        pad = int(min(w, h) * 0.05)
        img2 = img.crop((pad, pad, w - pad, h - pad))
        return img2.resize((w, h), Image.BILINEAR)

    return img

# ======================================
# EXECUÇÃO
# ======================================

df = pd.read_csv(csv_path)
new_rows = []

print("\nAumentando classes raras...\n")

for species, target in target_counts.items():

    subset = df[df["species"] == species]
    current = len(subset)

    print(f"Classe {species}: {current} → {target}")

    # copiar originais — sempre como dict
    for _, row in subset.iterrows():
        original_row = row.to_dict()
        original_row.setdefault("origin", "manual")
        original_row.setdefault("source_origin", original_row["origin"])
        new_rows.append(original_row)

    # já suficiente
    if current >= target:
        continue

    needed = target - current

    for i in range(needed):
        row = subset.sample(1).iloc[0]
        img_path = row["crop"]

        if not os.path.exists(img_path):
            print("⚠ crop não encontrado:", img_path)
            continue

        img = Image.open(img_path).convert("RGB")
        aug_img = augment(img)

        # novo nome único
        new_name = f"{row['image']}_aug_{row['tree_id']}_{i}.png"
        new_path = os.path.join(output_dir, new_name)
        aug_img.save(new_path)

        # linha nova como dict
        new_rows.append({
            "image": row["image"],
            "tree_id": row["tree_id"],
            "crop": new_path,
            "species": species,
            "origin": "aumentado",
            "source_origin": row.get("source_origin", row.get("origin", "manual")),
        })

# salvar novo CSV
out_csv = os.path.join(output_dir, "species_labels_balanced.csv")
pd.DataFrame(new_rows).to_csv(out_csv, index=False)

print("\n✔ Balanceamento concluído!")
print("Novo dataset salvo em:", output_dir)
print("Novo CSV:", out_csv)
