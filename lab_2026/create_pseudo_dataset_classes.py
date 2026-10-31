import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torchvision.transforms as T
from efficientnet_pytorch import EfficientNet
from segmentation_models_pytorch import Unet

# ==============================
# CONFIG
# ==============================

clf_model_path = "/home/paulomaciel/proj/species_model/efficientnet_b0_best.pth"
seg_model_path = "/home/paulomaciel/proj/unet_binary_tree_detector512_softAug2.pth"

input_dir  = "/home/paulomaciel/proj/species_crops/"
output_base = "/home/paulomaciel/proj/dataset_auto/"

img_size = 512
num_classes = 6

confidence_threshold = 0.92
mask_threshold = 0.6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# CORES DAS ESPÉCIES (MESMO PADRÃO DO SEU DATASET)
# ==============================

colors = {
    1: [128, 128, 128],
    2: [0, 255, 0],
    3: [0, 0, 255],
    4: [255, 255, 0],
    5: [0, 255, 255],
    6: [255, 0, 255],
}

# ==============================
# TRANSFORMS
# ==============================

transform = T.Compose([
    T.Resize((img_size, img_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ==============================
# MODELOS
# ==============================

clf_model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=num_classes)
clf_model.load_state_dict(torch.load(clf_model_path, map_location=device))
clf_model.to(device).eval()

seg_model = Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
seg_model.load_state_dict(torch.load(seg_model_path, map_location=device))
seg_model.to(device).eval()

print("Modelos carregados!")

# ==============================
# LISTAR IMAGENS
# ==============================

def list_images(folder):
    exts = (".png", ".jpg", ".jpeg", ".tif")
    files = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            if f.lower().endswith(exts):
                files.append(os.path.join(root, f))
    return files

image_files = list_images(input_dir)
print("Total imagens:", len(image_files))

# ==============================
# PROCESSAMENTO
# ==============================

log_data = []

for img_path in tqdm(image_files):

    try:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        # =========================
        # CLASSIFICAÇÃO
        # =========================
        with torch.no_grad():
            out_clf = clf_model(tensor)
            probs = torch.softmax(out_clf, dim=1)

        conf, pred = torch.max(probs, dim=1)

        conf = conf.item()
        pred = pred.item() + 1

        if conf < confidence_threshold:
            continue

        # =========================
        # SEGMENTAÇÃO
        # =========================
        with torch.no_grad():
            seg_out = seg_model(tensor)
            mask = torch.sigmoid(seg_out)

        mask = (mask > mask_threshold).cpu().numpy()[0, 0]

        # =========================
        # MÁSCARA COLORIDA
        # =========================
        color = np.array(colors[pred], dtype=np.uint8)

        colored_mask = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        colored_mask[mask == 1] = color

        # =========================
        # SALVAR
        # =========================
        filename = os.path.basename(img_path)

        class_dir = os.path.join(output_base, str(pred))
        img_dir = os.path.join(class_dir, "images")
        mask_dir = os.path.join(class_dir, "masks")

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        # imagem
        img_save_path = os.path.join(img_dir, filename)
        img.save(img_save_path)

        # máscara colorida
        mask_save_path = os.path.join(mask_dir, filename.replace(".tif", ".png"))
        Image.fromarray(colored_mask).save(mask_save_path)

        log_data.append({
            "image": img_save_path,
            "mask": mask_save_path,
            "species": pred,
            "confidence": conf
        })

    except Exception as e:
        print("Erro:", img_path, e)

# ==============================
# CSV
# ==============================

df = pd.DataFrame(log_data)
csv_path = os.path.join(output_base, "pseudo_full.csv")
df.to_csv(csv_path, index=False)

print("\nFinalizado!")
print("Total gerado:", len(df))
print("CSV:", csv_path)