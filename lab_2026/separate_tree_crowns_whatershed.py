"""
whatershed
Gera uma máscara por instância (copa individual) a partir de máscaras binárias (árvore vs fundo).
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image

import cv2
from skimage import morphology, segmentation, feature, measure
from scipy import ndimage as ndi

# ==============================
# CONFIGURAÇÕES
# ==============================

# diretório das máscaras binárias geradas pelo modelo
binary_masks_dir = "/home/paulomaciel/proj/masks_bin_pred/"

# onde serão salvas as máscaras de instância
output_dir = "/home/paulomaciel/proj/masks_instances/"
os.makedirs(output_dir, exist_ok=True)

# parâmetros principais
min_area = 150        # remove copas muito pequenas (ajuste se necessário)
peak_footprint = 15   # tamanho da área para procurar picos (centroides)
close_kernel = 5      # fecho morfológico

# ==============================
# FUNÇÕES
# ==============================

def load_binary_mask(path):
    """Carrega máscara binária (0/255) e retorna array 0/1"""
    m = Image.open(path).convert("L")
    arr = np.array(m)
    arr = (arr > 127).astype(np.uint8)
    return arr

def segment_tree_crowns(binary_mask):
    """
    Separa copas individuais usando:
    - closing morfológico
    - distance transform
    - peak_local_max
    - watershed
    """

    # 1. MORPHOLOGY CLEANUP
    kernel = morphology.disk(close_kernel)
    clean = morphology.closing(binary_mask, kernel)
    clean = morphology.remove_small_holes(clean.astype(bool), area_threshold=32)
    clean = morphology.remove_small_objects(clean.astype(bool), min_size=32)
    clean = clean.astype(np.uint8)

    if clean.sum() == 0:
        return np.zeros_like(clean, dtype=np.int32), 0

    # 2. DISTANCE TRANSFORM
    dist = ndi.distance_transform_edt(clean)

    # 3. PICOS → CENTROIDES
    coords = feature.peak_local_max(
        dist,
        indices=True,
        footprint=np.ones((peak_footprint, peak_footprint)),
        labels=clean
    )

    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    # fallback se nenhum pico for encontrado
    if markers.max() == 0:
        labeled, n = ndi.label(clean)
        return labeled, n

    markers = ndi.label(markers)[0]

    # 4. WATERSHED
    labels = segmentation.watershed(-dist, markers, mask=clean)

    # 5. REMOVER REGIÕES MUITO PEQUENAS
    props = measure.regionprops(labels)
    final = np.zeros_like(labels, dtype=np.int32)
    idx = 1

    for p in props:
        if p.area >= min_area:
            final[labels == p.label] = idx
            idx += 1

    return final, final.max()

# ==============================
# PROCESSAMENTO DE TODOS OS ARQUIVOS
# ==============================

mask_files = sorted([p for p in Path(binary_masks_dir).glob("*.png")])
print(f"Encontradas {len(mask_files)} máscaras binárias.")

for mask_path in mask_files:
    print("Processando:", mask_path.name)

    # carregar máscara binária
    bin_mask = load_binary_mask(mask_path)

    # aplicar watershed
    inst_mask, n_instances = segment_tree_crowns(bin_mask)

    # salvar
    out_img = (inst_mask.astype(np.uint16))  # 16-bit para suportar muitas instâncias
    out_pil = Image.fromarray(out_img)

    out_path = os.path.join(
        output_dir,
        mask_path.stem + "_instances.png"
    )

    out_pil.save(out_path)

    print(f" → {n_instances} copas detectadas.")

print("\n Finalizado!")
print("Máscaras de instância salvas em:", output_dir)
