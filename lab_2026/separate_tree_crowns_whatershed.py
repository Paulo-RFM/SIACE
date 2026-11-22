"""
whatershed
Gera uma máscara por instância (copa individual) a partir de máscaras binárias (árvore vs fundo).

Pode ser usado como script (CLI abaixo) ou importado: `segment_tree_crowns(mask, ...)`.
Parâmetros calibrados na validação (máscaras de referência): peak_footprint=251, min_area=1000
(runs/watershed_ref_calib_20261102/frozen_watershed_params.json). Os defaults da CLI foram
mantidos nos valores originais para não mudar execuções antigas; passe os calibrados explicitamente.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import cv2
from skimage import morphology, segmentation, feature, measure
from scipy import ndimage as ndi

# ==============================
# PARÂMETROS PADRÃO (originais)
# ==============================

MIN_AREA = 150          # remove copas muito pequenas (pixels)
PEAK_FOOTPRINT = 15     # tamanho da área para procurar picos (pixels)
CLOSE_KERNEL = 5        # fecho morfológico (raio do disco)
SMALL_HOLES_AREA = 32
SMALL_OBJECTS_SIZE = 32

# ==============================
# FUNÇÕES
# ==============================

def load_binary_mask(path):
    """Carrega máscara binária (0/255) e retorna array 0/1"""
    m = Image.open(path).convert("L")
    arr = np.array(m)
    arr = (arr > 127).astype(np.uint8)
    return arr


def clean_binary_mask(binary_mask, close_kernel=CLOSE_KERNEL, small_holes_area=SMALL_HOLES_AREA,
                      small_objects_size=SMALL_OBJECTS_SIZE):
    """Limpeza morfologica comum aos dois metodos de individualizacao (fechamento, buracos, objetos pequenos)."""
    kernel = morphology.disk(close_kernel)
    clean = morphology.closing(binary_mask, kernel)
    clean = morphology.remove_small_holes(clean.astype(bool), area_threshold=small_holes_area)
    clean = morphology.remove_small_objects(clean.astype(bool), min_size=small_objects_size)
    return clean.astype(np.uint8)


def label_connected_components(binary_mask, min_area, close_kernel=CLOSE_KERNEL,
                               small_holes_area=SMALL_HOLES_AREA, small_objects_size=SMALL_OBJECTS_SIZE):
    """Metodo adotado no pipeline final: cada regiao conexa (vizinhanca-4) da mascara limpa e uma arvore;
    regioes com area < min_area pixels sao descartadas. Escolhido na validacao contra o watershed
    (runs/watershed_vs_cc_val_20261103/)."""
    clean = clean_binary_mask(binary_mask, close_kernel, small_holes_area, small_objects_size)
    lab, n = ndi.label(clean)  # vizinhanca-4, como na comparacao da validacao
    if n == 0:
        return np.zeros_like(clean, dtype=np.int32), 0
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = sizes >= min_area
    remap = np.zeros(n + 1, np.int32)
    remap[1:][keep] = np.arange(1, int(keep.sum()) + 1)
    final = remap[lab]
    return final, int(final.max())


PIXEL_M = 0.0319  # tamanho do pixel (m) usado na calibracao do h-maxima (calibrar_individualizacao_val.py)


def segment_tree_crowns_hmax(binary_mask, h_m, min_area):
    """Watershed com marcadores h-maxima da transformada de distancia (2026-11-11). Mesma implementacao de
    hmax_watershed em calibrar_individualizacao_val.py, onde foi escolhido na validacao com as mascaras da
    SegFormer MiT-B1: h em metros convertido para pixels com PIXEL_M; regioes < min_area pixels descartadas."""
    clean = clean_binary_mask(binary_mask)
    if clean.sum() == 0:
        return np.zeros(binary_mask.shape, np.int32), 0
    dist = ndi.distance_transform_edt(clean)
    markers = ndi.label(morphology.h_maxima(dist, round(h_m / PIXEL_M)))[0]
    lab = segmentation.watershed(-dist, markers, mask=clean)
    ids, sizes = np.unique(lab[lab > 0], return_counts=True)
    keep = np.zeros(lab.max() + 1, np.int32)
    big = ids[sizes >= min_area]
    keep[big] = np.arange(1, len(big) + 1)
    final = keep[lab]
    return final, int(final.max())


def segment_tree_crowns(binary_mask, min_area=MIN_AREA, peak_footprint=PEAK_FOOTPRINT,
                        close_kernel=CLOSE_KERNEL, small_holes_area=SMALL_HOLES_AREA,
                        small_objects_size=SMALL_OBJECTS_SIZE):
    """
    Separa copas individuais usando:
    - closing morfológico
    - distance transform
    - peak_local_max
    - watershed
    """

    # 1. MORPHOLOGY CLEANUP
    clean = clean_binary_mask(binary_mask, close_kernel, small_holes_area, small_objects_size)

    if clean.sum() == 0:
        return np.zeros_like(clean, dtype=np.int32), 0

    # 2. DISTANCE TRANSFORM
    dist = ndi.distance_transform_edt(clean)

    # 3. PICOS → CENTROIDES
    # scikit-image >= 0.20 removeu `indices`; o retorno padrao ja sao coordenadas.
    coords = feature.peak_local_max(
        dist,
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


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ==============================
# PROCESSAMENTO DE TODOS OS ARQUIVOS (CLI)
# ==============================

def main():
    parser = argparse.ArgumentParser(description="Watershed sobre mascaras binarias, com proveniencia.")
    # diretório das máscaras binárias (predição do modelo ou referência manual)
    parser.add_argument("--binary-masks-dir", default="/home/paulomaciel/proj/masks_bin_pred/")
    # onde serão salvas as máscaras de instância
    parser.add_argument("--output-dir", default="/home/paulomaciel/proj/masks_instances/")
    parser.add_argument("--mask-provenance", choices=["predicao", "referencia"], required=True,
                        help="origem das mascaras binarias de entrada")
    parser.add_argument("--segmentation-checkpoint", default=None,
                        help="checkpoint que gerou as mascaras (obrigatorio se --mask-provenance predicao)")
    parser.add_argument("--segmentation-threshold", type=float, default=None)
    parser.add_argument("--min-area", type=int, default=MIN_AREA)
    parser.add_argument("--peak-footprint", type=int, default=PEAK_FOOTPRINT)
    parser.add_argument("--close-kernel", type=int, default=CLOSE_KERNEL)
    args = parser.parse_args()
    if args.mask_provenance == "predicao" and not args.segmentation_checkpoint:
        parser.error("--segmentation-checkpoint e obrigatorio para mascaras preditas.")

    binary_masks_dir = args.binary_masks_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    params = dict(min_area=args.min_area, peak_footprint=args.peak_footprint, close_kernel=args.close_kernel,
                  small_holes_area=SMALL_HOLES_AREA, small_objects_size=SMALL_OBJECTS_SIZE)

    mask_files = sorted([p for p in Path(binary_masks_dir).glob("*.png")])
    print(f"Encontradas {len(mask_files)} máscaras binárias.")
    instance_counts = []

    for mask_path in mask_files:
        print("Processando:", mask_path.name)
        bin_mask = load_binary_mask(mask_path)
        inst_mask, n_instances = segment_tree_crowns(bin_mask, **params)
        # 16-bit para suportar muitas instâncias
        Image.fromarray(inst_mask.astype(np.uint16)).save(
            os.path.join(output_dir, mask_path.stem + "_instances.png"))
        print(f" → {n_instances} copas detectadas.")
        instance_counts.append({"tile": mask_path.stem, "n_instances": int(n_instances)})

    pd.DataFrame(instance_counts).to_csv(os.path.join(output_dir, "instances_per_tile.csv"), index=False)
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "binary_masks_dir": str(binary_masks_dir),
        "mask_provenance": args.mask_provenance,
        "segmentation_checkpoint": args.segmentation_checkpoint,
        "segmentation_checkpoint_sha256": _sha256(args.segmentation_checkpoint) if args.segmentation_checkpoint else None,
        "segmentation_threshold": args.segmentation_threshold,
        "cleaning": {"closing_disk_radius": args.close_kernel, "remove_small_holes_area": SMALL_HOLES_AREA,
                     "remove_small_objects_min_size": SMALL_OBJECTS_SIZE},
        "watershed": {"marker": "peak_local_max(distance_transform_edt)", "peak_footprint": args.peak_footprint,
                      "fallback": "ndi.label se nenhum pico"},
        "min_area": args.min_area,
        "n_tiles": len(instance_counts),
        "n_instances_total": int(sum(r["n_instances"] for r in instance_counts)),
    }
    with open(os.path.join(output_dir, "watershed_provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, ensure_ascii=False)

    print("\n Finalizado!")
    print("Máscaras de instância salvas em:", output_dir)


if __name__ == "__main__":
    main()
