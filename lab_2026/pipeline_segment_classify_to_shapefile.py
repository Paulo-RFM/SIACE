#!/usr/bin/env python3
"""
Pipeline completo:
1) Usa um modelo binário (U-Net) para segmentar copas em cada tile (.tif)
2) Faz pós-processamento (closing, distance transform, peak_local_max, watershed)
3) Extrai instâncias, gera crop zoom e classifica com EfficientNet-B0
4) Converte máscara da instância em polígono no sistema de coordenadas do raster
5) (Opcional) Agrupa instâncias próximas (mescla duplicatas entre tiles)
6) Salva GeoPackage (.gpkg) e GeoJSON com predicted_species, confidence, tile, instance_id, area_m2

Ajuste os caminhos e parâmetros na seção CONFIG abaixo.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
from PIL import Image
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping, Polygon
from shapely.ops import unary_union
import geopandas as gpd

import torch
import torch.nn.functional as F
from torchvision import transforms

from sklearn.cluster import DBSCAN
from scipy import ndimage as ndi
from skimage import morphology, segmentation, feature, measure
import cv2
from efficientnet_pytorch import EfficientNet

# ============================
# CONFIGURAÇÕES
# ============================

BASE = "/home/paulomaciel/proj/"

# Modelos
seg_model_checkpoint = BASE + "unet_binary_tree_detector.pth"   # seu UNet binário
cls_model_checkpoint = BASE + "species_model/efficientnet_b0_best.pth"  # EfficientNet salvo

# Diretórios
tiles_dir      = BASE + "tiles_512/"            # .tif tiles
out_dir        = BASE + "inference_output/"   # saída geral
os.makedirs(out_dir, exist_ok=True)

# Temporários / outputs
masks_pred_dir = os.path.join(out_dir, "masks_pred/")       # máscara binária predita per-tile
inst_dir       = os.path.join(out_dir, "instances/")       # máscara de instâncias per-tile (labels)
crops_dir      = os.path.join(out_dir, "crops/")           # crops usados na classificação
os.makedirs(masks_pred_dir, exist_ok=True)
os.makedirs(inst_dir, exist_ok=True)
os.makedirs(crops_dir, exist_ok=True)

# Hiperparâmetros
seg_img_size = (512, 512)      # usado ao treinar o UNet; manter igual ao treino
cls_img_size = 512             # crops já são 512x512; EfficientNet foi treinado com 512
threshold = 0.5                # limiar para binarizar sigmoid
min_area_px = 150              # filtrar objetos muito pequenos (pixels)
padding_px = 16                # padding ao extrair crop para classificador
merge_duplicates = True        # se True, agrupa instâncias próximas entre tiles
merge_distance_m = 2.0         # distância em metros para agrupar duplicatas (ajuste conforme resolução)

# Output geofile
out_gpkg = os.path.join(out_dir, "predicted_species.gpkg")
out_geojson = os.path.join(out_dir, "predicted_species.geojson")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================
# UTILITÁRIOS
# ============================

def ensure_rgb(arr):
    if arr.ndim == 2:
        return np.stack([arr]*3, axis=-1)
    return arr

# transforma máscara de instância (inteiros) em polígonos georreferenciados
def mask_to_polygons(mask, transform, crs):
    """Retorna lista de (polygon_shapely, pixel_area) convertidas para coords do raster"""
    results = []
    # shapes yields (geometry, value)
    mask_uint = mask.astype(np.uint8)
    for geom, val in shapes(mask_uint, mask=mask_uint > 0, transform=transform):
        poly = shape(geom)
        # area em pixels: area of mask in mask coords (but we'll compute area in meters via transform)
        results.append(poly)
    return results

# converte bbox pixel coords para world using transform
def pixel_bbox_to_world(transform, x0, y0, x1, y1):
    # rasterio transform: world = transform * (col, row)
    # returns polygon in world coords
    xs = [x0, x1, x1, x0]
    ys = [y0, y0, y1, y1]
    coords = [rasterio.transform.xy(transform, int(y), int(x), offset='center')[::-1] for (x,y) in zip(xs, ys)]
    return coords

# ============================
# CARREGAR MODELOS
# ============================

# Segmentation model (U-Net, same architecture used in training)
from segmentation_models_pytorch import Unet
seg_model = Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
if os.path.exists(seg_model_checkpoint):
    seg_model.load_state_dict(torch.load(seg_model_checkpoint, map_location=device))
    print("Segmentation checkpoint loaded.")
else:
    raise FileNotFoundError("Segmentation checkpoint not found: " + seg_model_checkpoint)
seg_model.eval()

# Classification model (EfficientNet-B0)
num_classes = 6
cls_model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=num_classes)
if os.path.exists(cls_model_checkpoint):
    cls_model.load_state_dict(torch.load(cls_model_checkpoint, map_location=device))
    print("Classification checkpoint loaded.")
else:
    raise FileNotFoundError("Classification checkpoint not found: " + cls_model_checkpoint)
cls_model = cls_model.to(device)
cls_model.eval()

# transforms
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

# ============================
# PROCESSAMENTO POR TILE
# ============================

rows = []  # cada entrada será convertida para GeoDataFrame

tile_paths = sorted([p for p in Path(tiles_dir).glob("*.tif")])
print("Tiles found:", len(tile_paths))

for tile_path in tqdm(tile_paths):
    basename = tile_path.stem
    # load image and geoinfo
    with rasterio.open(tile_path) as src:
        img = src.read([1,2,3])  # (3, H, W) assuming tif has RGB bands 1..3
        transform = src.transform
        crs = src.crs
        res = src.res  # pixel size (xres, yres) in units of CRS
    # convert to H,W,3
    img = np.transpose(img, (1,2,0))
    img_rgb = ensure_rgb(img)

    # ==== segmentação ====
    pil = Image.fromarray(img_rgb)
    inp = seg_transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = seg_model(inp)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # HxW (resized to seg_img_size)

    # resize mask back to original tile size
    prob_resized = cv2.resize(prob, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask_bin = (prob_resized > threshold).astype(np.uint8)  # 0/1

    # save mask prediction (optional)
    mask_save_path = os.path.join(masks_pred_dir, basename + "_pred_bin.png")
    Image.fromarray((mask_bin*255).astype(np.uint8)).save(mask_save_path)

    # ==== post-process + watershed ====
    # morphological cleanup
    kernel = morphology.disk(5)
    clean = morphology.closing(mask_bin, kernel)
    clean = morphology.remove_small_holes(clean.astype(bool), area_threshold=64)
    clean = morphology.remove_small_objects(clean.astype(bool), min_size=32)
    clean = clean.astype(np.uint8)

    if clean.sum() == 0:
        # nothing detected in this tile
        continue

    # distance and peaks
    dist = ndi.distance_transform_edt(clean)
    coords = feature.peak_local_max(dist, indices=True, footprint=np.ones((15,15)), labels=clean)
    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (r,c) in enumerate(coords, start=1):
        markers[r,c] = i

    if markers.max() == 0:
        labeled, n = ndi.label(clean)
    else:
        markers = ndi.label(markers)[0]
        labels = segmentation.watershed(-dist, markers, mask=clean)
        labeled = labels

    # remove small regions and reindex
    props = measure.regionprops(labeled)
    inst_mask = np.zeros_like(labeled, dtype=np.int32)
    idx = 1
    for p in props:
        if p.area >= min_area_px:
            inst_mask[labeled == p.label] = idx
            idx += 1

    n_instances = inst_mask.max()
    if n_instances == 0:
        continue

    # save instance label mask
    inst_save_path = os.path.join(inst_dir, basename + "_instances.png")
    # save as 16-bit to allow many labels
    inst_img = (inst_mask.astype(np.uint16))
    Image.fromarray(inst_img).save(inst_save_path)

    # ==== para cada instância: crop + classificar + polígonos georreferenciados ====
    for inst_id in range(1, n_instances+1):
        mask_inst = (inst_mask == inst_id).astype(np.uint8)
        area_px = int(mask_inst.sum())
        if area_px < min_area_px:
            continue

        ys, xs = np.where(mask_inst)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        # padding
        y0p = max(0, y0 - padding_px)
        x0p = max(0, x0 - padding_px)
        y1p = min(img_rgb.shape[0]-1, y1 + padding_px)
        x1p = min(img_rgb.shape[1]-1, x1 + padding_px)

        crop = img_rgb[y0p:y1p+1, x0p:x1p+1]
        crop_path = os.path.join(crops_dir, f"{basename}_inst{inst_id:04d}.png")
        Image.fromarray(crop).save(crop_path)

        # preparar tensor e classificar (usa transform que redimensiona pra cls_img_size)
        crop_pil = Image.fromarray(crop)
        inp_cls = cls_transform(crop_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            out = cls_model(inp_cls)
            probs = softmax(out).squeeze().cpu().numpy()
            pred = int(probs.argmax()) + 1   # back to 1..6
            conf = float(probs.max())

        # converter máscara da instância em polígonos no sistema do raster
        # criar mask_inst cropped to bbox, then get shapes and transform to world coords
        mask_crop = mask_inst[y0p:y1p+1, x0p:x1p+1].astype(np.uint8)
        # we need transform of the crop in world coords: compute its top-left pixel world coords
        # rasterio.transform: world coordinates = transform * (col, row)
        # note: transform expects (col, row) = (x, y)
        tile_transform = transform  # from rasterio.open
        # compute pixel -> world for the crop top-left
        top_left_world = rasterio.transform.xy(tile_transform, y0p, x0p, offset='ul')  # returns (y,x)?? returns (x,y)
        # Actually rasterio.transform.xy returns (x,y)
        crop_origin_x, crop_origin_y = top_left_world[0], top_left_world[1]

        # Using shapes on the mask_crop but need to supply transform for mask_crop:
        # crop_transform maps pixel coordinates (col,row) to world: 
        # new_transform = Affine(tile_transform.a, tile_transform.b, tile_transform.c + x0p*tile_transform.a, ...)
        from rasterio.transform import Affine
        a = tile_transform
        # build transform for crop: shift by x0p,y0p
        crop_transform = Affine(a.a, a.b, a.c + x0p * a.a + x0p * a.b,
                                a.d, a.e, a.f + y0p * a.d + y0p * a.e)
        # simpler and robust: use rasterio.transform.from_origin with crop_origin_x, crop_origin_y and pixel size
        pix_w, pix_h = res[0], res[1]
        crop_transform = rasterio.transform.from_origin(crop_origin_x, crop_origin_y, pix_w, abs(pix_h))

        # shapes yields polygons in world coords when given transform
        polys = []
        for geom, val in shapes(mask_crop, mask=mask_crop > 0, transform=crop_transform):
            poly = shape(geom)
            polys.append(poly)

        if len(polys) == 0:
            # fallback: use bbox polygon in world coords
            # compute four corners world coords
            x0_world, y0_world = rasterio.transform.xy(tile_transform, y0p, x0p, offset='ul')
            x1_world, y1_world = rasterio.transform.xy(tile_transform, y1p, x1p, offset='ul')
            bbox_poly = Polygon([(x0_world,y0_world),(x1_world,y0_world),(x1_world,y1_world),(x0_world,y1_world)])
            poly_union = bbox_poly
        else:
            # union polygons if multiple
            poly_union = unary_union(polys)

        # area in map units (square meters if crs is metric)
        area_map = poly_union.area

        rows.append({
            "tile": basename,
            "instance_id": inst_id,
            "geometry": poly_union,
            "pred_species": int(pred),
            "confidence": float(conf),
            "area_m2": float(area_map),
            "crop_path": crop_path
        })

# ============================
# AGREGAR E SALVAR GEOFILE
# ============================

if len(rows) == 0:
    raise RuntimeError("Nenhuma instância detectada em todos os tiles.")

gdf = gpd.GeoDataFrame(rows, crs=crs)
# algumas versões do geopandas não aceita geometry key, force:
gdf = gdf.set_geometry("geometry")

# ============================
# OPCIONAL: agrupar duplicatas próximas
# ============================
if merge_duplicates:
    print("\nAgrupando instâncias próximas (merge duplicates)...")
    # calcular centróides
    gdf["centroid"] = gdf.geometry.centroid
    coords = np.array([[pt.x, pt.y] for pt in gdf["centroid"]])
    # cluster by distance (meters)
    clustering = DBSCAN(eps=merge_distance_m, min_samples=1, metric='euclidean').fit(coords)
    gdf["cluster"] = clustering.labels_

    merged_rows = []
    for cl in np.unique(clustering.labels_):
        sub = gdf[gdf["cluster"] == cl]
        # decidir species: majority vote of pred_species
        species_mode = int(sub["pred_species"].mode().iloc[0])
        # confidence: max
        conf_max = float(sub["confidence"].max())
        # union geometries
        geom_union = sub.unary_union
        # choose representative tile (first)
        tile_rep = sub["tile"].iloc[0]
        merged_rows.append({
            "tile": tile_rep,
            "instance_id": int(cl),
            "geometry": geom_union,
            "pred_species": species_mode,
            "confidence": conf_max,
            "area_m2": float(geom_union.area),
            "crop_path": None
        })
    mgdf = gpd.GeoDataFrame(merged_rows, crs=crs)
    mgdf = mgdf.set_geometry("geometry")
    print("Clusters merged:", len(mgdf))
    out_gdf = mgdf
else:
    out_gdf = gdf.drop(columns=["centroid","crop_path"], errors='ignore')

print("\nSalvando arquivos vetoriais...")
out_gdf.to_file(out_gpkg, driver="GPKG")
out_gdf.to_file(out_geojson, driver="GeoJSON")

print("Gerando JSONs para QGIS...")
from export_qgis_jsons import export_jsons
export_jsons(out_gdf, out_dir)

print("Gerando projeto QGIS .qgz...")
from export_qgis_project import export_qgis_project
export_qgis_project(out_gpkg, mosaic_path="/home/paulomaciel/proj/ortomosaico.tif",
                    output_path=os.path.join(out_dir, "project.qgz"))

print("\nPIPELINE FINALIZADA!")
print("Total predicted instances:", len(out_gdf))

