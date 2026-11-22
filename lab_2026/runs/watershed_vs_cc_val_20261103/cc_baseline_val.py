"""Linha de base so na VALIDACAO: componentes conexas (mesma limpeza, min_area=1000) x watershed congelado."""
import json, sys, numpy as np, pandas as pd, cv2
from PIL import Image
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy import ndimage as ndi
from skimage import morphology
sys.path.insert(0, str(Path.cwd()))
import separate_tree_crowns_whatershed as ws
P = Path("/home/paulomaciel/proj")
s = pd.read_csv(P / "unet_spatial_split/tiles_split.csv"); tiles = s[s.split == "val"].tile

def cc(mask, min_area=1000):
    c = morphology.closing(mask, morphology.disk(5))
    c = morphology.remove_small_holes(c.astype(bool), area_threshold=32)
    c = morphology.remove_small_objects(c, min_size=32)
    lab, n = ndi.label(c)
    out = np.zeros_like(lab); k = 1
    for i in range(1, n + 1):
        r = lab == i
        if r.sum() >= min_area: out[r] = k; k += 1
    return out

def score(inst, gts):
    preds = [inst == i for i in np.unique(inst) if i > 0]
    if not gts or not preds: return len(preds), 0
    M = np.zeros((len(gts), len(preds)))
    for a, g in enumerate(gts):
        for b, p in enumerate(preds):
            it = (g & p).sum()
            if it: M[a, b] = it / (g | p).sum()
    r, c = linear_sum_assignment(-M)
    return len(preds), int((M[r, c] >= 0.5).sum())

tot = {"cc": [0, 0], "ws": [0, 0]}; G = 0; same = 0
for t in tiles:
    jp = next((q for q in [P / "labels" / f"{t}.json", P / "labels2" / f"{t}.json"] if q.exists()), None)
    gts = []
    if jp:
        for sh in json.load(open(jp))["shapes"]:
            if sh.get("label") != "tree" or len(sh["points"]) < 3: continue
            m = np.zeros((512, 512), np.uint8); cv2.fillPoly(m, [np.array(sh["points"], np.int32)], 1); gts.append(m.astype(bool))
    G += len(gts)
    mask = (np.array(Image.open(P / "masks" / f"{t}.png").convert("L")) > 127).astype(np.uint8)
    a = cc(mask); b, _ = ws.segment_tree_crowns(mask, min_area=1000, peak_footprint=251)
    same += int(a.max() == b.max())
    for k, inst in (("cc", a), ("ws", b)):
        n, tp = score(inst, gts); tot[k][0] += n; tot[k][1] += tp
res = {"n_val_tiles": len(tiles), "n_gt": G, "tiles_mesmo_numero_de_instancias": same}
for k, (n, tp) in tot.items():
    pr, rc = tp / n, tp / G
    res[k] = {"instancias": n, "tp": tp, "precisao": round(pr, 3), "recall": round(rc, 3), "f1": round(2 * pr * rc / (pr + rc), 3)}
print(json.dumps(res, indent=1, ensure_ascii=False))
