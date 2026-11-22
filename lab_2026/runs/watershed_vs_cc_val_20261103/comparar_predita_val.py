"""Componentes conexas x Watershed sobre mascaras PREDITAS pelo P1 (limiar 0,65), somente VALIDACAO.
Inclui pequena varredura (os dois metodos com a mesma limpeza) para comparacao justa.
Rodar a partir de lab_2026/. Saida: resultado_predita_val.csv/.json nesta pasta."""
import json, sys, numpy as np, pandas as pd, cv2
from PIL import Image
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy import ndimage as ndi
from skimage import morphology
sys.path.insert(0, str(Path.cwd()))
import separate_tree_crowns_whatershed as ws
P = Path("/home/paulomaciel/proj"); HERE = Path(__file__).resolve().parent
MASKS = HERE / "pipeline_val_predita" / "mascaras" / "binary"
s = pd.read_csv(P / "unet_spatial_split/tiles_split.csv"); tiles = list(s[s.split == "val"].tile)

def clean(mask):
    c = morphology.closing(mask, morphology.disk(5))
    c = morphology.remove_small_holes(c.astype(bool), area_threshold=32)
    return morphology.remove_small_objects(c, min_size=32)

def cc(mask, min_area):
    lab, n = ndi.label(clean(mask))
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    keep = np.zeros(n + 1, np.int32); keep[1:][sizes >= min_area] = np.arange(1, (sizes >= min_area).sum() + 1)
    return keep[lab]

def gts_of(t):
    jp = next((q for q in [P / "labels" / f"{t}.json", P / "labels2" / f"{t}.json"] if q.exists()), None)
    out = []
    if jp:
        for sh in json.load(open(jp))["shapes"]:
            if sh.get("label") != "tree" or len(sh["points"]) < 3: continue
            m = np.zeros((512, 512), np.uint8); cv2.fillPoly(m, [np.array(sh["points"], np.int32)], 1); out.append(m.astype(bool))
    return out

def score(inst, gts):
    ids = [i for i in np.unique(inst) if i > 0]
    if not gts or not ids: return len(ids), 0
    M = np.zeros((len(gts), len(ids)))
    for b, i in enumerate(ids):
        p = inst == i
        for a, g in enumerate(gts):
            it = (g & p).sum()
            if it: M[a, b] = it / (g | p).sum()
    r, c = linear_sum_assignment(-M)
    return len(ids), int((M[r, c] >= 0.5).sum())

configs = [("cc", None, a) for a in (500, 1000, 2000)] + \
          [("watershed", fp, a) for fp in (41, 75, 121, 201, 251) for a in (1000,)]
data = [(t, gts_of(t), (np.array(Image.open(MASKS / f"{t}.png")) > 127).astype(np.uint8)) for t in tiles]
G = sum(len(g) for _, g, _ in data)
rows = []
for meth, fp, a in configs:
    n_tot = tp_tot = 0; err = []
    for t, gts, m in data:
        inst = cc(m, a) if meth == "cc" else ws.segment_tree_crowns(m, min_area=a, peak_footprint=fp)[0]
        n, tp = score(inst, gts); n_tot += n; tp_tot += tp; err.append(n - len(gts))
    pr, rc = tp_tot / max(n_tot, 1), tp_tot / G
    rows.append(dict(metodo=meth, peak_footprint=fp, min_area=a, instancias=n_tot, tp=tp_tot, precisao=pr, recall=rc,
                     f1=2 * pr * rc / max(pr + rc, 1e-9), mae_contagem=float(np.mean(np.abs(err)))))
    print(rows[-1], flush=True)
d = pd.DataFrame(rows); d.to_csv(HERE / "resultado_predita_val.csv", index=False)
(HERE / "resultado_predita_val.json").write_text(json.dumps({"n_val_tiles": len(tiles), "n_gt": G,
    "melhor": d.loc[d.f1.idxmax()].to_dict(), "configs": rows}, indent=1, default=str), encoding="utf-8")
