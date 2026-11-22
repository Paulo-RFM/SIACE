"""Reavalia no teste, com as mascaras de REFERENCIA, o watershed com parametros originais (fp 15, a 150) e
congelados (fp 251, a 1000) de runs/watershed_ref_calib_20261102, contando os 2 poligonos do tile
OrtoRP-3cm_082_013, cujo arquivo de anotacao (antes 'OrtoRP-3cm_082_013 - Copia.json') passou a ter o nome do tile
depois daquela avaliacao. Nenhum parametro e escolhido aqui. Rodar a partir de lab_2026/."""
import sys, json
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, '.')
import separate_tree_crowns_whatershed as ws
from calibrar_individualizacao_val import gts_of
OUT = Path('runs/watershed_ref_teste_420_20261106')
CFG = [(15, 150), (251, 1000)]

def run(t):
    m = (np.array(Image.open(f'../masks/{t}.png')) > 127).astype(np.uint8)
    g = gts_of(t); res = {}
    for fp, a in CFG:
        inst = ws.segment_tree_crowns(m, min_area=a, peak_footprint=fp)[0]
        ids = [i for i in np.unique(inst) if i > 0]; tp, ious = 0, []
        if g and ids:
            M = np.zeros((len(g), len(ids)))
            for b, i in enumerate(ids):
                p = inst == i
                for k, gg in enumerate(g):
                    it = (gg & p).sum()
                    if it: M[k, b] = it / (gg | p).sum()
            r, c = linear_sum_assignment(-M); ok = M[r, c] >= 0.5; tp = int(ok.sum()); ious = list(M[r, c][ok])
        res[(fp, a)] = (len(g), len(ids), tp, ious)
    return t, res

if __name__ == '__main__':
    s = pd.read_csv('/home/paulomaciel/proj/unet_spatial_split/tiles_split.csv'); tiles = list(s[s.split == 'test'].tile)
    with Pool(3) as p: R = p.map(run, tiles)
    out = {}
    for cfg in CFG:
        d = pd.DataFrame([dict(tile=t, n_gt=r[cfg][0], n_pred=r[cfg][1], tp=r[cfg][2]) for t, r in R])
        d.to_csv(OUT / f'per_tile_fp{cfg[0]}_a{cfg[1]}.csv', index=False)
        ious = [x for _, r in R for x in r[cfg][3]]
        G, P, TP = int(d.n_gt.sum()), int(d.n_pred.sum()), int(d.tp.sum()); pr, rc = TP / P, TP / G; e = d.n_pred - d.n_gt
        out[f'fp{cfg[0]}_a{cfg[1]}'] = dict(arvores=G, instancias=P, tp=TP, precisao=pr, recall=rc, f1=2*pr*rc/(pr+rc),
            iou_medio_tp=float(np.mean(ious)), erro_medio=float(e.mean()), mae=float(e.abs().mean()), exata=float((e == 0).mean()))
    (OUT / 'resultado.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
