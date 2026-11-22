import json, sys, numpy as np, pandas as pd, cv2
from PIL import Image
from pathlib import Path
from scipy.optimize import linear_sum_assignment
SP=sys.argv[1]
s=pd.read_csv('unet_spatial_split/tiles_split.csv'); tiles=s[s.split=='test'].tile
rows=[]; ious_all=[]
for t in tiles:
    jp=next((p for p in [Path('labels')/f'{t}.json',Path('labels2')/f'{t}.json'] if p.exists()),None)
    inst=np.array(Image.open(f'{SP}/out/{t}_instances.png')).astype(np.int32)
    gts=[]
    if jp:
        for sh in json.load(open(jp))['shapes']:
            if sh.get('label')!='tree' or len(sh['points'])<3: continue
            m=np.zeros((512,512),np.uint8); cv2.fillPoly(m,[np.array(sh['points'],np.int32)],1); gts.append(m.astype(bool))
    preds=[inst==i for i in np.unique(inst) if i>0]
    tp=0
    if gts and preds:
        M=np.zeros((len(gts),len(preds)))
        for a,g in enumerate(gts):
            for b,p in enumerate(preds):
                inter=(g&p).sum()
                if inter: M[a,b]=inter/(g|p).sum()
        r,c=linear_sum_assignment(-M); good=M[r,c]>=0.5; tp=int(good.sum()); ious_all+=list(M[r,c][good])
    rows.append(dict(tile=t,n_gt=len(gts),n_pred=len(preds),tp=tp))
d=pd.DataFrame(rows); d.to_csv(f'{SP}/per_tile.csv',index=False)
TP,G,P=d.tp.sum(),d.n_gt.sum(),d.n_pred.sum(); pr,rc=TP/P,TP/G
print(f"arvores anotadas={G} instancias={P} TP(IoU>=0.5)={TP}")
print(f"precisao={pr:.3f} recall={rc:.3f} F1={2*pr*rc/(pr+rc):.3f} IoU medio dos acertos={np.mean(ious_all):.3f}")
e=d.n_pred-d.n_gt
print(f"erro de contagem por tile: medio={e.mean():.2f} MAE={e.abs().mean():.2f} contagem exata={(e==0).mean():.1%} | razao inst/arv={P/G:.2f}")
