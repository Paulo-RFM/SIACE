#!/usr/bin/env python3
"""Pipeline de inferencia: tile -> U-Net ResNet-34 (run p1_20261101_140519) -> individualizacao -> EfficientNet-B0 -> arvore georreferenciada.

Etapas por tile (mesmo pre-processamento do treino/avaliacao):
  1. RGB (descarta alfa), ToTensor + Normalize ImageNet; sigmoide >= limiar congelado da U-Net ResNet-34;
     pixels sem dado (RGB == 0) nunca sao arvore (valid_pixel_mask do protocolo).
  2. Individualizacao congelada na validacao: componentes conexas (padrao, frozen_instance_params.json)
     ou watershed (frozen_watershed_params.json, via --ws-params).
  3. Recorte de cada copa: bbox da instancia + 20 px, redimensionado para 512 (igual aos recortes
     de treino de generate_crops_from_json.py) -> EfficientNet-B0 -> classe, confianca, probabilidades.
  4. Georreferencia pelas tags GeoTIFF (tree_georef.py); copas cortadas pela borda sao unidas (no maximo uma parte por tile, --merge-rule) e a
     classe da arvore unida = argmax das probabilidades ponderadas por pixels das partes.

Saida em --output-dir: copas.geojson, arvores.geojson, arvores.csv, instancias.csv (por tile, antes
da uniao), contagem_por_classe.csv, summary.json; parciais em partes/ permitem retomar (--resume).

Exemplos:
  # teste (204 tiles), salvando mascaras para avaliacao:
  python pipeline_cidade.py --tiles-dir /home/paulomaciel/proj/imagens --partition test \
      --cls-run runs/<run_efficientnet> --output-dir runs/pipeline_test_X --save-masks
  # cidade inteira:
  python pipeline_cidade.py --tiles-dir /home/paulomaciel/Datasets/drone_rp_v01/tiles512 \
      --cls-run runs/<run_efficientnet> --output-dir runs/pipeline_cidade_X --resume
"""
import argparse
import contextlib
import json
import multiprocessing as mp
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import separate_tree_crowns_whatershed as ws
import species_data as sd
import tree_georef as tg

LAB = Path(__file__).resolve().parent
DEFAULT_SEG_RUN = LAB / "runs" / "p1_20261101_140519"
# Metodo de individualizacao adotado (escolhido na validacao). O watershed antigo continua disponivel com
# --ws-params runs/watershed_ref_calib_20261102/frozen_watershed_params.json
DEFAULT_WS = LAB / "runs" / "watershed_vs_cc_val_20261103" / "frozen_instance_params.json"
CROP_PAD = 20
CROP_SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles-dir", required=True)
    p.add_argument("--partition", choices=["train", "val", "test"], default=None,
                   help="processa so os tiles desta particao do split espacial")
    p.add_argument("--seg-run", default=str(DEFAULT_SEG_RUN))
    # 2026-11-11: arquitetura do run de segmentacao (padrao = U-Net ResNet-34, comportamento original)
    p.add_argument("--seg-arch", default="Unet", choices=["Unet", "Segformer", "DeepLabV3Plus", "UnetPlusPlus"])
    p.add_argument("--seg-encoder", default="resnet34")
    p.add_argument("--ws-params", default=str(DEFAULT_WS))
    p.add_argument("--cls-run", required=True, help="run da EfficientNet (checkpoint_best.pth + config.json)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cls-batch-size", type=int, default=8,
                   help="lote do classificador (crops 512x512); 32 estoura a GPU de 4 GB")
    p.add_argument("--workers", type=int, default=3,
                   help="processos de CPU para watershed/recortes (0 = sequencial no processo principal)")
    p.add_argument("--chunk", type=int, default=400, help="tiles por parcial salva em disco")
    p.add_argument("--limit", type=int, default=None, help="apenas os N primeiros tiles (smoke)")
    p.add_argument("--save-masks", action="store_true", help="salva mascara binaria e de instancias por tile")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--merge-rule", choices=["uma_parte_por_tile", "encadeada"], default=tg.MERGE_RULE,
                   help="regra de uniao das copas cortadas pela borda (tree_georef.merge_border_crowns)")
    return p.parse_args()


def load_tile(path):
    """Pixels sem dado: RGB == 0 ou, em tiles RGBA, alpha == 0 (no mosaico da cidade a area fora da
    ortofoto e branca e transparente). Pixels sem dado viram preto, como no criterio do treino.
    Os 1.018 tiles anotados nao tem pixel transparente, entao nada muda para eles."""
    im = Image.open(path)
    rgb = np.array(im.convert("RGB"), dtype=np.uint8)
    valid = np.any(rgb != 0, axis=2)
    if "A" in im.getbands():
        valid &= np.asarray(im.getchannel("A")) > 0
        rgb[~valid] = 0
    return rgb, valid


TILE_SIZE = 512


def pad_to(rgb, size):
    h, w = rgb.shape[:2]
    if h > size or w > size:
        raise ValueError(f"tile maior que {size}x{size}: {rgb.shape}")
    if h == size and w == size:
        return rgb
    return np.pad(rgb, ((0, size - h), (0, size - w), (0, 0)))


def to_tensor(rgb):
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def process_tile(job):
    """CPU: watershed + recortes + geometria de cada copa de um tile."""
    tile, tif_path, pred, rgb, ws_params, save_dir, prob = job
    params = dict(ws_params)
    metodo = params.pop("method", "watershed")
    if metodo == "componentes_conexas":
        inst, n = ws.label_connected_components(pred, **params)
    elif metodo == "watershed_hmax":  # 2026-11-11
        inst, n = ws.segment_tree_crowns_hmax(pred, **params)
    else:
        inst, n = ws.segment_tree_crowns(pred, **params)
    if save_dir is not None:
        Image.fromarray((pred * 255).astype(np.uint8)).save(Path(save_dir) / "binary" / f"{tile}.png")
        Image.fromarray(inst.astype(np.uint16)).save(Path(save_dir) / "instances" / f"{tile}_instances.png")
    if n == 0:
        return tile, [], []
    gt = tg.read_geotransform(tif_path)
    recs = tg.tile_records(inst, tile, gt, inst.shape)
    # certeza de "ser arvore": probabilidade media da U-Net nos pixels da copa (nao altera a copa)
    for r in recs:
        r["seg_prob_mean"] = float(prob[inst == r["instance_id"]].astype(np.float32).mean())
    crops = []
    h, w = inst.shape
    for r in recs:
        ys, xs = np.nonzero(inst == r["instance_id"])
        y0, y1 = max(0, ys.min() - CROP_PAD), min(h - 1, ys.max() + CROP_PAD)
        x0, x1 = max(0, xs.min() - CROP_PAD), min(w - 1, xs.max() + CROP_PAD)
        crop = Image.fromarray(rgb[y0:y1 + 1, x0:x1 + 1]).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)
        crops.append(np.asarray(crop, dtype=np.uint8))
    return tile, recs, crops


def main():
    args = parse_args()
    out = Path(args.output_dir)
    parts_dir = out / "partes"
    if out.exists() and not args.resume and any(out.iterdir()):
        raise FileExistsError(f"{out} ja existe; use --resume para continuar.")
    parts_dir.mkdir(parents=True, exist_ok=True)
    save_dir = None
    if args.save_masks:
        save_dir = out / "mascaras"
        (save_dir / "binary").mkdir(parents=True, exist_ok=True)
        (save_dir / "instances").mkdir(parents=True, exist_ok=True)

    # ---------------- modelos e parametros congelados ----------------
    seg_run, cls_run = Path(args.seg_run), Path(args.cls_run)
    seg_ckpt = seg_run / "checkpoint_best.pth"
    threshold = json.loads((seg_run / "frozen_threshold.json").read_text())
    if threshold.get("criterion") != "validation_iou":
        raise ValueError(f"Limiar do run de segmentacao nao foi selecionado na validacao: {threshold}")
    thr = float(threshold["threshold"])
    wsp = json.loads(Path(args.ws_params).read_text())
    method = wsp.get("method", "watershed")
    if method == "componentes_conexas":
        ws_params = {"method": method, "min_area": int(wsp["min_area"])}
    elif method == "watershed":
        ws_params = {"method": method, "min_area": int(wsp["min_area"]), "peak_footprint": int(wsp["peak_footprint"])}
    elif method == "watershed_hmax":  # 2026-11-11: escolhido na validacao com as mascaras da SegFormer
        ws_params = {"method": method, "min_area": int(wsp["min_area"]), "h_m": float(wsp["h_m"])}
    else:
        raise ValueError(f"metodo de individualizacao desconhecido: {method}")
    cls_cfg = json.loads((cls_run / "config.json").read_text())
    if cls_cfg.get("smoke"):
        raise ValueError("O run de classificacao informado e um smoke; use um run completo.")
    class_ids, class_names = cls_cfg["class_ids"], cls_cfg["class_names"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import segmentation_models_pytorch as smp
    from efficientnet_pytorch import EfficientNet
    seg = getattr(smp, args.seg_arch)(encoder_name=args.seg_encoder, encoder_weights=None, in_channels=3,
                                      classes=1).to(device)
    seg.load_state_dict(torch.load(seg_ckpt, map_location=device, weights_only=True))
    seg.eval()
    cls = EfficientNet.from_name("efficientnet-b0", num_classes=len(class_ids)).to(device)
    cls.load_state_dict(torch.load(cls_run / "checkpoint_best.pth", map_location=device, weights_only=True))
    cls.eval()

    # ---------------- lista de tiles ----------------
    tiles = sorted(Path(args.tiles_dir).glob("*.tif"))
    if args.partition:
        tile_split, _ = sd.load_tile_split(sd.resolve_split_path())
        tiles = [t for t in tiles if tile_split.get(t.stem) == args.partition]
    if args.limit:
        tiles = tiles[:args.limit]
    if not tiles:
        raise FileNotFoundError("Nenhum tile selecionado.")
    chunks = [tiles[i:i + args.chunk] for i in range(0, len(tiles), args.chunk)]
    provenance = {
        "script": Path(__file__).name, "tiles_dir": args.tiles_dir, "partition": args.partition,
        "n_tiles": len(tiles), "tile_list_sha256": sd.sha256_text_lines([t.stem for t in tiles]),
        "segmentation_run": str(seg_run), "segmentation_arch": f"{args.seg_arch}_{args.seg_encoder}",
        "segmentation_checkpoint_sha256": sd.sha256_file(seg_ckpt),
        "segmentation_threshold": thr, "threshold_criterion": threshold.get("criterion"),
        "watershed_params": ws_params, "watershed_params_file": str(args.ws_params),
        "classifier_run": str(cls_run), "classifier_checkpoint_sha256": sd.sha256_file(cls_run / "checkpoint_best.pth"),
        "classifier_selection": f"best_val_f1_macro={cls_cfg.get('best_val_f1_macro')}",
        "class_ids": class_ids, "class_names": class_names, "crop_pad_px": CROP_PAD, "cls_batch_size": args.cls_batch_size, "crop_size": CROP_SIZE,
        "device": str(device), "border_merge_rule": args.merge_rule,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: provenance[k] for k in ("n_tiles", "segmentation_threshold", "watershed_params",
                                                 "classifier_run")}, indent=2, ensure_ascii=False), flush=True)

    t_start = time.perf_counter()
    done_tiles = 0
    # 'spawn': processos limpos. Com 'fork' depois de inicializar o PyTorch os filhos podem travar.
    pool_cm = (ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"))
               if args.workers > 0 else contextlib.nullcontext(None))
    with pool_cm as pool:
        mapper = pool.map if pool is not None else (lambda f, it, chunksize=1: map(f, it))
        for ci, chunk in enumerate(chunks):
            part = parts_dir / f"parte_{ci:05d}.pkl"
            if part.exists():
                done_tiles += len(chunk)
                continue
            jobs, nodata = [], []
            with torch.no_grad():
                for bi in range(0, len(chunk), args.batch_size):
                    batch = chunk[bi:bi + args.batch_size]
                    loaded = [load_tile(p) for p in batch]
                    # tiles de borda do mosaico sao menores que 512: preenche com zeros (nodata) a
                    # direita/abaixo so para a rede; a predicao e recortada de volta ao tamanho original,
                    # entao a origem do georreferenciamento (canto superior esquerdo) nao muda.
                    x = torch.stack([to_tensor(pad_to(rgb, TILE_SIZE)) for rgb, _ in loaded]).to(device)
                    prob = torch.sigmoid(seg(x))[:, 0].float().cpu().numpy()
                    prob = [pr[:rgb.shape[0], :rgb.shape[1]] for (rgb, _), pr in zip(loaded, prob)]
                    for p, (rgb, valid), pr in zip(batch, loaded, prob):
                        if not valid.any():
                            nodata.append(p.stem)
                            continue
                        pred = ((pr >= thr) & valid).astype(np.uint8)
                        jobs.append((p.stem, str(p), pred, rgb, ws_params,
                                     str(save_dir) if save_dir else None, pr.astype(np.float16)))
            recs, crops = [], []
            for tile, r, c in mapper(process_tile, jobs, chunksize=4):
                recs += r
                crops += c
            if crops:
                probs = []
                if device.type == "cuda":
                    torch.cuda.empty_cache()  # libera o cache da segmentacao antes do classificador
                with torch.no_grad():
                    for k in range(0, len(crops), args.cls_batch_size):
                        x = torch.stack([to_tensor(c) for c in crops[k:k + args.cls_batch_size]]).to(device)
                        probs.append(torch.softmax(cls(x).float(), 1).cpu().numpy())
                probs = np.concatenate(probs)
                for r, pv in zip(recs, probs):
                    r["probs"] = pv
            with open(part, "wb") as fh:
                pickle.dump({"records": recs, "nodata": nodata, "tiles": [p.stem for p in chunk]}, fh)
            done_tiles += len(chunk)
            el = time.perf_counter() - t_start
            rate = done_tiles / max(el, 1e-9)
            print(f"[{datetime.now():%H:%M:%S}] {done_tiles}/{len(tiles)} tiles | {len(recs)} copas no bloco | "
                  f"{rate:.2f} tiles/s | ETA {(len(tiles) - done_tiles) / max(rate, 1e-9) / 3600:.2f} h", flush=True)

    # ---------------- consolidacao ----------------
    recs, nodata = [], []
    for part in sorted(parts_dir.glob("parte_*.pkl")):
        d = pickle.load(open(part, "rb"))
        recs += d["records"]
        nodata += d["nodata"]
    if not recs:
        raise RuntimeError("Nenhuma copa detectada.")
    inst = pd.DataFrame(recs)
    probs = np.stack(inst["probs"].to_numpy())
    inst["pred_class"] = [class_ids[i] for i in probs.argmax(1)]
    inst["pred_name"] = [class_names[i] for i in probs.argmax(1)]
    inst["confidence"] = probs.max(1)
    for k, n in enumerate(class_names):
        inst[f"prob_{n}"] = probs[:, k]
    if "seg_prob_mean" not in inst.columns:  # parciais gerados antes de 2026-11-04 nao tem o campo
        inst["seg_prob_mean"] = np.nan
    inst[["tile", "instance_id", "n_pixels", "touches_tile_border", "seg_prob_mean", "pred_class", "pred_name",
          "confidence"]
         + [f"prob_{n}" for n in class_names]].to_csv(out / "instancias.csv", index=False)

    trees = tg.merge_border_crowns(inst[["tile", "instance_id", "n_pixels", "touches_tile_border",
                                         "geometry", "centroid"]], rule=args.merge_rule)
    key = {f"{t}#{i}": k for k, (t, i) in enumerate(zip(inst["tile"], inst["instance_id"]))}
    agg_probs, conf, pc, pn, segp = [], [], [], [], []
    seg_mean = inst["seg_prob_mean"].to_numpy(dtype=float)
    for parts in trees["parts"]:
        idx = [key[p] for p in parts.split(";")]
        w = inst.loc[idx, "n_pixels"].to_numpy(dtype=float)
        pv = (probs[idx] * w[:, None]).sum(0) / w.sum()
        agg_probs.append(pv); conf.append(float(pv.max()))
        pc.append(class_ids[int(pv.argmax())]); pn.append(class_names[int(pv.argmax())])
        segp.append(float((seg_mean[idx] * w).sum() / w.sum()))  # media ponderada por pixels das partes
    trees["pred_class"], trees["pred_name"], trees["confidence"] = pc, pn, conf
    trees["seg_prob_mean"] = segp
    for k, n in enumerate(class_names):
        trees[f"prob_{n}"] = [float(v[k]) for v in agg_probs]
    trees.insert(0, "tree_id", range(1, len(trees) + 1))
    trees["lon"] = [c.x for c in trees["centroid"]]
    trees["lat"] = [c.y for c in trees["centroid"]]
    props = [c for c in trees.columns if c not in ("geometry", "centroid")]
    tg.write_geojson(out / "copas.geojson", trees, "geometry", props)
    tg.write_geojson(out / "arvores.geojson", trees, "centroid", props)
    trees[props].to_csv(out / "arvores.csv", index=False)
    counts = trees.groupby("pred_name").agg(n_arvores=("tree_id", "size"), area_m2_total=("area_m2", "sum"),
                                            confianca_media=("confidence", "mean")).reset_index()
    counts.to_csv(out / "contagem_por_classe.csv", index=False)
    summary = {**provenance,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "duration_s": time.perf_counter() - t_start,
               "n_tiles_nodata": len(nodata), "n_instances": int(len(inst)), "n_trees": int(len(trees)),
               "n_trees_merged_across_tiles": int((trees["n_parts"] > 1).sum()),
               "trees_by_class": dict(zip(counts["pred_name"], counts["n_arvores"].astype(int))),
               "note": ("Contagem e classes sao estimativas do modelo (ver avaliacao no teste); centroide da copa "
                        "vista de cima, nao do tronco; precisao absoluta depende do GPS da ortofoto.")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_tiles", "n_tiles_nodata", "n_instances", "n_trees",
                                              "trees_by_class", "duration_s")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
