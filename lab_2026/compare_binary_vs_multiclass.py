#!/usr/bin/env python3
"""Compara, para UMA classe, o classificador binario (--binary-target) com o multiclasse.

Usa as predicoes de TESTE ja salvas pelos dois runs (nao roda modelo nem reavalia o teste):
  * multiclasse: a classe-alvo contra as demais (um-contra-todos) a partir de pred_species/prob_<alvo>;
  * binario: pred_species/prob_<alvo> do run binario.
Exige que os dois runs tenham exatamente os mesmos recortes de teste.
Saida: <run_binario>/comparacao_vs_multiclasse.{json,csv}
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)


def metrics(y, pred, prob):
    return {"n": int(len(y)), "positivos": int(y.sum()),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "auc_roc": roc_auc_score(y, prob), "average_precision": average_precision_score(y, prob),
            "tp": int(((pred == 1) & (y == 1)).sum()), "fp": int(((pred == 1) & (y == 0)).sum()),
            "fn": int(((pred == 0) & (y == 1)).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary-run", required=True)
    ap.add_argument("--multiclass-run", default="runs/classif_20261102_163825_efficientnet_anotados")
    args = ap.parse_args()
    b_dir, m_dir = Path(args.binary_run), Path(args.multiclass_run)
    cfg = json.loads((b_dir / "config.json").read_text())
    t = cfg.get("binary_target")
    if t is None:
        raise SystemExit(f"{b_dir} nao e um run binario (config sem binary_target).")
    name = cfg["class_names"][1]
    b = pd.read_csv(b_dir / "test_predictions.csv")
    m = pd.read_csv(m_dir / "test_predictions.csv")
    if set(b.crop_path) != set(m.crop_path) or len(b) != len(m):
        raise SystemExit("Os runs nao tem o mesmo conjunto de teste; comparacao invalida.")
    j = m.merge(b, on="crop_path", suffixes=("_m", "_b"))
    y = (j["species_m"] == t).astype(int).to_numpy()
    res = {"classe_alvo": {"id": t, "nome": name}, "binary_run": str(b_dir), "multiclass_run": str(m_dir),
           "multiclasse_um_contra_todos": metrics(y, (j["pred_species_m"] == t).astype(int).to_numpy(),
                                                  j[f"prob_{name}_m"].to_numpy()),
           "binario": metrics(y, (j["pred_species_b"] == t).astype(int).to_numpy(), j[f"prob_{name}_b"].to_numpy()),
           "selecao_binario": f"best_val_f1_macro={cfg.get('best_val_f1_macro')} (epoca {cfg.get('effective_epoch')})"}
    (b_dir / "comparacao_vs_multiclasse.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    tab = pd.DataFrame({"multiclasse": res["multiclasse_um_contra_todos"], "binario": res["binario"]})
    tab.to_csv(b_dir / "comparacao_vs_multiclasse.csv")
    print(f"Classe-alvo: {name} (id {t}) | teste: {len(y)} recortes, {int(y.sum())} positivos")
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
