#!/usr/bin/env python3
"""Confronta os classificadores binarios por especie (bananeira, palmeira, mangueira) com o multiclasse.

Para cada especie-alvo, na tarefa "alvo contra todas as demais", nos 418 recortes de teste:
  A. multiclasse com a regra normal (argmax das 6 classes);
  B. multiclasse com limiar proprio para a especie, escolhido NA VALIDACAO (maior F1 de validacao em
     prob_<alvo>, grade 0,05..0,95): testa se o ganho do binario e so um ponto de decisao diferente;
  C. binario (regra normal do modelo: argmax de 2 classes).
Estatistica: bootstrap pareado por tile (B=2000, semente 42) para as diferencas de F1, AUC-ROC e AP;
teste exato de McNemar sobre os acertos da decisao (A x C e B x C). Matrizes 2x2 e origem dos falsos
positivos (classe original) sao salvas. Nada e retreinado e o teste nao e usado para escolher nada.
Saida: runs/analise_binarios_<data>/ (analise.json, metricas.csv, diferencas.csv, matrizes_2x2.csv,
falsos_positivos.csv, f1_binario_vs_multiclasse.png). Rodar a partir de lab_2026/.
"""
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

LAB = Path(__file__).resolve().parent
R = LAB / "runs"
MULTI = R / "classif_20261102_163825_efficientnet_anotados"
B, SEED = 2000, 42
NAMES = {2: "palmeira", 4: "mangueira", 5: "bananeira"}
LABEL = {"A": "Multiclasse (argmax)", "B": "Multiclasse (limiar da validação)", "C": "Binário"}


def find_binary(t):
    runs = sorted(d for d in R.glob(f"classif_*_efficientnet_binaria_{NAMES[t]}") if (d / "test_predictions.csv").exists())
    if not runs:
        return None
    return runs[-1]


def load(t, part, b_dir):
    m = pd.read_csv(MULTI / f"{part}_predictions.csv")
    b = pd.read_csv(b_dir / f"{part}_predictions.csv")
    if set(m.crop_path) != set(b.crop_path):
        raise SystemExit(f"{NAMES[t]}/{part}: recortes diferentes entre os runs.")
    j = m.merge(b[["crop_path", "pred_species", f"prob_{NAMES[t]}"]], on="crop_path", suffixes=("", "_bin"))
    j["y"] = (j.species == t).astype(int)
    j["pA"] = (j.pred_species == t).astype(int)
    j["sA"] = j[f"prob_{NAMES[t]}"]
    j["pC"] = (j.pred_species_bin == t).astype(int)
    j["sC"] = j[f"prob_{NAMES[t]}_bin"]
    return j


def metrics(y, p, s):
    tp, fp, fn = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum())
    return {"precisao": precision_score(y, p, zero_division=0), "recall": recall_score(y, p, zero_division=0),
            "f1": f1_score(y, p, zero_division=0), "auc_roc": roc_auc_score(y, s), "ap": average_precision_score(y, s),
            "tp": tp, "fp": fp, "fn": fn, "tn": int(len(y) - tp - fp - fn)}


def mcnemar(y, p1, p2):
    a, b = (p1 == y) & (p2 != y), (p1 != y) & (p2 == y)
    n01, n10 = int(a.sum()), int(b.sum())
    pv = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"so_1_acerta": n01, "so_2_acerta": n10, "p_valor": float(pv)}


def main():
    out = R / f"analise_binarios_{date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    rows, diffs, mats, fps, summary = [], [], [], [], {}
    for t, name in NAMES.items():
        b_dir = find_binary(t)
        if b_dir is None:
            print(f"{name}: run binario ainda nao existe; pulando.")
            continue
        cfg = json.loads((b_dir / "config.json").read_text())
        val, te = load(t, "val", b_dir), load(t, "test", b_dir)
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
        f1v = [f1_score(val.y, (val.sA >= g).astype(int), zero_division=0) for g in grid]
        thr = float(grid[int(np.argmax(f1v))])
        te["pB"], te["sB"] = (te.sA >= thr).astype(int), te.sA
        res = {k: metrics(te.y.to_numpy(), te[f"p{k}"].to_numpy(), te[f"s{k}"].to_numpy()) for k in "ABC"}
        for k in "ABC":
            rows.append({"especie": name, "modelo": LABEL[k], **res[k]})
            mats.append({"especie": name, "modelo": LABEL[k], "real_outra_pred_outra": res[k]["tn"],
                         "real_outra_pred_alvo": res[k]["fp"], "real_alvo_pred_outra": res[k]["fn"],
                         "real_alvo_pred_alvo": res[k]["tp"]})
            fp = te[(te[f"p{k}"] == 1) & (te.y == 0)].species.value_counts().to_dict()
            fps.append({"especie": name, "modelo": LABEL[k], **{f"classe_{c}": int(n) for c, n in fp.items()}})
        # bootstrap pareado por tile
        tiles = sorted(te.tile.unique())
        g = {tt: d for tt, d in te.groupby("tile")}
        rng = np.random.default_rng(SEED)
        bs = {f"{a}-{c}": {"f1": [], "auc_roc": [], "ap": []} for a, c in (("C", "A"), ("C", "B"))}
        ci_f1 = {k: [] for k in "ABC"}
        for _ in range(B):
            d = pd.concat([g[tiles[i]] for i in rng.integers(0, len(tiles), len(tiles))])
            y = d.y.to_numpy()
            if y.min() == y.max():
                continue
            mm = {k: (f1_score(y, d[f"p{k}"], zero_division=0), roc_auc_score(y, d[f"s{k}"]),
                      average_precision_score(y, d[f"s{k}"])) for k in "ABC"}
            for k in "ABC":
                ci_f1[k].append(mm[k][0])
            for a, c in (("C", "A"), ("C", "B")):
                for i, met in enumerate(("f1", "auc_roc", "ap")):
                    bs[f"{a}-{c}"][met].append(mm[a][i] - mm[c][i])
        for k in "ABC":
            lo, hi = np.percentile(ci_f1[k], [2.5, 97.5])
            rows[-3 + "ABC".index(k)].update({"f1_ic95_inf": lo, "f1_ic95_sup": hi})
        for pair, mets in bs.items():
            a, c = pair.split("-")
            for met, v in mets.items():
                lo, hi = np.percentile(v, [2.5, 97.5])
                diffs.append({"especie": name, "comparacao": f"{LABEL[a]} - {LABEL[c]}", "metrica": met,
                              "diferenca": res[a][met] - res[c][met], "ic95_inf": lo, "ic95_sup": hi,
                              "distinguivel": bool(lo > 0 or hi < 0)})
        summary[name] = {"run_binario": b_dir.name, "val_f1_macro_binario": cfg.get("best_val_f1_macro"),
                         "epoca_binario": cfg.get("effective_epoch"), "positivos_teste": int(te.y.sum()),
                         "limiar_multiclasse_val": thr,
                         "f1_val_multiclasse_no_limiar": float(max(f1v)),
                         "mcnemar_A_vs_C": mcnemar(te.y, te.pA, te.pC), "mcnemar_B_vs_C": mcnemar(te.y, te.pB, te.pC)}
    met = pd.DataFrame(rows)
    met.to_csv(out / "metricas.csv", index=False)
    pd.DataFrame(diffs).to_csv(out / "diferencas.csv", index=False)
    pd.DataFrame(mats).to_csv(out / "matrizes_2x2.csv", index=False)
    pd.DataFrame(fps).fillna(0).to_csv(out / "falsos_positivos.csv", index=False)
    (out / "analise.json").write_text(json.dumps({"B": B, "seed": SEED, "especies": summary}, indent=2,
                                                 ensure_ascii=False, default=float), encoding="utf-8")
    # figura: F1 com IC por especie e modelo
    esp = list(dict.fromkeys(met.especie))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.26
    cores = {"A": "#9bb7d4", "B": "#4f7cac", "C": "#d9822b"}
    for i, k in enumerate("ABC"):
        d = met[met.modelo == LABEL[k]].set_index("especie").loc[esp]
        x = np.arange(len(esp)) + (i - 1) * w
        ax.bar(x, d.f1, w, color=cores[k], label=LABEL[k])
        ax.errorbar(x, d.f1, yerr=[d.f1 - d.f1_ic95_inf, d.f1_ic95_sup - d.f1], fmt="none", ecolor="#333", capsize=3, lw=1)
    ax.set_xticks(np.arange(len(esp)), [e.capitalize() for e in esp])
    ax.set_ylabel("F1 no teste (alvo contra as demais)")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "f1_binario_vs_multiclasse.png", dpi=200)
    pd.set_option("display.width", 220)
    print(met.round(3).to_string())
    print(pd.DataFrame(diffs).round(3).to_string())
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
