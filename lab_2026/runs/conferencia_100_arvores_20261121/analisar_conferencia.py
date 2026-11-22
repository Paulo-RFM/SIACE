#!/usr/bin/env python3
"""Metricas da conferencia manual das 100 arvores da area urbana (2026-11-21).

Entrada: amostra_100_preencher.csv (amostra original, semente 42) + respostas_transcritas_qgis.csv
(respostas do autor, transcritas da tabela de atributos do QGIS colada na conversa).
* Precisao de deteccao = P(e_arvore = s); '?' tratado de 3 formas: excluido, como n (pior caso), como s (melhor caso).
* Acerto de classe = P(classe_correta = s), sobre todas as deteccoes e sobre as confirmadas como arvore.
* Cidade: estimativa estratificada, peso = estrato_N (arvores da cidade em cada classe predita).
* IC 95%: Wilson por estrato; cidade por bootstrap estratificado (B = 2000, semente 42).
* Copas fundidas: so as linhas em que o autor marcou s/n (contagem descritiva, sem reponderacao).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
a = pd.read_csv(H / "amostra_100_preencher.csv").drop(columns=["e_arvore", "classe_correta", "classe_observada",
                                                                "copas_fundidas", "deslocamento_m", "referencia", "obs"])
r = pd.read_csv(H / "respostas_transcritas_qgis.csv", dtype=str)
r["amostra_id"] = r.amostra_id.astype(int)
d = a.merge(r, on="amostra_id", validate="1:1")
assert len(d) == 100
NOMES = {"arvore_sem_identificacao": "Sem identificação", "palmeira": "Palmeira", "mangueira": "Mangueira",
         "arvore_seca": "Árvore seca", "bananeira": "Bananeira", "jabuticabeira": "Jabuticabeira"}
rng = np.random.default_rng(42)
B = 2000


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    z, p = 1.959964, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def estratificada(df, col, modo):
    """proporcao da cidade; modo: 'excl' (? fora), 'pior' (?=n), 'melhor' (?=s)"""
    tot, num = 0.0, 0.0
    for cls, g in df.groupby("pred_name"):
        v = g[col]
        if modo == "excl":
            v = v[v != "?"]
        x = (v == "s") if modo != "melhor" else v.isin(["s", "?"])
        if len(x) == 0:
            continue
        N = g.estrato_N.iloc[0]
        tot += N
        num += N * x.mean()
    return num / tot


def boot(df, col, modo):
    gs = [g for _, g in df.groupby("pred_name")]
    out = []
    for _ in range(B):
        bs = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in gs])
        out.append(estratificada(bs, col, modo))
    return np.percentile(out, [2.5, 97.5])


linhas = []
for cls in NOMES:
    g = d[d.pred_name == cls]
    n = len(g)
    ns, nn, nq = (g.e_arvore == "s").sum(), (g.e_arvore == "n").sum(), (g.e_arvore == "?").sum()
    cs, cn, cq = (g.classe_correta == "s").sum(), (g.classe_correta == "n").sum(), (g.classe_correta == "?").sum()
    conf = g[g.e_arvore == "s"]
    lo, hi = wilson(ns, ns + nn)
    clo, chi = wilson(cs, cs + cn)
    linhas.append(dict(classe=cls, nome=NOMES[cls], N_cidade=int(g.estrato_N.iloc[0]), n=n, arvore_s=int(ns),
                       arvore_n=int(nn), arvore_duvida=int(nq), precisao_det=ns / (ns + nn), det_ic_inf=lo,
                       det_ic_sup=hi, classe_s=int(cs), classe_n=int(cn), classe_duvida=int(cq),
                       acerto_classe=cs / (cs + cn), classe_ic_inf=clo, classe_ic_sup=chi,
                       acerto_classe_entre_arvores=(conf.classe_correta == "s").mean(),
                       n_arvores_confirmadas=len(conf)))
t = pd.DataFrame(linhas)
t.to_csv(H / "metricas_por_classe.csv", index=False)

res = {"n_amostra": 100, "B_bootstrap": B, "semente": 42, "deteccao": {}, "classe": {}}
for col, chave in [("e_arvore", "deteccao"), ("classe_correta", "classe")]:
    for modo in ["excl", "pior", "melhor"]:
        p = estratificada(d, col, modo)
        lo, hi = boot(d, col, modo)
        res[chave][modo] = {"cidade_ponderada": p, "ic95": [lo, hi]}
    v = d[col]
    res[chave]["amostra_bruta"] = {"s": int((v == "s").sum()), "n": int((v == "n").sum()), "?": int((v == "?").sum())}
# acerto de classe entre as arvores confirmadas, ponderado
conf = d[d.e_arvore == "s"]
res["classe"]["entre_arvores_confirmadas"] = {"cidade_ponderada": estratificada(conf, "classe_correta", "excl"),
                                              "ic95": list(boot(conf, "classe_correta", "excl")), "n": len(conf)}
cf = d[d.copas_fundidas.isin(["s", "n"])]
res["copas_fundidas"] = {"marcadas": len(cf), "s": int((cf.copas_fundidas == "s").sum()),
                         "n": int((cf.copas_fundidas == "n").sum()),
                         "area_mediana_m2_s": float(cf[cf.copas_fundidas == "s"].area_m2.median()),
                         "area_mediana_m2_n": float(cf[cf.copas_fundidas == "n"].area_m2.median()),
                         "por_classe": cf.groupby(["pred_name", "copas_fundidas"]).size().unstack(fill_value=0).to_dict()}
# confianca do classificador x acerto
res["confianca_mediana"] = {k: float(d[d.classe_correta == k].confidence.median()) for k in ["s", "n", "?"]}
res["area_mediana_m2_por_e_arvore"] = {k: float(d[d.e_arvore == k].area_m2.median()) for k in ["s", "n", "?"]}
(H / "metricas_conferencia.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
pd.set_option("display.width", 250)
print(t.round(3).to_string(index=False))
print(json.dumps(res, indent=2, ensure_ascii=False, default=lambda x: round(float(x), 4)))
