#!/usr/bin/env python3
"""Resultado da conferência de 100 árvores de João Pessoa (amostra_100.gpkg, camada pontos; leitura só).
Precisão de detecção = fração de copas que são árvores reais, com IC 95% de Wilson; linhas sem resposta
são excluídas (e, como sensibilidade, contadas como erro). Saída: metricas_conferencia_jp.json."""
import json
import math
import sqlite3
from pathlib import Path

import statistics as st

HERE = Path(__file__).resolve().parent
Z = 1.959964


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return p, c - h, c + h


con = sqlite3.connect(f"file:{HERE / 'amostra_100.gpkg'}?mode=ro", uri=True)
rows = con.execute("select amostra_id, area_m2, e_arvore, copas_fundidas from pontos").fetchall()
norm = lambda v: (v or "").strip().lower()  # noqa: E731
resp = [r for r in rows if norm(r[2]) in ("s", "n")]
s = sum(norm(r[2]) == "s" for r in resp)
p, lo, hi = wilson(s, len(resp))
p_pior = s / len(rows)
fund = [r for r in resp if norm(r[2]) == "s" and norm(r[3]) in ("s", "n")]
nf = sum(norm(r[3]) == "s" for r in fund)
pf, lof, hif = wilson(nf, len(fund))
area = {k: st.median([r[1] for r in fund if norm(r[3]) == k]) for k in ("s", "n")}
area_n = [r[1] for r in resp if norm(r[2]) == "n"]
out = {"amostra": len(rows), "respondidas": len(resp), "sem_resposta": len(rows) - len(resp),
       "e_arvore_s": s, "e_arvore_n": len(resp) - s,
       "precisao_deteccao": p, "ic95": [lo, hi], "precisao_sem_resposta_como_erro": p_pior,
       "copas_fundidas_s": nf, "copas_fundidas_avaliadas": len(fund), "frac_fundidas": pf, "frac_fundidas_ic95": [lof, hif],
       "area_mediana_m2_fundidas": area["s"], "area_mediana_m2_individuais": area["n"],
       "area_mediana_m2_nao_arvore": st.median(area_n) if area_n else None}
(HERE / "metricas_conferencia_jp.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
