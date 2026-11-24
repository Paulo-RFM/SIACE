#!/usr/bin/env python3
"""Gera METRICAS_MODELOS.md: métricas dos modelos usados no SIACE, lidas dos arquivos dos runs (2026-11-23).

Nenhum número é digitado à mão: cada tabela vem de um CSV/JSON em runs/ (ou da pasta de João Pessoa), e o
arquivo de origem é citado abaixo de cada tabela. Os modelos treinados antes da correção do alinhamento da
augmentation não entram.
Uso: python gerar_metricas_modelos.py
"""
import csv
import json
from pathlib import Path

LAB = Path(__file__).resolve().parent
R = LAB / "runs"
JP = Path("/home/paulomaciel/proj_JoaoPessoa211126")
OUT = LAB / "METRICAS_MODELOS.md"


def rcsv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def rjson(p):
    return json.loads(Path(p).read_text())


def f3(x):
    return f"{float(x):.3f}"


def pct(x, d=1):
    return f"{100 * float(x):.{d}f}%"


def tab(cab, linhas):
    s = "| " + " | ".join(cab) + " |\n|" + "---|" * len(cab) + "\n"
    return s + "".join("| " + " | ".join(map(str, l)) + " |\n" for l in linhas)


def fonte(*ps):
    return "Fonte: " + ", ".join(f"`{p}`" for p in ps) + "\n"


def main():
    md = ["# Métricas dos modelos do SIACE\n",
          "Gerado por `gerar_metricas_modelos.py` a partir dos arquivos dos runs; cada tabela cita o arquivo de origem. "
          "Divisão espacial por componentes conexas de tiles: 611 tiles de treino, 203 de validação e 204 de teste "
          "(1.018 tiles anotados, 2.113 copas). Limiar, checkpoint e parâmetros escolhidos na validação; teste avaliado "
          "uma vez por modelo. IC 95% por bootstrap por tile.\n",
          "## Sistema final\n",
          "| Etapa | Modelo | Run | Arquivo do modelo |\n|---|---|---|---|\n"
          "| Segmentação de copas | SegFormer MiT-B1, limiar 0,40 | `runs/arq_segformer_mit_b1_20261110` | `checkpoint_best.pth` |\n"
          "| Individualização | Watershed h-maxima (h = 0,75 m, área mínima 3000 px) | `runs/individualizacao_val_segformer_20261111` | `frozen_instance_params_hmax.json` |\n"
          "| Classificação de espécies | EfficientNet-B0 (recortes anotados) | `runs/classif_20261102_163825_efficientnet_anotados` | `checkpoint_best.pth` |\n"
          "| Variante para imagens de 8 cm | SegFormer MiT-B1 treinada a 8 cm | `runs/res_segformer_8cm_20261121` | `checkpoint_best.pth` |\n"]

    # 1. arquiteturas
    p = "runs/arquiteturas_comparacao_20261110/resultados.csv"
    rows = rcsv(LAB / p)
    md.append("## 1. Segmentação binária: comparação de arquiteturas (semente 42, teste)\n")
    md.append(tab(["Modelo", "Parâm.", "Limiar", "IoU val.", "IoU teste [IC 95%]", "F1", "Precisão", "Recall",
                   "Dif. IoU vs U-Net [IC 95%]", "Inferência (s/tile)"],
                  [[r["modelo"], f"{int(r['parametros']):,}".replace(",", "."), r["limiar"], f3(r["iou_val"]),
                    f"{f3(r['iou'])} [{f3(r['iou_ic95_inf'])}; {f3(r['iou_ic95_sup'])}]", f3(r["dice_f1"]),
                    f3(r["precision"]), f3(r["recall"]),
                    "—" if not r["dif_iou_vs_unet"] else
                    f"{float(r['dif_iou_vs_unet']):+.3f} [{float(r['dif_ic95_inf']):+.3f}; {float(r['dif_ic95_sup']):+.3f}]",
                    f"{float(r['inferencia_s_por_tile']):.3f}" if r["inferencia_s_por_tile"] else "—"] for r in rows]))
    md.append("U-Net (ResNet-34) é o modelo experimentado de referência; a SegFormer MiT-B1 é o modelo do sistema.\n")
    md.append(fonte(p, "runs/arquiteturas_comparacao_20261110/ic_95.csv"))

    # 2. sementes
    p1, p2, p3 = ("runs/seeds_segformer_comparacao_20261110/resultados_por_run.csv",
                  "runs/seeds_segformer_comparacao_20261110/resumo.csv",
                  "runs/seeds_segformer_comparacao_20261110/decisao.txt")
    md.append("## 2. SegFormer MiT-B1 x U-Net ResNet-34 em quatro sementes\n")
    md.append(tab(["Modelo", "Semente", "Limiar", "IoU val.", "IoU teste", "F1", "Precisão", "Recall"],
                  [[r["modelo"], r["semente"], r["limiar"], f3(r["iou_val"]), f3(r["iou"]), f3(r["dice_f1"]),
                    f3(r["precision"]), f3(r["recall"])] for r in rcsv(LAB / p1)]))
    md.append("\n")
    md.append(tab(["Métrica", "U-Net (média ± dp)", "SegFormer (média ± dp)", "Diferença", "p (permutação)",
                   "Sementes com SegFormer melhor"],
                  [[r["metrica"], f"{f3(r['media_unet'])} ± {f3(r['dp_unet'])}",
                    f"{f3(r['media_segformer'])} ± {f3(r['dp_segformer'])}", f"{float(r['dif_medias']):+.3f}",
                    f3(r["p_permutacao_bicaudal"]), r["sementes_segformer_melhor"]]
                   for r in rcsv(LAB / p2) if r["metrica"] in ("iou_val", "iou", "dice_f1", "precision", "recall")]))
    md.append("\nCritério de troca, fixado antes dos treinos: IoU médio de validação **e** de teste maiores nas quatro "
              "sementes. O teste, portanto, participou da escolha; a validação sozinha (4/4 sementes) levaria à mesma "
              "decisão.\n\n```\n" + (LAB / p3).read_text().strip() + "\n```\n")
    md.append(fonte(p1, p2, p3))

    # 3. resolução
    p = "runs/resolucao_comparacao_20261121/ic_95.csv"
    md.append("## 3. Resolução da imagem (SegFormer treinada e testada a 3, 8 e 15 cm)\n")
    md.append(tab(["Comparação", "Métrica", "Estimativa [IC 95%]"],
                  [[r["comparacao"], r["metrica"],
                    f"{float(r['estimativa']):+.3f} [{float(r['ic95_inf']):+.3f}; {float(r['ic95_sup']):+.3f}]"
                    if "dif" in r["metrica"] else f"{f3(r['estimativa'])} [{f3(r['ic95_inf'])}; {f3(r['ic95_sup'])}]"]
                   for r in rcsv(LAB / p)]))
    md.append(fonte(p))

    # 4. individualização
    pj = "runs/individualizacao_val_segformer_20261111/frozen_instance_params_hmax.json"
    j = rjson(LAB / pj)
    md.append("## 4. Individualização das copas (validação, máscara predita pela SegFormer)\n")
    md.append(f"Escolhido: Watershed h-maxima, h = {j['h_m']} m, área mínima {j['min_area']} px. "
              f"F1 de detecção na validação (IoU ≥ 0,5): {f3(j['f1_val'])}, contra {f3(j['f1_val_componentes_conexas'])} "
              f"das componentes conexas ({j['selected_on']}).\n\n")
    md.append(fonte(pj, "runs/individualizacao_val_segformer_20261111/resultado.csv"))

    # 5. ponta a ponta no teste
    md.append("## 5. Detecção de árvores de ponta a ponta no teste (204 tiles, 420 árvores, IoU ≥ 0,5)\n")
    linhas, fontes = [], []
    for nome, d in [("Componentes conexas", "pipeline_test_segformer_20261111"),
                    ("Watershed de janela fixa", "pipeline_test_ws_segformer_20261111"),
                    ("Watershed h-maxima (sistema)", "pipeline_test_hmax_segformer_20261111")]:
        p = f"runs/{d}/avaliacao_teste/metricas_teste.json"
        m = rjson(LAB / p)["deteccao"]
        linhas.append([nome, m["n_instances"], m["tp"], f3(m["precision"]), f3(m["recall"]), f3(m["f1"]),
                       f"{m['instances_per_gt']:.2f}"])
        fontes.append(p)
    md.append(tab(["Método", "Instâncias", "Acertos", "Precisão", "Recall", "F1", "Instâncias por árvore"], linhas))
    md.append(fonte(*fontes))
    m = rjson(LAB / fontes[-1])["classificacao_nas_arvores_detectadas"]
    md.append(f"\nClassificação nas {m['number_of_samples']} árvores detectadas corretamente pelo sistema: acurácia "
              f"{pct(m['accuracy'])}, F1-macro {f3(m['f1_macro'])}, acurácia balanceada {f3(m['balanced_accuracy'])}, "
              f"kappa {f3(m['cohen_kappa'])}.\n")
    md.append(fonte(fontes[-1]))

    # 6. classificador
    p = "runs/classif_20261102_163825_efficientnet_anotados/test_metrics_global.json"
    m = rjson(LAB / p)
    md.append("## 6. Classificação de espécies: EfficientNet-B0 (teste, recortes anotados)\n")
    md.append(tab(["N", "Acurácia", "Acurácia balanceada", "F1-macro", "F1 ponderado", "Kappa", "ECE", "Melhor época"],
                  [[m["number_of_samples"], f3(m["accuracy"]), f3(m["balanced_accuracy"]), f3(m["f1_macro"]),
                    f3(m["f1_weighted"]), f3(m["cohen_kappa"]), f3(m["ece"]), m["best_epoch"]]]))
    pr = "runs/classif_20261102_163825_efficientnet_anotados/test_classification_report.txt"
    md.append("\nPor classe:\n\n```\n" + (LAB / pr).read_text().rstrip() + "\n```\n")
    md.append(fonte(p, pr))
    ps = "runs/classif_seeds_efficientnet_20261120/resultados_por_semente.csv"
    md.append("\nVariação entre sementes (variante do sistema = `anotados`; `aug_treino` = variante com aumento offline):\n\n")
    md.append(tab(["Variante", "Semente", "F1-macro val.", "F1-macro teste", "Acurácia bal. teste", "Acurácia teste"],
                  [[r["variante"], r["seed"], f3(r["val_f1"]), f3(r["test_f1"]), f3(r["test_bacc"]), f3(r["test_acc"])]
                   for r in rcsv(LAB / ps)]))
    pj = "runs/classif_seeds_efficientnet_20261120/resumo.json"
    md.append("\nMédia ± dp (4 sementes) e p de permutação:\n\n```\n" + (LAB / pj).read_text().strip() + "\n```\n")
    md.append(fonte(ps, pj))

    # 7. aplicação em Rio Paranaíba
    s = rjson(R / "pipeline_cidade_segformer_20261111/summary.json")
    c = rjson(R / "pipeline_perimetro_segformer_20261121/contagem_perimetro.json")
    md.append("## 7. Aplicação ao ortomosaico de Rio Paranaíba\n")
    md.append(f"- Tiles: {s['n_tiles']} ({s['n_tiles_nodata']} sem imagem); tempo: {s['duration_s'] / 60:.0f} min.\n"
              f"- Árvores no ortomosaico inteiro (inclui área rural): {s['n_trees']}; "
              f"no perímetro urbano ({c['area_ha_perimetro']} ha): {c['n_arvores_perimetro']}.\n"
              f"- Perímetro urbano por classe: " + ", ".join(f"{k} {v}" for k, v in c["por_classe"].items()) + ".\n")
    md.append(fonte("runs/pipeline_cidade_segformer_20261111/summary.json",
                    "runs/pipeline_perimetro_segformer_20261121/contagem_perimetro.json"))
    m = rjson(R / "conferencia_100_arvores_20261121/metricas_conferencia.json")
    d, k = m["deteccao"]["excl"], m["classe"]["excl"]
    md.append(f"\nConferência visual de 100 árvores (amostra estratificada, um avaliador): detecção correta "
              f"{pct(d['cidade_ponderada'])} (IC 95% {pct(d['ic95'][0])} a {pct(d['ic95'][1])}); classe correta "
              f"{pct(k['cidade_ponderada'])} (IC 95% {pct(k['ic95'][0])} a {pct(k['ic95'][1])}).\n")
    md.append(fonte("runs/conferencia_100_arvores_20261121/metricas_conferencia.json"))
    m = rjson(R / "cobertura_dossel_20261120/resumo.json")
    v = rjson(R / "cobertura_dossel_20261120/validacao_teste.json")
    md.append(f"\nCobertura de copa no perímetro urbano: {m['cobertura_area_observada_pct']:.1f}% "
              f"({m['area_copa_ha']:.1f} de {m['area_perimetro_ha']:.1f} ha); {m['n_celulas_exibidas']} células de "
              f"100 m avaliadas; por faixa: " + ", ".join(f"{a}% {b}" for a, b in m["celulas_por_classe"].items()) + ".\n"
              f"Validação no teste: cobertura predita {v['cob_pred_total']:.2f}% contra {v['cob_ref_total']:.2f}% da "
              f"referência (superestimativa relativa de {pct(v['vies_relativo'])}, IC 95% {pct(v['vies_relativo_ic95'][0])} "
              f"a {pct(v['vies_relativo_ic95'][1])}); erro absoluto médio por tile {v['por_tile']['erro_absoluto_medio_pp']:.1f} "
              f"p.p.; correlação por tile {v['por_tile']['pearson_r']:.2f}.\n")
    md.append(fonte("runs/cobertura_dossel_20261120/resumo.json", "runs/cobertura_dossel_20261120/validacao_teste.json"))
    o = rjson(R / "osm_limiares_segformer_20261111/teste_resultado.json")
    md.append(f"\nFiltro para o OpenStreetMap (limiar fixado na validação), no teste: {o['copas_aprovadas']} de "
              f"{o['copas_detectadas']} copas aprovadas, precisão {pct(o['precisao_arvore'])} "
              f"(sem filtro: {pct(o['precisao_arvore_sem_filtro'])}).\n")
    md.append(fonte("runs/osm_limiares_segformer_20261111/teste_resultado.json"))

    # 8. João Pessoa
    md.append("## 8. Teste em João Pessoa (ortoimagem de 8 cm, 3,61 km²)\n")
    pc = JP / "resultados/joao_pessoa/contagens_joao_pessoa.csv"
    md.append(tab(["Modelo", "Tiles", "Árvores", "Área de copa (ha)", "Minutos de inferência"],
                  [[r["modelo"], r["tiles"], r["arvores"], r["area_copa_total_ha"], r["minutos"]] for r in rcsv(pc)]))
    m = rjson(JP / "conferencia_100/metricas_conferencia_jp.json")
    md.append(f"\nConferência visual de 100 árvores (amostra aleatória simples, semente 42): {m['e_arvore_s']} sim, "
              f"{m['e_arvore_n']} não, {m['sem_resposta']} sem resposta; precisão de detecção "
              f"{pct(m['precisao_deteccao'])} (IC 95% Wilson {pct(m['ic95'][0])} a {pct(m['ic95'][1])}); copas "
              f"fundidas em {m['copas_fundidas_s']} de {m['copas_fundidas_avaliadas']}.\n")
    c = rjson(JP / "resultados/joao_pessoa/cobertura/resumo.json")
    md.append(f"\nCobertura de copa da área imageada: {c['cobertura_pct']:.1f}% ({c['area_copa_ha']} de "
              f"{c['area_imageada_ha']} ha); {c['celulas_avaliadas']} células avaliadas, mediana {c['mediana_pct']}%.\n")
    md.append("\nFonte: pasta de João Pessoa (fora do repositório): `resultados/joao_pessoa/contagens_joao_pessoa.csv`, "
              "`conferencia_100/metricas_conferencia_jp.json`, `resultados/joao_pessoa/cobertura/resumo.json`. "
              "Os scripts estão em `joao_pessoa/`.\n")

    OUT.write_text("\n".join(md))
    print("saída:", OUT)


if __name__ == "__main__":
    main()
