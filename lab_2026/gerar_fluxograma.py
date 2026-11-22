"""Gera o fluxograma das etapas do projeto (fluxograma_projeto.png na pasta do LaTeX).

Reflete o que foi executado: somente imagens aereas; envio ao OSM feito no servidor de testes (2026-11-05).
Imagens de rua aparecem tracejadas como trabalho futuro.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "LaTex" / "TCC2___Paulo_Maciel (1)" / "fluxograma_projeto.png"

C_DADOS = "#dbe9f6"     # aquisicao e dados
C_TREINO = "#e3f1df"    # treinamento e avaliacao
C_APLIC = "#fbe8d3"     # aplicacao na cidade
C_SAIDA = "#f6d7d7"     # produto final
C_FUT = "#ffffff"       # trabalhos futuros
W, H = 3.0, 0.78


def box(ax, x, y, text, color, w=W, h=H, dashed=False, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec="#555555", lw=1.1, ls="--" if dashed else "-"))
    ax.text(x, y, text, ha="center", va="center", fontsize=9.2, wrap=True,
            fontweight="bold" if bold else "normal", color="#777777" if dashed else "black")
    return (x, y, w, h)


def arrow(ax, a, b, side_a="b", side_b="t", dashed=False, text=None, rad=0.0):
    def pt(bx, side):
        x, y, w, h = bx
        return {"b": (x, y - h / 2), "t": (x, y + h / 2), "l": (x - w / 2, y), "r": (x + w / 2, y)}[side]
    pa, pb = pt(a, side_a), pt(b, side_b)
    ax.annotate("", xy=pb, xytext=pa, arrowprops=dict(arrowstyle="-|>", color="#777777" if dashed else "#333333",
                lw=1.2, ls="--" if dashed else "-", shrinkA=1, shrinkB=1, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2 + 0.12, text, ha="center", va="bottom", fontsize=8,
                style="italic", color="#444444")


def main():
    fig, ax = plt.subplots(figsize=(13, 11.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 11.6); ax.axis("off")

    # faixas
    for x0, x1, lab, col in ((0.15, 7.55, "Preparação, treinamento e avaliação", C_TREINO),
                             (7.75, 12.85, "Aplicação à área urbana", C_APLIC)):
        ax.add_patch(FancyBboxPatch((x0, 0.2), x1 - x0, 8.55, boxstyle="round,pad=0.02,rounding_size=0.2",
                                    fc=col, ec="none", alpha=0.35))
        ax.text((x0 + x1) / 2, 8.5, lab, ha="center", va="center", fontsize=11, fontweight="bold", color="#333333")

    # aquisicao (topo)
    a1 = box(ax, 2.0, 10.6, "Voo do VANT\n(câmera RGB, 200 m)", C_DADOS)
    a2 = box(ax, 6.5, 10.6, "Ortomosaico RGB\n(ortorretificado)", C_DADOS)
    a3 = box(ax, 11.0, 10.6, "Recorte em 61.426 tiles\nGeoTIFF de 512 × 512 px", C_DADOS)
    arrow(ax, a1, a2, "r", "l"); arrow(ax, a2, a3, "r", "l")

    # treinamento (esquerda)
    t1 = box(ax, 3.85, 7.75, "Seleção de 1.018 tiles\ncom boa qualidade de imagem", C_DADOS)
    t2 = box(ax, 3.85, 6.65, "Anotação manual das copas (LabelMe)\npolígonos + classe da espécie", C_DADOS, w=4.0)
    t3 = box(ax, 3.85, 5.55, "Divisão espacial por componentes de tiles\ntreino 60% · validação 20% · teste 20%", C_DADOS, w=4.4)
    t4 = box(ax, 1.9, 4.3, "Máscaras binárias → U-Net, U-Net++,\nDeepLabV3+ e SegFormer; modelo e\nlimiar escolhidos na validação", C_TREINO, w=3.3, h=1.0)
    t5 = box(ax, 5.8, 4.3, "Recortes de copa\n→ EfficientNet-B0\n(6 classes)", C_TREINO, w=3.3, h=1.0)
    t6 = box(ax, 1.9, 2.85, "Escolha da individualização\nna validação (Watershed ×\ncomponentes conexas × h-maxima)", C_TREINO, w=3.3, h=1.0)
    t7 = box(ax, 3.85, 1.2, "Avaliação única no conjunto de teste\n(segmentação, detecção, espécie e ponta a ponta)",
             C_TREINO, w=5.6, h=0.9)
    arrow(ax, a3, t1, "b", "t", rad=-0.12)
    arrow(ax, t1, t2); arrow(ax, t2, t3)
    arrow(ax, t3, t4, "b", "t"); arrow(ax, t3, t5, "b", "t")
    arrow(ax, t4, t6)
    arrow(ax, t6, t7, "b", "t"); arrow(ax, t5, t7, "b", "t")

    # aplicacao (direita)
    p0 = box(ax, 10.3, 7.75, "Tiles com imagem\n(32.651)", C_DADOS)
    p1 = box(ax, 10.3, 6.65, "1. Segmentação das copas\n(SegFormer, limiar 0,40)", C_APLIC)
    p2 = box(ax, 10.3, 5.55, "2. Individualização das copas\n(Watershed h-maxima)", C_APLIC)
    p3 = box(ax, 10.3, 4.45, "3. Classificação de cada copa\n(EfficientNet-B0)", C_APLIC)
    p4 = box(ax, 10.3, 3.35, "4. Georreferenciamento e\nunião das copas entre tiles", C_APLIC)
    p5 = box(ax, 10.3, 2.15, "Mapa de árvores por espécie\n(GeoJSON / CSV)", C_SAIDA, bold=True)
    p6 = box(ax, 9.05, 0.85, "Envio ao OpenStreetMap\n(servidor de testes)", C_APLIC, w=2.35)
    p7 = box(ax, 11.55, 0.85, "Integração com imagens\nde nível de rua", C_FUT, w=2.35, dashed=True)
    ax.text(11.55, 0.3, "trabalho futuro", ha="center", fontsize=8, style="italic", color="#777777")
    arrow(ax, a3, p0, "b", "t")
    for a, b in ((p0, p1), (p1, p2), (p2, p3), (p3, p4), (p4, p5)):
        arrow(ax, a, b)
    x5, y5, w5, _ = p5
    ax.annotate("", xy=(9.05, 0.85 + H / 2), xytext=(x5 - 0.6, y5 - H / 2),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2))
    ax.annotate("", xy=(11.55, 0.85 + H / 2), xytext=(x5 + 0.6, y5 - H / 2),
                arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.2, ls="--"))

    # modelos treinados -> aplicacao: barramento vertical no espaco entre as faixas
    bus_x = 8.25
    st = dict(color="#666666", lw=1.3, ls="--")
    ax.plot([t5[0] + t5[2] / 2, bus_x], [t5[1], t5[1]], **st)
    ax.plot([bus_x, bus_x], [t5[1], p1[1]], **st)
    for tgt in (p1, p2, p3):
        ax.annotate("", xy=(tgt[0] - tgt[2] / 2, tgt[1]), xytext=(bus_x, tgt[1]),
                    arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.3, ls="--", shrinkA=0, shrinkB=1))
    ax.text(bus_x - 0.12, 5.0, "modelos e parâmetros\ncongelados", ha="right", va="center", fontsize=8,
            style="italic", color="#555555", rotation=90)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.1)
    print("salvo em", OUT)


if __name__ == "__main__":
    main()
