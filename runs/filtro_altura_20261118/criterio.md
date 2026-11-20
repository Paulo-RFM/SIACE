# Filtro de altura para falsas árvores — critério fixado antes dos resultados (2026-11-18)

Fonte da altura: `/home/paulomaciel/proj/arvores com altura.geojson` (coluna `alt_1`, altura MDS − MDT no ponto
central de cada árvore da cidade `runs/pipeline_cidade_segformer_20261111`, calculada pelo autor fora do pipeline).
As instâncias de `pipeline_val_hmax_segformer_20261111` e `pipeline_test_hmax_segformer_20261111` são idênticas
às da cidade (mesmo tile, instance_id e n_pixels: 390/390 e 355/355); cada instância recebe a altura da árvore
da cidade que a contém (árvores unidas entre tiles: a mesma altura para todas as partes).

Regra: remove a copa se alt_1 < h. Altura ausente (NaN): a copa é mantida.
Candidatos: sem filtro e h = 0,25; 0,50; ...; 3,00 m.
Escolha na VALIDAÇÃO (203 tiles): maior F1 de detecção (IoU >= 0,5, casamento húngaro, mesmo código do
avaliador). Empate: o menor h. Se nenhum h superar "sem filtro", o filtro não é adotado.
Teste: avaliado uma única vez com o h escolhido; IC 95% por bootstrap por tile (B = 2000, semente 42) do F1 e
da diferença pareada com/sem filtro.

---
# Adendo (2026-11-18, antes de qualquer resultado com altura no OSM): altura no filtro do OpenStreetMap

O filtro de F1 acima não foi adotado. Como o envio ao OSM prioriza a precisão, testa-se a altura no filtro do OSM,
mantendo a regra já usada (runs/osm_limiares_segformer_20261111, osm_calibrar_limiares.py):
- mesmas copas e rótulos (copas_val.csv / copas_teste.csv: "é árvore" = ponto da copa dentro de polígono anotado);
- mesmo filtro de área (≤ 217,39 m²) e mesma meta: precisão "é árvore" ≥ 0,95 na VALIDAÇÃO;
- grade conjunta: s (seg_prob_mean) de 0,50 a 0,98 (passo 0,01) × h ∈ {sem filtro, 0,25, 0,50, ..., 3,00 m};
  copa aprovada se área ≤ limite, seg_prob_mean ≥ s e (altura ≥ h ou altura ausente);
- escolha: entre os pares com precisão de validação ≥ 0,95, o que aprova MAIS copas na validação (mesma lógica do
  "menor s que atinge a meta"); empate: maior precisão, depois menor h, depois menor s;
- a altura é adotada só se o par escolhido tiver h definido (isto é, se aprovar mais copas que o melhor sem altura);
- teste avaliado uma única vez com o par escolhido; relatar também o par atual (s = 0,91, sem altura) no teste,
  que já era conhecido.
