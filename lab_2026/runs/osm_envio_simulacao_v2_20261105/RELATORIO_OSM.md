# Envio em massa ao OpenStreetMap (servidor de testes) — 2026-11-05

## Arquivos
- `osm_calibrar_limiares.py`: escolhe os limiares na validação e mede o teste uma única vez. Saída em `runs/osm_limiares_v2_20261105/`; a v1, que não liberava nenhuma espécie, fica em `runs/osm_limiares_20261105/`.
- `osm_envio_dev.py`: seleciona as árvores, gera o osmChange e envia ao servidor `master.apis.dev.openstreetmap.org`, sem opção de produção. Por padrão só simula.
- `osm_especies_tags.json`: etiquetas de espécie sugeridas. Está com `confirmado_pelo_autor: false`, então nenhuma espécie é enviada.
- `tests/test_osm_envio.py`: 5 testes.
- `pipeline_cidade.py` agora grava `seg_prob_mean` (probabilidade média da U-Net ResNet-34 dentro da copa).
  - Run da cidade: `runs/pipeline_cidade_segprob_20261105`, com as mesmas 11.077 árvores e classes de `pipeline_cidade_cc_uniao_20261104`.

## Regras, todas fixadas na validação
1. **Área da copa ≤ 217,4 m²:** percentil 99 dos 1.296 polígonos do treino. Exclui maciços de vegetação.
2. **`seg_prob_mean` ≥ 0,94:** o menor limiar com precisão "é árvore" ≥ 0,95 na validação (0,951, 82 copas).
3. **Espécie:** confiança ≥ 0,70, mais precisão da classe ≥ 0,80 na validação com pelo menos 10 copas.
   - Só a bananeira passa: 0,80 com 20 copas.
   - Palmeira (0,50 com 10) e mangueira (0,44 com 27) vão só como `natural=tree`.

## Teste (uma avaliação)
- **Precisão "é árvore":** 0,988 (80 de 301 copas aprovadas), contra 0,847 sem filtro.
- **Custo:** só 31% das copas que estão sobre árvores reais sobrevivem.
- **Espécie bananeira:** 13 de 14 corretas em todas as copas; 3 de 3 no conjunto enviado.

## Cidade (simulação, nada enviado)
- **Total:** 755 de 11.077 árvores selecionadas (6,8%), em 2 lotes `.osc`.
- **Com espécie:** 8 elegíveis, todas bananeiras. Hoje vão 0 com espécie, porque as etiquetas não foram confirmadas.
- **Viés do filtro:** a mangueira passa de 7,2% das árvores da cidade para 23,8% da seleção, e a palmeira cai de 7,3% para 1,2%. Copas densas têm `seg_prob_mean` mais alto.
  - Medianas na validação por classe: 0,955 na mangueira contra 0,898 na palmeira.
  - Ou seja, o conjunto enviado não representa a composição da arborização.

  | Limiar `seg_prob_mean` | Árvores selecionadas na cidade |
  |---|---|
  | 0,90 | 2.721 |
  | 0,92 | 1.729 |
  | 0,94 (adotado) | 755 |

  Com 0,90, a precisão na validação fica em torno de 0,91.

## Interpretação
- **Sinal fraco:** `seg_prob_mean` separa pouco árvore de não-árvore. Exigir 95% de precisão custa cerca de 70% das árvores reais. Um sinal de certeza melhor, como um classificador "árvore × não-árvore" sobre o recorte, seria trabalho futuro.
- **Mesma lição da análise binária:** a certeza da espécie só é útil onde o modelo é confiável. Hoje isso vale só para a bananeira.

## Como enviar (servidor de testes)
1. Criar uma conta em https://master.apis.dev.openstreetmap.org.
2. Em "OAuth 2 applications", registrar um app com os escopos `read_prefs` e `write_api` (redirect `urn:ietf:wg:oauth:2.0:oob`) e gerar um token.
3. Revisar `osm_especies_tags.json` e trocar para `"confirmado_pelo_autor": true` se as etiquetas estiverem corretas (bananeira = `genus=Musa`).
4. Fazer um primeiro envio pequeno e conferir no site:
   ```
   export OSM_DEV_TOKEN=...
   python osm_envio_dev.py --cidade-run runs/pipeline_cidade_segprob_20261105 \
     --limiares runs/osm_limiares_v2_20261105/limiares_osm.json --enviar --limite 5
   ```
5. Repetir sem `--limite`. O registro `runs/osm_envios_dev_registro.csv` evita reenviar o que já foi. Antes de cada envio, o script descarta candidatas a menos de 3 m de uma árvore que já exista no servidor.

## Envio realizado (servidor de testes, 2026-11-05)
- **Nomes científicos:** confirmados pelo autor em `osm_especies_tags.json`.
- **Changesets:**
  - 719860: teste com 5 nós;
  - 719867: 500 nós;
  - 719868: 250 nós.
- **Total:** 755 nós `natural=tree`, sem duplicatas, todos registrados em `runs/osm_envios_dev_registro.csv`.
- **Espécie:** 8 nós levam `genus=Musa`. Conferido no servidor, por exemplo no nó 4365739586.
- **Decisão do autor (2026-11-05):** o limiar `seg_prob_mean` ≥ 0,94 fica como definitivo. A prioridade é a precisão (~95% na validação, 98,8% no teste), aceitando que menos árvores sejam enviadas (755 de 11.077).
- **Decisão do autor (2026-11-05): bananeiras não vão ao OSM.** A bananeira é uma planta herbácea perene, não uma árvore, então `natural=tree` estaria errado.
  - **Código:** `osm_envio_dev.py` ganhou `CLASSES_NAO_ARVORE = {5}`, e há um teste novo (43 testes, todos passam).
  - **Pontos já enviados:** os 12 pontos de bananeira no servidor de testes (8 com `genus=Musa`) são removidos com `osm_remover_dev.py`. As remoções ficam registradas em `runs/osm_remocoes_dev_registro.csv`.
  - **Envio que permanece:** 743 árvores, nenhuma com espécie, e a mangueira passa a 24,2% delas.
  - **Precisão no teste sem as bananeiras:** 98,7% (75 de 76 copas). Na validação, 0,961.
  - **Na monografia:** a bananeira continua como classe do inventário e da classificação.
- **Revisão da decisão (2026-11-05): bananeiras vão ao OSM como `natural=plant` + `genus=Musa`.** É a etiqueta genérica de planta, com status "in use" no wiki do OSM. Elas só são enviadas quando a espécie é indicada. Isso substitui a exclusão anunciada antes.
  - `osm_remover_dev.py` foi renomeado para `osm_corrigir_bananeiras_dev.py` e nunca chegou a ser rodado.
  - O script corrigido altera os 8 nós com espécie e apaga os 4 sem espécie. As correções ficam registradas em `runs/osm_correcoes_dev_registro.csv`.
  - Resultado esperado no servidor: 743 nós `natural=tree` e 8 `natural=plant`.
  - **Aplicado:** changeset 720132 (2026-11-06 00:29 UTC). 8 nós passaram a `natural=plant` + `genus=Musa` (versão 2) e 4 foram apagados (o servidor responde 410). O servidor fica com 743 `natural=tree` e 8 `natural=plant`.
