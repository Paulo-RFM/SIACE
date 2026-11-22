# Relatório técnico: correção do alinhamento imagem-máscara na augmentation

Data: 2026-11-06. Branch `fix/aligned-augmentation`. Baseline preservado no commit `e1f8ab5`, tag `baseline_before_alignment_fix`.
Todos os números abaixo vêm dos arquivos desta pasta (`comparacao.csv`, `e2e_comparacao.csv`, `contagens_por_tile.csv`) e de `runs/alinhamento_auditoria_20261105/resumo.json`.

## 1. Resumo executivo

- **O defeito é real e foi corrigido.** Em 5 dos 6 scripts do benchmark binário, imagem e máscara recebiam recortes aleatórios diferentes durante o treino. A máscara também era reamostrada com interpolação bilinear. A correção foi verificada por testes automáticos: depois dela, o IoU entre a máscara usada e a máscara correta é 1,0 em 100% dos casos.
- **Retreino.** As 6 configurações foram retreinadas com o código corrigido, em diretórios novos. Mantiveram o mesmo split, os mesmos hiperparâmetros e o mesmo protocolo: limiar escolhido na validação e teste avaliado uma única vez. No teste:
  - 5 das 6 configurações melhoraram em IoU, de +0,020 a +0,038;
  - uma piorou (512softAug2, −0,044);
  - o modelo principal (softAug4) foi de IoU 0,723 para 0,751 (+3,9%; IC95% da diferença [+0,011; +0,045]);
  - o F1 de detecção de ponta a ponta do modelo principal foi de 0,631 para 0,666 (IC95% [+0,001; +0,069]).
- **Ressalva decisiva.** No script 512semAug, todo o bloco de augmentation está desativado (dentro de uma string), então o código efetivo dele **não mudou**. O retreino dele funcionou como um **controle de ruído**: mesmo código, mesmos dados, só um novo treino. Mesmo assim, o IoU de teste subiu 0,020 e o F1 de borda (2 px) subiu 0,072. Isso tem a mesma ordem de grandeza das melhorias nas configurações corrigidas, cuja diferença mediana foi +0,025 de IoU e +0,076 de F1 de borda.
- **Conclusão da primeira rodada.** Com **um treino por configuração**, não era possível atribuir a melhora à correção.
- **Atualização (repetição com sementes, item 16).** Treinei o modelo principal com as sementes 1, 2, 3 e 42, uma vez com o código antigo e outra com o corrigido, somando 4 runs de cada lado.
  - O corrigido foi melhor em IoU nas 4 sementes. Média 0,745 contra 0,725 (+0,020, +2,8%), com p = 0,11 no teste de permutação, que não é significativo com n = 4.
  - O recall e o F1 de borda melhoraram com **separação completa entre os grupos**: todos os runs corrigidos ficaram acima de todos os antigos (p = 0,029, o menor possível com 4 contra 4).
    - Recall: +0,026.
    - F1 de borda 2 px: 0,184 → 0,247.
    - F1 de borda 5 px: 0,372 → 0,458.
  - A precisão não mudou.
  - Conclusão: **a correção melhora de forma consistente o contorno das copas e a cobertura (recall)**. O ganho de IoU tem direção consistente, mas é pequeno perto da variação entre sementes.

## 2. Descrição do problema

Os scripts faziam assim:

```python
seed = np.random.randint(0, 100000)
random.seed(seed); img = self.crop(img)    # transforms.RandomResizedCrop
random.seed(seed); mask = self.crop(mask)
```

A intenção era repetir o mesmo seed para obter o mesmo recorte. Só que o `RandomResizedCrop` do torchvision sorteia os parâmetros com o gerador do **PyTorch** (`torch.randint` e `torch.empty().uniform_()`), e não com o módulo `random` do Python. Por isso, cada chamada sorteava um recorte novo.

Há também um segundo defeito. A máscara RGB era reamostrada com a interpolação **bilinear**, padrão da transformação. Como `rgb_to_binary` considera árvore qualquer pixel não preto, a borda interpolada virava árvore.

Os flips já estavam corretos (um único sorteio aplicado à imagem e à máscara). A máscara de validação usava NEAREST nos scripts principais, mas não em `banchmark_Unet.py` e `Unet_plus_banchmark.py`.

## 3. Evidências encontradas

Auditoria em `runs/alinhamento_auditoria_20261105/`: 200 tiles reais do treino. Em cada um, reproduzi o método antigo e comparei a máscara usada com a máscara certa (recortada com os parâmetros da imagem).

| Faixa de escala do crop | Scripts que a usam | Deslocamento mediano (p90; máx.) | IoU mediano entre máscara usada e certa | % com IoU < 0,7 |
|---|---|---|---|---|
| 0,85–1,0 | softAug4, 512semAug2, 512softAug2 | 18 px (42; 65) | 0,793 | 27,5% |
| 0,4–1,0 | base, _26 | 53 px (105; 151) | 0,615 | 70,5% |
| 0,3–1,0 | 512semAug (**bloco desativado no script**: linha hipotética) | 59 px (122; 204) | 0,496 | 69% |

- Com o método novo, o IoU é 1,0 em 100% dos casos.
- As figuras `exemplo_*.png` mostram o contorno da máscara usada sobre a imagem recortada.
- **Correção de uma afirmação anterior.** Eu tinha atribuído a faixa 0,3–1,0 ao 512semAug como se ela fosse usada no treino. A análise estática mostra que nesse script a chamada do crop está dentro de uma string `""""…`, ou seja, é código morto. O próprio nome indica "sem augmentation". O retreino confirma isso: não gerou nenhuma linha em `aug_params.csv`.
- O `MANIFEST.md` do arquivo histórico (somente leitura) repete a faixa como configurada no código. Ela nunca foi aplicada.

## 4. Scripts corrigidos

- Novo módulo `aligned_augmentation.py` com `paired_random_resized_crop`:
  - os parâmetros são sorteados uma única vez e aplicados à imagem (BILINEAR) e à máscara (NEAREST);
  - imagem e máscara de tamanhos diferentes são recusadas;
  - log opcional dos parâmetros via `ALIGNED_AUG_LOG`.
- Scripts que passam a chamar o módulo:
  - `tree_segmentation_resnet.py` (base);
  - `tree_segmentation_resnet_26.py`;
  - `tree_segmentation_resnet_26_512semAug.py` (código morto, sem efeito);
  - `tree_segmentation_resnet_26_512semAug2.py`;
  - `tree_segmentation_resnet_26_512softAug2.py`;
  - `tree_segmentation_resnet_26_512softAug4.py` (modelo principal, a U-Net ResNet-34);
  - `tree_segmentation_resnet_26_512Aug5.py`;
  - `tree_segmentation_resnet_26_256softAug2.py`;
  - `banchmark_Unet.py`;
  - `Unet_plus_banchmark.py`;
  - `resnet3.py` (multiclasse).
- `banchmark_Unet.py` e `Unet_plus_banchmark.py` também passam a redimensionar a máscara de validação com NEAREST.
- As escalas, as proporções e todos os outros hiperparâmetros ficaram iguais.
- Testes: `tests/test_aligned_augmentation.py`, com 6 testes que passam. Eles cobrem:
  - alinhamento contra um rótulo recalculado da própria imagem transformada, com IoU ≥ `ALIGN_MIN_IOU` (padrão 0,98) em 4 faixas de escala;
  - igualdade exata com a máscara de referência;
  - ausência de valores intermediários;
  - um controle que detecta o método antigo;
  - recusa de tamanhos diferentes;
  - uma guarda estática que impede voltar a aplicar o crop à máscara isoladamente.
- **Achado documentado e não corrigido.** No softAug4, `valid_pixel_mask(img)` é calculado depois do color jitter. Como o contraste do jitter pode tornar não nulos os pixels pretos de nodata, a máscara de pixels válidos fica errada nesses casos. Não corrigi para não misturar dois efeitos.

## 5. Commits realizados

Branch `fix/aligned-augmentation`, sem push:

| Commit | Mensagem |
|---|---|
| `e1f8ab5` (master, tag `baseline_before_alignment_fix`) | Baseline preservado antes da correção do alinhamento imagem-máscara |
| `37e111c` | Fix random crop alignment between image and mask |
| `99fca95` | Add synchronized augmentation tests and misalignment audit |
| `c3b7a50` | Add retraining chain and baseline-vs-corrected comparison scripts |
| (este) | Retrain U-Net using corrected labels |
| (este) | Recompute benchmark after augmentation fix |

## 6. Branch criada

`fix/aligned-augmentation`, criada a partir de `e1f8ab5`. A branch master não foi alterada depois do commit de baseline.

## 7. Artefatos preservados

- `artifacts/legacy_run_20261106/`, somente leitura, contém:
  - cópia integral de `runs/` (6,2 GB);
  - os scripts, as configs do split e os relatórios;
  - `SHA256SUMS` com 699 arquivos;
  - `MANIFEST.md`.
- Nenhum run antigo foi alterado. Isso inclui `runs/p1_20261101_140519/`.
- Os retreinos estão em `runs/alinhado_20261105_2215_<config>/`. Cada um tem `log.txt` (comando, commit, início e fim, código de saída), `metrics.csv`, os checkpoints e `aug_params.csv` (um recorte por linha).

## 8. Resultados antigos (com desalinhamento)

Teste: 204 tiles, limiar congelado na validação, avaliados pelo mesmo código.

| Config | Limiar | IoU val | IoU teste [IC95] | Dice | Precisão | Recall | Acurácia | F1 borda 2 px | F1 borda 5 px | Treino (min, épocas) |
|---|---|---|---|---|---|---|---|---|---|---|
| softAug4 (principal) | 0,65 | 0,726 | 0,723 [0,691; 0,756] | 0,839 | 0,805 | 0,876 | 0,937 | 0,157 | 0,332 | 27,8 (17) |
| 512semAug | 0,30 | 0,728 | 0,737 [0,703; 0,770] | 0,849 | 0,812 | 0,889 | 0,940 | 0,261 | 0,480 | 37,5 (22) |
| _26 | 0,55 | 0,714 | 0,711 [0,677; 0,742] | 0,831 | 0,794 | 0,873 | 0,933 | 0,133 | 0,277 | 57,3 (31) |
| base | 0,45 | 0,694 | 0,695 [0,659; 0,728] | 0,820 | 0,786 | 0,857 | 0,929 | 0,123 | 0,256 | 31,5 (17) |
| 512semAug2 | 0,20 | 0,693 | 0,703 [0,667; 0,736] | 0,826 | 0,784 | 0,871 | 0,931 | 0,140 | 0,325 | 62,2 (31) |
| 512softAug2 | 0,30 | 0,678 | 0,681 [0,638; 0,723] | 0,811 | 0,805 | 0,817 | 0,928 | 0,139 | 0,305 | 32,2 (16) |

## 9. Resultados novos (corrigidos)

| Config | Limiar | IoU val | IoU teste [IC95] | Dice | Precisão | Recall | Acurácia | F1 borda 2 px | F1 borda 5 px | Treino (min, épocas) |
|---|---|---|---|---|---|---|---|---|---|---|
| softAug4 (principal) | 0,40 | 0,748 | 0,751 [0,717; 0,781] | 0,858 | 0,810 | 0,912 | 0,943 | 0,233 | 0,445 | 25,0 (15) |
| 512semAug (controle) | 0,55 | 0,737 | 0,757 [0,724; 0,787] | 0,862 | 0,822 | 0,904 | 0,945 | 0,334 | 0,543 | 30,7 (18) |
| _26 | 0,60 | 0,722 | 0,734 [0,698; 0,767] | 0,847 | 0,803 | 0,895 | 0,939 | 0,193 | 0,384 | 23,7 (13) |
| base | 0,50 | 0,719 | 0,733 [0,697; 0,767] | 0,846 | 0,811 | 0,885 | 0,939 | 0,210 | 0,391 | 23,7 (13) |
| 512semAug2 | 0,55 | 0,708 | 0,728 [0,690; 0,764] | 0,842 | 0,793 | 0,898 | 0,937 | 0,263 | 0,477 | 28,2 (14) |
| 512softAug2 | 0,10 | 0,661 | 0,637 [0,593; 0,680] | 0,779 | 0,752 | 0,807 | 0,914 | 0,144 | 0,319 | 30,2 (15) |

- Tempo de inferência: entre 0,107 e 0,108 s por tile nos runs novos, contra 0,108 a 0,133 s nos antigos. A arquitetura é a mesma, então essa diferença reflete a carga da máquina no momento da medição, não a correção.
- Os treinos novos pararam mais cedo pelo early stopping (13 a 18 épocas, contra 16 a 31). Uma hipótese é que rótulos coerentes estabilizam a validação mais cedo, mas ela não foi testada.

## 10. Comparação estatística

Diferença pareada (novo − antigo) por bootstrap de tiles do teste, B = 2000, seed 42.

| Config | ΔIoU [IC95] | Δ% | ΔF1 borda 2 px [IC95] | ΔRecall [IC95] | ΔPrecisão [IC95] |
|---|---|---|---|---|---|
| softAug4 | +0,028 [+0,011; +0,045] | +3,9% | +0,075 [+0,063; +0,088] | +0,036 [+0,019; +0,053] | +0,004 [−0,008; +0,016] |
| **512semAug (controle, código igual)** | **+0,020 [+0,004; +0,039]** | +2,7% | **+0,072 [+0,060; +0,085]** | +0,015 [−0,003; +0,037] | +0,011 [+0,001; +0,021] |
| _26 | +0,023 [+0,008; +0,040] | +3,2% | +0,060 [+0,048; +0,073] | +0,023 [+0,004; +0,042] | +0,010 [−0,004; +0,022] |
| base | +0,038 [+0,019; +0,056] | +5,5% | +0,087 [+0,072; +0,102] | +0,028 [+0,015; +0,042] | +0,024 [+0,003; +0,044] |
| 512semAug2 | +0,025 [+0,007; +0,044] | +3,5% | +0,122 [+0,107; +0,138] | +0,026 [+0,009; +0,045] | +0,009 [−0,007; +0,026] |
| 512softAug2 | −0,044 [−0,076; −0,017] | −6,5% | +0,005 [−0,008; +0,018] | −0,010 [−0,048; +0,027] | −0,053 [−0,072; −0,036] |

- Nas 5 configurações em que a correção atua de fato:
  - ΔIoU médio +0,014 e mediano +0,025, contra +0,020 no controle;
  - ΔF1 de borda 2 px médio +0,070, contra +0,072 no controle.
- **Não há relação dose-resposta.** As configurações com o desalinhamento mais forte (base e _26, escala 0,4–1,0, IoU mediano de 0,62 entre a máscara usada e a certa) não melhoraram claramente mais que as de escala 0,85–1,0, nem mais que o controle.
- 512semAug2 e 512softAug2 são **idênticos**, exceto pela paciência do early stopping (7 contra 5). Mesmo assim, o primeiro melhorou +0,025 e o segundo piorou −0,044.
- Os ICs do bootstrap medem só a variação entre tiles do teste. Eles **não incluem a variação entre treinos**, que o controle mostra ser da mesma ordem que o efeito.

**Ponta a ponta, modelo principal** (pipeline do teste com componentes conexas min_area 3000 e a mesma EfficientNet; acerto com IoU ≥ 0,5):

| Métrica | Antigo | Novo | Diferença [IC95] |
|---|---|---|---|
| F1 de detecção | 0,631 | 0,666 | +0,034 [+0,001; +0,069] |
| Precisão | 0,754 | 0,767 | +0,013 [−0,029; +0,058] |
| Recall | 0,543 | 0,588 | +0,045 [+0,012; +0,077] |
| Instâncias por árvore anotada | 0,720 | 0,767 | +0,047 [+0,011; +0,083] |
| MAE da contagem por tile | 0,779 | 0,794 | +0,015 [−0,059; +0,088] |
| Copas fundidas (árvores anotadas extras numa mesma instância) | 72 | 71 | −1 [−9; +7] |
| Acerto de espécie nas árvores detectadas | 0,806 (227 árvores) | 0,838 (247 árvores) | conjuntos diferentes, não pareado |

- Não há run de controle de ponta a ponta. Pelo controle de pixel, a melhora de F1 cabe na variação entre treinos.
- A fusão de copas vizinhas **não mudou**: o problema de separar árvores próximas vem das componentes conexas, não do alinhamento.
- Por tile, o IoU do modelo principal melhorou em 133 dos 204 tiles e piorou em 52 (mediana +0,023).

## 11. Análise técnica

- **Bordas.** O F1 de borda subiu bastante em quase todos os runs. Só que o controle subiu o mesmo tanto, então essa subida acompanha principalmente o **limiar escolhido na validação**. Um limiar baixo (0,20–0,30 em vários runs antigos) engorda a máscara e piora a borda. O limiar do modelo principal caiu de 0,65 para 0,40, sinal de outra calibração das probabilidades.
- **Copas fundidas e separação de árvores próximas.** Não houve mudança: 72 contra 71 fusões, com IC que inclui zero. Isso já era esperado, porque a separação depende da individualização (componentes conexas), que não mudou.
- **Consistência espacial.** O recall de pixel subiu de forma consistente, e a precisão mudou pouco. Nos exemplos visuais (`exemplo_teste_*.png`), o modelo corrigido recupera copas inteiras que o antigo perdia, como OrtoRP-3cm_045_043 (IoU 0,02 → 0,87). Os dois tiles do fim da lista mostram o caso inverso. Exemplos escolhidos pela maior mudança são ilustrações, não prova.
- **Por que o efeito é pequeno, apesar do defeito grande.**
  - Os deslocamentos são aleatórios e mudam a cada época. O ruído do rótulo tende a se cancelar em média, e funciona mais como regularização ruidosa do que como viés sistemático.
  - A rede também vê o tile inteiro na validação, e o checkpoint é escolhido por ela.
  - Esta explicação é plausível, mas não foi testada.

## 12. Conclusões

1. O desalinhamento existia em 5 dos 6 treinos do benchmark binário, inclusive no modelo principal (deslocamento mediano de 18 px). Ele foi corrigido e está coberto por testes.
2. Com um treino por configuração, a melhora observada (+0,028 de IoU no modelo principal) **não se distingue da variação entre treinos**: o controle sem mudança de código variou +0,020.
3. Não há evidência de que os resultados antigos da monografia estejam inflados ou invalidados por esse defeito. Os modelos antigos foram avaliados contra rótulos corretos no teste; o defeito afetava só os rótulos de treino.
4. Para afirmar um ganho, eram necessárias repetições com seeds diferentes, código antigo e novo, na mesma máquina. Isso foi feito para o modelo principal (item 16).
5. **Com 4 sementes de cada lado:**
   - a correção melhora de forma estatisticamente consistente o recall e o F1 de borda (p = 0,029, separação completa);
   - o IoU sobe +0,020 em média, maior nas 4 sementes, mas sem significância com n = 4 (p = 0,11);
   - a precisão não muda.
6. As conclusões 2 e 3 continuam valendo para as outras 5 configurações, que tiveram um só treino.

## 13. Recomendações para a monografia

- **Não trocar o modelo principal antes da entrega de 07/11.**
  - O ganho não está demonstrado.
  - O limiar mudaria (0,65 → 0,40).
  - O mapa da cidade, o limiar de 0,94 do envio ao OSM e todas as figuras teriam de ser refeitos.
- **Registrar o defeito com transparência** na seção de limitações (texto no item 14). Isso é melhor do que omitir, porque o código está versionado.
- Se quiser citar o resultado: "um retreino com a correção obteve IoU de teste 0,751, contra 0,723, mas um retreino de controle sem alteração variou 0,020, o que impede atribuir a diferença à correção".
- Trabalho futuro: repetir o benchmark com várias seeds.
- **Atualização após o item 16.** A melhora no contorno e no recall agora está demonstrada. Adotar o modelo corrigido até a banca (05/12) é defensável, mas exige refazer:
  - o mapa da cidade;
  - a avaliação de ponta a ponta;
  - a calibração do limiar de 0,94 do envio ao OSM;
  - as figuras e tabelas.

  Para a entrega de 07/11, a recomendação continua: manter o modelo original e usar o texto do item 14.

## 14. Sugestão de texto para limitações

> Após a conclusão dos experimentos, uma auditoria do código identificou que, em cinco das seis configurações de treino da segmentação binária, inclusive na U-Net ResNet-34 adotada, o recorte aleatório da *data augmentation* não era sincronizado entre imagem e máscara: os parâmetros eram sorteados duas vezes, e a máscara era reamostrada por interpolação bilinear. Em 200 tiles do treino, o deslocamento mediano entre os recortes foi de 18 pixels na configuração adotada. O defeito afetou apenas os rótulos usados no treino; validação e teste usaram as máscaras corretas. Após a correção, a U-Net ResNet-34 foi treinada quatro vezes com sementes aleatórias diferentes, e o código original outras quatro, nas mesmas condições. A versão corrigida obteve, em média, IoU de teste 0,745 contra 0,725 (maior nas quatro sementes, mas sem significância estatística com quatro repetições, p = 0,11), recall 0,905 contra 0,879 e F1 de borda (tolerância de 2 pixels) 0,247 contra 0,184. Nestas duas últimas métricas, todas as execuções corrigidas superaram todas as originais (teste de permutação, p = 0,029). Os resultados apresentados neste trabalho correspondem ao modelo original. A correção deve, portanto, melhorar principalmente a delimitação das copas.

## 15. Sugestão de texto para a defesa

> "Durante a revisão final, encontrei um erro na minha *data augmentation*: o recorte aleatório não era o mesmo na imagem e na máscara, porque a semente que eu fixava não controlava o gerador usado pela biblioteca. Corrigi e escrevi testes automáticos que garantem o alinhamento. Depois, para medir o efeito de verdade, treinei o modelo principal quatro vezes com o código antigo e quatro com o corrigido, mudando só a semente. O corrigido foi melhor em todas as quatro sementes. No IoU, o ganho foi de 0,02, menor que a variação entre sementes, então não posso afirmar que seja significativo. Já no contorno das copas e no recall, todas as execuções corrigidas ficaram acima de todas as antigas. Os números da monografia são do modelo original, avaliado contra rótulos corretos. A correção melhora sobretudo a delimitação das copas."

## 16. Repetição com sementes do modelo principal (2026-11-06)

**Desenho**
- Script do código antigo: `tree_segmentation_resnet_26_512softAug4_codigo_antigo.py`, uma cópia exata do script da tag `baseline_before_alignment_fix`, com o crop desalinhado.
- Script do código corrigido: `tree_segmentation_resnet_26_512softAug4.py`.
- A única mudança nos dois é o argumento `--seed`. O padrão continua 42, a semente original.
- Sementes 1, 2 e 3 novas, intercaladas (cadeia `runs/seeds_alinhamento_20261106.sh`). A semente 42 reutiliza os runs que já existiam (`p1_20261101_140519` e `alinhado_20261105_2215_softAug4`).
- Mesmo split, dados, hiperparâmetros e protocolo: limiar escolhido na validação pelo próprio script, teste avaliado uma vez.
- Resultados: `runs/seeds_alinhamento_comparacao_20261106/` (`resultados_por_run.csv`, `resumo.csv`, `seeds_iou_borda.png`).

| Grupo | Semente | Limiar | IoU val | IoU teste | Dice | Precisão | Recall | F1 borda 2 px | F1 borda 5 px | Épocas |
|---|---|---|---|---|---|---|---|---|---|---|
| antigo | 1 | 0,55 | 0,735 | 0,740 | 0,850 | 0,824 | 0,879 | 0,206 | 0,408 | 29 |
| antigo | 2 | 0,60 | 0,711 | 0,703 | 0,825 | 0,771 | 0,888 | 0,174 | 0,344 | 11 |
| antigo | 3 | 0,60 | 0,736 | 0,733 | 0,846 | 0,819 | 0,874 | 0,199 | 0,403 | 20 |
| antigo | 42 | 0,65 | 0,726 | 0,723 | 0,839 | 0,806 | 0,876 | 0,157 | 0,332 | 17 |
| corrigido | 1 | 0,50 | 0,756 | 0,755 | 0,860 | 0,810 | 0,917 | 0,275 | 0,495 | 27 |
| corrigido | 2 | 0,50 | 0,732 | 0,732 | 0,845 | 0,797 | 0,899 | 0,233 | 0,439 | 13 |
| corrigido | 3 | 0,40 | 0,730 | 0,740 | 0,851 | 0,813 | 0,893 | 0,246 | 0,455 | 17 |
| corrigido | 42 | 0,40 | 0,748 | 0,751 | 0,858 | 0,810 | 0,912 | 0,233 | 0,445 | 15 |

O p é de um teste de permutação exato bicaudal sobre as 70 divisões possíveis dos 8 runs. Com 4 contra 4, o menor p possível é 0,029.

| Métrica | Média antigo (dp) | Média corrigido (dp) | Diferença | Δ% | Sementes com melhora | p |
|---|---|---|---|---|---|---|
| IoU | 0,725 (0,016) | 0,744 (0,011) | +0,020 | +2,8% | 4/4 | 0,11 |
| Dice | 0,840 (0,011) | 0,853 (0,007) | +0,013 | +1,6% | 4/4 | 0,11 |
| Precisão | 0,805 (0,024) | 0,807 (0,007) | +0,003 | +0,3% | 2/4 | 0,91 |
| Recall | 0,879 (0,006) | 0,905 (0,011) | +0,026 | +2,9% | 4/4 | **0,029** |
| Acurácia | 0,937 (0,006) | 0,941 (0,003) | +0,005 | +0,5% | 4/4 | 0,17 |
| F1 borda 2 px | 0,184 (0,022) | 0,247 (0,020) | +0,063 | +34% | 4/4 | **0,029** |
| F1 borda 5 px | 0,372 (0,039) | 0,458 (0,025) | +0,087 | +23% | 4/4 | **0,029** |

**Leitura**
- **IoU e Dice.** O corrigido foi melhor nas 4 sementes, mas os grupos se sobrepõem: o melhor antigo (0,740) supera o pior corrigido (0,732). O efeito provável é de cerca de +0,02 de IoU, sem significância com n = 4.
- **Recall e borda.** Há separação completa entre os grupos. Com a máscara alinhada, o modelo aprende a cobrir a copa inteira e a acertar o contorno.
- **Precisão.** Não mudou.
- **Limiar escolhido na validação.** Ficou sistematicamente mais baixo no código corrigido (0,40–0,50 contra 0,55–0,65). Isso indica probabilidades mais bem calibradas em relação ao rótulo real.
- **Revisão do controle 512semAug.** Na primeira rodada, o 512semAug variou +0,072 no F1 de borda sem mudar o código. Aqui, a variação do F1 de borda entre sementes do código antigo foi de 0,157 a 0,206. A separação observada (0,233 a 0,275 no corrigido) fica fora dessa faixa.
- **Limitações.**
  - São só 4 repetições, numa única configuração (a principal).
  - As outras 5 configurações continuam com um treino só.
  - A semente igual nos dois códigos não torna os runs pareados de fato, porque o consumo do gerador aleatório muda com a correção. A coluna "sementes com melhora" é só descritiva.


## Arquivos desta pasta

| Arquivo | Conteúdo |
|---|---|
| `comparacao.csv` / `comparacao.json` | métricas de pixel, antigo × novo, com ICs |
| `contagens_por_tile.csv` | TP, FP e FN por tile e por configuração |
| `e2e_comparacao.csv` | comparação de ponta a ponta do modelo principal |
| `iou_borda_antigo_vs_corrigido.png` | gráfico de IoU e F1 de borda |
| `exemplo_teste_*.png` | exemplos visuais (anotado, antigo, corrigido) |
| `<config>/frozen_threshold.json`, `<config>/test_eval/` | limiar de validação e avaliação de teste de cada run novo |

Pipeline de teste do modelo corrigido: `runs/pipeline_test_alinhado_20261105/`. A regressão entre as máscaras do pipeline e as predições independentes deu 67 pixels diferentes em 204 tiles.
