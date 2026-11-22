# Classificadores binários por espécie × multiclasse (2026-11-05)

## O que foi feito
- **Treinos:** três EfficientNet-B0 binárias, cada uma treinando uma espécie contra todas as outras classes, com o mesmo protocolo do multiclasse:
  - mesmos recortes anotados e mesma divisão espacial;
  - semente 42;
  - seleção pelo F1-macro de validação;
  - teste avaliado uma única vez.

  | Espécie | Run | Época escolhida / total | F1-macro val | Treino |
  |---|---|---|---|---|
  | bananeira | `classif_20261103_115814_efficientnet_binaria_bananeira` (rodado pelo autor) | 6 / 11 | 0,865 | 23 min |
  | palmeira | `classif_20261104_235839_efficientnet_binaria_palmeira` | 8 / 13 | 0,767 | 28 min |
  | mangueira | `classif_20261104_235839_efficientnet_binaria_mangueira` | 6 / 11 | 0,781 | 23 min |

  Árvore seca e jabuticabeira não foram treinadas: têm 3 e 6 exemplos de teste, o que não permite conclusão.
- **Comparação:** na tarefa "espécie contra as demais" (418 recortes de teste), três formas de decidir:
  - **A.** multiclasse com a regra normal (argmax);
  - **B.** multiclasse com limiar próprio da espécie, escolhido na validação (maior F1 em `prob_<espécie>`);
  - **C.** binário.
- **Estatística:**
  - bootstrap pareado por tile (B = 2000) para F1, AUC-ROC e AP;
  - McNemar exato sobre os acertos da decisão.
- **Arquivos** (`analisar_binarios.py`):
  - `metricas.csv`, `diferencas.csv`, `matrizes_2x2.csv`, `falsos_positivos.csv`, `analise.json`;
  - a figura `f1_binario_vs_multiclasse.png`, copiada para o LaTeX como `binarios_especie_f1.png`;
  - em cada run, as matrizes de confusão completas, as curvas e a calibração.

## Resultados (teste)
| Espécie (positivos) | Modelo | P | R | F1 [IC95] | AUC | AP | TP/FP/FN |
|---|---|---|---|---|---|---|---|
| bananeira (48) | A multiclasse | 0,815 | 0,917 | 0,863 [0,788; 0,933] | 0,963 | 0,856 | 44/10/4 |
| | B limiar val 0,55 | 0,863 | 0,917 | 0,889 [0,815; 0,958] | 0,963 | 0,856 | 44/7/4 |
| | C binário | 0,935 | 0,896 | 0,915 [0,857; 0,967] | 0,964 | 0,881 | 43/3/5 |
| palmeira (20) | A | 0,667 | 0,800 | 0,727 [0,529; 0,862] | 0,947 | 0,755 | 16/8/4 |
| | B limiar 0,85 | 0,857 | 0,600 | 0,706 [0,471; 0,857] | 0,947 | 0,755 | 12/2/8 |
| | C | 0,652 | 0,750 | 0,698 [0,485; 0,835] | 0,974 | 0,779 | 15/8/5 |
| mangueira (19) | A | 0,340 | 0,895 | 0,493 [0,311; 0,630] | 0,956 | 0,420 | 17/33/2 |
| | B limiar 0,85 | 0,480 | 0,632 | 0,545 [0,333; 0,717] | 0,956 | 0,420 | 12/13/7 |
| | C | 0,500 | 0,579 | 0,537 [0,286; 0,714] | 0,955 | 0,423 | 11/11/8 |

Diferenças pareadas de F1 (binário − multiclasse A): bananeira +0,052 [−0,001; 0,111]; palmeira −0,030 [−0,151; 0,083]; mangueira +0,044 [−0,162; 0,223]. Todas as diferenças de AUC e AP têm IC contendo zero. Binário − B: bananeira +0,026 [−0,019; 0,074]; palmeira −0,008; mangueira −0,009.

McNemar: mangueira A × C, 7 contra 23 recortes, p = 0,005; mangueira B × C, 5 contra 6, p = 1,0; bananeira A × C, p = 0,07; palmeira, p = 1,0.

Os falsos positivos vêm quase só da classe 1: mangueira A, 31 de 33; mangueira C, 11 de 11; palmeira A e C, 7 de 8.

## Evidências e conclusão
1. **Nenhuma melhora distinguível do acaso.** Em nenhuma das três espécies o binário superou o multiclasse em F1, AUC ou AP. Na palmeira ele foi até um pouco pior.
2. **O binário não distingue melhor as espécies.** A AUC é praticamente igual em todas, ou seja, os modelos ordenam as copas igualmente bem. O que muda é o ponto de decisão: o binário é mais cauteloso ao indicar a espécie.
3. **A prova mais clara é a mangueira.** O binário acerta significativamente mais decisões que o multiclasse com argmax (p = 0,005). Contra o multiclasse com limiar escolhido na validação, a diferença some (p = 1,0; F1 0,537 × 0,545).
4. **O mesmo efeito sai sem novo treino.** Ajustar o limiar por espécie na validação, no próprio multiclasse, reproduz o ganho: na mangueira a precisão sobe de 0,34 para 0,48 e os falsos positivos caem de 33 para 13. É a forma mais barata de reduzir a superestimação de mangueiras na cidade.
5. **O gargalo é a classe 1.** Ela é a fonte de quase todos os falsos positivos, o que aponta para a heterogeneidade dela, não para o formato do classificador.

**Limitações:**
- um único treino (semente) por modelo;
- 19 a 48 positivos por espécie;
- na palmeira, o limiar da validação (escolhido com 20 positivos) não se transferiu bem para o teste.

**Na monografia:** subseção `sec:cls_binario` reescrita, com a Tabela `tab:cls_binario` e a Figura `fig:cls_binario`, e frase correspondente na conclusão.
