import pandas as pd
import matplotlib.pyplot as plt
import os

# ================================
# CONFIGURAÇÃO
# ================================
csv_path = "/home/paulomaciel/proj/species_crops/species_labels.csv"
save_plot = True      # mudar para False se não quiser salvar
output_path = "/home/paulomaciel/proj/species_crops/species_distribution.png"

# ================================
# CARREGAR CSV
# ================================
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

df = pd.read_csv(csv_path)

if "species" not in df.columns:
    raise ValueError("A coluna 'species' não existe no CSV.")

# ================================
# CONTAR CLASSES
# ================================
counts = df["species"].value_counts().sort_index()

print("\nDistribuição das espécies:")
print(counts)

# ================================
# GERAR GRÁFICO
# ================================
plt.figure(figsize=(8,5))
plt.bar(counts.index.astype(str), counts.values)
plt.xlabel("Espécie (1 a 6)")
plt.ylabel("Quantidade")
plt.title("Distribuição das espécies no dataset")
plt.grid(axis="y", linestyle="--", alpha=0.4)

if save_plot:
    plt.savefig(output_path, dpi=200)
    print("\nGráfico salvo em:", output_path)

plt.show()
