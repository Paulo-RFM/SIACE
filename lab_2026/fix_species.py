import pandas as pd

csv_path = "/home/paulomaciel/proj/species_crops/species_labels.csv"

df = pd.read_csv(csv_path)

print("Antes da correção:")
print(df["species"].value_counts().sort_index())

# corrigir erro: 11 → 1
df["species"] = df["species"].replace({11: 1})

print("\nDepois da correção:")
print(df["species"].value_counts().sort_index())

# salvar de volta
df.to_csv(csv_path, index=False)
print("\n✔ Correção salva no arquivo!")
