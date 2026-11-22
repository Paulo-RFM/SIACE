from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import joblib

train = pd.read_csv("train.csv")

feature_cols = [f"f{i}" for i in range(524)]
X = train[feature_cols].values.astype("float32")
y = (train["target"] == "attack").astype(int).values

scaler = joblib.load("robust_scaler.pkl")
X_scaled = scaler.transform(X).astype("float32")

# Usamos o best iteration encontrada no treino com split
BEST_ITER = 290

model = CatBoostClassifier(
    iterations=BEST_ITER,
    depth=8,
    learning_rate=0.05,
    l2_leaf_reg=3,
    loss_function="Logloss",
    task_type="GPU",
    devices="0",
    random_seed=42,
    verbose=200
)

print("\n🚀 Treinando MODELO FINAL com todas as amostras e best_iter=", BEST_ITER)
model.fit(X_scaled, y)

model.save_model("catboost_gpu_FINAL.cbm")
print("\n📦 Modelo final salvo como: catboost_gpu_FINAL.cbm")
