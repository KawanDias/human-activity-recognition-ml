import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# =============================================
# CONFIGURAÇÃO DE DIRETÓRIOS E CARREGAMENTO
# =============================================
DIR_PROCESSADO = "dados/processados"
DIR_MODELOS = "modelos"

# 1. Carregar o dataset da nova amostra gerada
caminho_dataset = os.path.join(DIR_PROCESSADO, "dataset_caracteristicas.csv")

if not os.path.exists(caminho_dataset):
  raise FileNotFoundError(f"Arquivo {caminho_dataset} não encontrado!")

df = pd.read_csv(caminho_dataset)

# Separa as características do rótulo real
if "label" in df.columns:
  X = df.drop(columns=["label"])
  y_real = df["label"]
else:
  X = df
  y_real = None

# =============================================
# CARREGAMENTO DO MODELO E PADRONIZADOR SALVOS
# =============================================
caminho_modelo = os.path.join(DIR_MODELOS, "melhor_modelo.pkl")
caminho_scaler = os.path.join(DIR_MODELOS, "scaler.pkl")

if not os.path.exists(caminho_modelo) or not os.path.exists(caminho_scaler):
  raise FileNotFoundError(
      "Modelo ou Scaler não encontrados na pasta 'modelos/'."
  )

modelo = joblib.load(caminho_modelo)
scaler = joblib.load(caminho_scaler)

# =============================================
# PREDICAO / INFERÊNCIA NO DADO NOVO
# =============================================

X_scaled = scaler.transform(X)

# Realiza a predição
y_pred = modelo.predict(X_scaled)

print("==================================================")
print("     INFERÊNCIA DA NOVA AMOSTRA EM TEMPO REAL     ")
print("==================================================")
print(f"Total de janelas analisadas: {len(y_pred)}\n")

for i, pred in enumerate(y_pred):
  print(f"Janela {i+1:02d}: Classe Predita -> {pred}")

print("\n--------------------------------------------------")

if y_real is not None:
  acc = accuracy_score(y_real, y_pred)
  print(f"Taxa de acerto para a nova amostra: {acc * 100:.2f}%\n")
  print("Resumo das classificações:")
  print(pd.Series(y_pred).value_counts())

print("==================================================")