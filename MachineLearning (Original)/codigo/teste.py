import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# =============================================
# CONFIGURAÇÃO DE DIRETÓRIOS E CARREGAMENTO
# =============================================
DIR_PROCESSADO = "dados/processados"
DIR_MODELOS = "modelos"
DIR_GRAFICOS = "graficos"

os.makedirs(DIR_GRAFICOS, exist_ok=True)

# 1. Carregar o dataset completo
caminho_dataset = os.path.join(DIR_PROCESSADO, "dataset_caracteristicas.csv")
df = pd.read_csv(caminho_dataset)

X = df.drop(columns=["label"])
y = df["label"]

# 2. Divisão estratificada 70/15/15 
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(
    f"Conjunto de Teste isolado com sucesso: {len(X_test)} amostras (15% do dataset).\n"
)

# =============================================
# CARREGAMENTO DO MODELO E PADRONIZADOR SALVOS
# =============================================

caminho_modelo = os.path.join(DIR_MODELOS, "melhor_modelo.pkl")
caminho_scaler = os.path.join(DIR_MODELOS, "scaler.pkl")

if not os.path.exists(caminho_modelo) or not os.path.exists(caminho_scaler):
  raise FileNotFoundError(
      "Modelo ou Scaler não encontrados em 'modelos/'. Execute o"
      " 'treinamento.py' primeiro."
  )

modelo = joblib.load(caminho_modelo)
scaler = joblib.load(caminho_scaler)

# =============================================
# AVALIAÇÃO NO CONJUNTO DE TESTE (15% INÉDITO)
# =============================================

X_test_scaled = scaler.transform(X_test)

# Realiza a predição
y_pred = modelo.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("==================================================")
print("     AVALIAÇÃO FINAL DO MODELO SALVO (TESTE)      ")
print("==================================================")
print(f" Acurácia no Teste (15%): {acc * 100:.2f}%\n")
print(" Relatório de Classificação:")
print(classification_report(y_test, y_pred))
print(" Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("-" * 50)

# ==============================================
# SALVAR GRÁFICO DA MATRIZ DE CONFUSÃO DO TESTE
# ==============================================
classes_ordenadas = np.unique(y_test)
cm = confusion_matrix(y_test, y_pred, labels=classes_ordenadas)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes_ordenadas,
    yticklabels=classes_ordenadas,
)

plt.title("Matriz de Confusão no Conjunto de Teste (15% Inédito)")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.tight_layout()

caminho_cm_png = os.path.join(DIR_GRAFICOS, "matriz_confusao.png")
plt.savefig(caminho_cm_png, dpi=300)
plt.close()

print(f"\nGráfico da Matriz de Confusão salvo em: {caminho_cm_png}")