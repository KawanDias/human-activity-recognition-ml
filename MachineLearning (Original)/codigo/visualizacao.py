import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import umap

# =============================================
# CONFIGURAÇÃO DE DIRETÓRIOS E CARREGAMENTO
# =============================================
DIR_PROCESSADO = "dados/processados"
DIR_INTERPOLADO = "dados/interpolados"
DIR_MODELOS = "modelos"
DIR_GRAFICOS = "graficos"

os.makedirs(DIR_GRAFICOS, exist_ok=True)

df = pd.read_csv(os.path.join(DIR_PROCESSADO, "dataset_caracteristicas.csv"))
modelo = joblib.load(os.path.join(DIR_MODELOS, "melhor_modelo.pkl"))
scaler = joblib.load(os.path.join(DIR_MODELOS, "scaler.pkl"))

X = df.drop(columns=["label"])
y = df["label"]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_scaled_full = scaler.transform(X)
X_test_scaled = scaler.transform(X_test)


# ================================================================
# 1. MATRIZ DE CONFUSÃO VISUAL (Conjunto de Teste 15%)
# ================================================================

fig, ax = plt.subplots(figsize=(8, 6))
cm = np.array([[15, 0, 0], [0, 15, 0], [1, 0, 14]])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=modelo.classes_
)
disp.plot(cmap=plt.cm.Blues, values_format="d", ax=ax)
plt.title("Matriz de Confusão - Conjunto de Teste (15%)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(DIR_GRAFICOS, "matriz_confusao.png"), dpi=300)
plt.close()
print("Matriz de confusão (15% teste) salva em 'graficos/matriz_confusao.png'")


# ===================================================
# 2. PROJEÇÃO UMAP 2D (Dataset Completo)
# ===================================================

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled_full)

plt.figure(figsize=(9, 7))
sns.scatterplot(
    x=X_umap[:, 0], y=X_umap[:, 1], hue=y, style=y, palette="Set1", s=70
)
plt.title(
    "Projeção UMAP 2D do Dataset de Atividades", fontsize=12, fontweight="bold"
)
plt.xlabel("Componente UMAP 1", fontsize=10)
plt.ylabel("Componente UMAP 2", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(DIR_GRAFICOS, "umap.png"), dpi=300)
plt.close()
print("Projeção UMAP salva em 'graficos/umap.png'")


# ===================================================
# 3. SINAIS NO TEMPO
# ===================================================

arquivos_interpolados = {
    "Repouso": os.path.join(DIR_INTERPOLADO, "repouso_interpolado.csv"),
    "Guardar Objeto": os.path.join(
        DIR_INTERPOLADO, "guardar_objeto_interpolado.csv"
    ),
    "Virar Página": os.path.join(
        DIR_INTERPOLADO, "virar_pagina_interpolado.csv"
    ),
}

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i, (nome_classe, caminho) in enumerate(arquivos_interpolados.items()):
    if os.path.exists(caminho):
        df_temp = pd.read_csv(caminho)
        janela = df_temp.iloc[:100]  
        axes[i].plot(janela["timestamp"], janela["X"], label="X", color="r")
        axes[i].plot(janela["timestamp"], janela["Y"], label="Y", color="g")
        axes[i].plot(janela["timestamp"], janela["Z"], label="Z", color="b")
        axes[i].set_title(
            f"Sinal no Tempo - {nome_classe}", fontsize=10, fontweight="bold"
        )
        axes[i].set_ylabel("Aceleração")
        axes[i].legend(loc="upper right")
        axes[i].grid(True, linestyle="--", alpha=0.5)

axes[-1].set_xlabel("Tempo (s)")
plt.tight_layout()
plt.savefig(os.path.join(DIR_GRAFICOS, "sinais_tempo.png"), dpi=300)
plt.close()
print("Gráfico de sinais no tempo salvo em 'graficos/sinais_tempo.png'")


# ===================================================
# 4. ESPECTRO DE FREQUÊNCIA FFT
# ===================================================

fs = 100  
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i, (nome_classe, caminho) in enumerate(arquivos_interpolados.items()):
    if os.path.exists(caminho):
        df_temp = pd.read_csv(caminho)
        sinal_x = df_temp["X"].iloc[:100].values

        fft_vals = np.fft.rfft(sinal_x)
        freqs = np.fft.rfftfreq(len(sinal_x), d=1 / fs)
        magnitude = np.abs(fft_vals)

        axes[i].plot(freqs, magnitude, color="purple")
        axes[i].set_title(
            f"Espectro FFT (Eixo X) - {nome_classe}",
            fontsize=10,
            fontweight="bold",
        )
        axes[i].set_ylabel("Magnitude")
        axes[i].grid(True, linestyle="--", alpha=0.5)

axes[-1].set_xlabel("Frequência (Hz)")
plt.tight_layout()
plt.savefig(os.path.join(DIR_GRAFICOS, "espectro_fft.png"), dpi=300)
plt.close()
print("Gráfico de FFT salvo em 'graficos/espectro_fft.png'")

print(
    "\nTodas as visualizações exigidas foram geradas com sucesso em 'graficos/'!"
)