import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =============================================
# CONFIGURAÇÃO DE DIRETÓRIOS E CARREGAMENTO
# =============================================
DIR_PROCESSADO = "dados/processados"
DIR_MODELOS = "modelos"
DIR_GRAFICOS = "graficos"

os.makedirs(DIR_MODELOS, exist_ok=True)
os.makedirs(DIR_GRAFICOS, exist_ok=True)

caminho_dataset = os.path.join(DIR_PROCESSADO, "dataset_caracteristicas.csv")
df = pd.read_csv(caminho_dataset)

# ===================================
# SEPARAÇÃO DE ATRIBUTOS E RÓTULOS
# ===================================
X = df.drop(columns=['label'])
y = df['label']

# Divisão estratificada: 70% treino, 15% validação e 15% teste
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Divisão dos dados concluída:")
print(f" - Treino:    {len(X_train)} amostras (70%)")
print(f" - Validação: {len(X_val)} amostras (15%)")
print(f" - Teste:     {len(X_test)} amostras (15%)\n")

# Padronização de escala 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ===========================================
# TREINAMENTO E SELEÇÃO NA VALIDAÇÃO (15%)
# ===========================================
modelos = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (Support Vector Machine)": SVC(kernel='rbf', C=1.0, random_state=42),
    "KNN (K-Nearest Neighbors)": KNeighborsClassifier(n_neighbors=5)
}

melhor_acuracia_val = 0.0
melhor_modelo_nome = ""
melhor_modelo_obj = None

print("==================================================")
print("      AVALIAÇÃO DOS MODELOS (VALIDAÇÃO)           ")
print("==================================================\n")

for nome, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_val_pred = modelo.predict(X_val_scaled)
    
    acc_val = accuracy_score(y_val, y_val_pred)
    print(f" Modelo: {nome}")
    print(f"   Acurácia (Validação): {acc_val * 100:.2f}%\n")
    print("   Relatório de Classificação (Validação):")
    print(classification_report(y_val, y_val_pred))
    print("   Matriz de Confusão (Validação):")
    print(confusion_matrix(y_val, y_val_pred))
    print("-" * 50)

    if acc_val > melhor_acuracia_val:
        melhor_acuracia_val = acc_val
        melhor_modelo_nome = nome
        melhor_modelo_obj = modelo

print(f"\nMelhor modelo selecionado na Validação: {melhor_modelo_nome} com {melhor_acuracia_val * 100:.2f}% de acurácia.")

# ==================================================
# AVALIAÇÃO FINAL NO CONJUNTO DE TESTE (15%)
# ==================================================
y_test_pred = melhor_modelo_obj.predict(X_test_scaled)
acc_test = accuracy_score(y_test, y_test_pred)

print("\n==================================================")
print("      AVALIAÇÃO FINAL NO CONJUNTO DE TESTE        ")
print("==================================================")
print(f" Modelo Final: {melhor_modelo_nome}")
print(f" Acurácia no Teste (15%): {acc_test * 100:.2f}%\n")
print(" Relatório de Classificação (Teste):")
print(classification_report(y_test, y_test_pred))

# ===============================
# EXPORTAÇÃO DO MELHOR MODELO
# ===============================
caminho_modelo = os.path.join(DIR_MODELOS, "melhor_modelo.pkl")
caminho_scaler = os.path.join(DIR_MODELOS, "scaler.pkl")

joblib.dump(melhor_modelo_obj, caminho_modelo)
joblib.dump(scaler, caminho_scaler)

print(f"Modelo e padronizador salvos em: {DIR_MODELOS}/")

# ==============================================
# SALVAR GRÁFICO DA MATRIZ DE CONFUSÃO DO TESTE
# ==============================================
classes_ordenadas = np.unique(y_test)
cm = confusion_matrix(y_test, y_test_pred, labels=classes_ordenadas)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes_ordenadas,
            yticklabels=classes_ordenadas)

plt.title(f'Matriz de Confusão no Conjunto de Teste (15%)\n({melhor_modelo_nome})')
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.tight_layout()

caminho_cm_png = os.path.join(DIR_GRAFICOS, "matriz_confusao.png")
plt.savefig(caminho_cm_png, dpi=300)
plt.close()

print(f"Gráfico da Matriz de Confusão salvo em: {caminho_cm_png}")