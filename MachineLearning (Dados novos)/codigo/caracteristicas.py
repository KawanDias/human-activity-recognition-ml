import os
import pandas as pd
import numpy as np
from scipy import stats

# ==========================================
# CONFIGURAÇÃO DE DIRETÓRIOS E PARÂMETROS
# ==========================================

DIR_INTERPOLADO = "dados/interpolados"
DIR_PROCESSADO = "dados/processados"

os.makedirs(DIR_PROCESSADO, exist_ok=True)

# Parâmetros de amostragem e janelamento

FS = 100                 # Frequência de amostragem em Hz
TAMANHO_JANELA = 100     # 1 segundo = 100 amostras
PASSO = 100              # Janelas sem sobreposição (0%)
MAX_JANELAS = 100        # Limite de 100 janelas (100s) por classe

arquivos_classes = {
    "virar_pagina_novo_interpolado.csv": "virar_pagina"
}


# ==========================================
# EXTRAÇÃO DE CARACTERÍSTICAS POR EIXO
# ==========================================

def extrair_caracteristicas_janela(sinal, prefixo):
    """Calcula métricas no domínio do tempo e da frequência."""
    media = np.mean(sinal)
    desvio = np.std(sinal)
    var = np.var(sinal)
    v_min = np.min(sinal)
    v_max = np.max(sinal)
    p2p = v_max - v_min
    rms = np.sqrt(np.mean(sinal**2))
    skewness = stats.skew(sinal)
    kurtosis = stats.kurtosis(sinal)

    # FFT
    fft_vals = np.abs(np.fft.rfft(sinal))
    freqs = np.fft.rfftfreq(len(sinal), d=1/FS)
    
    energia = np.sum(fft_vals**2)
    freq_dominante = freqs[np.argmax(fft_vals[1:]) + 1] if len(fft_vals) > 1 else 0.0

    return {
        f'{prefixo}_media': media,
        f'{prefixo}_std': desvio,
        f'{prefixo}_var': var,
        f'{prefixo}_min': v_min,
        f'{prefixo}_max': v_max,
        f'{prefixo}_p2p': p2p,
        f'{prefixo}_rms': rms,
        f'{prefixo}_skew': skewness,
        f'{prefixo}_kurtosis': kurtosis,
        f'{prefixo}_energia': energia,
        f'{prefixo}_freq_dom': freq_dominante
    }


# ===============================================
# PROCESSAMENTO DOS ARQUIVOS E JANELAMENTO
# ===============================================

dados_totais = []

for arq_in, classe in arquivos_classes.items():
    caminho_in = os.path.join(DIR_INTERPOLADO, arq_in)
    
    if not os.path.exists(caminho_in):
        print(f"Arquivo não encontrado: {caminho_in}")
        continue

    df = pd.read_csv(caminho_in)

    df['Mag'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)

    n_amostras = len(df)
    n_janelas = 0

    for i in range(0, n_amostras - TAMANHO_JANELA + 1, PASSO):
        if n_janelas >= MAX_JANELAS:
            break

        janela = df.iloc[i : i + TAMANHO_JANELA]

        # Extrai para X, Y, Z e Magnitude
        feat_x = extrair_caracteristicas_janela(janela['X'].values, 'X')
        feat_y = extrair_caracteristicas_janela(janela['Y'].values, 'Y')
        feat_z = extrair_caracteristicas_janela(janela['Z'].values, 'Z')
        feat_mag = extrair_caracteristicas_janela(janela['Mag'].values, 'Mag')

        linha_feat = {}
        linha_feat.update(feat_x)
        linha_feat.update(feat_y)
        linha_feat.update(feat_z)
        linha_feat.update(feat_mag)
        linha_feat['label'] = classe

        dados_totais.append(linha_feat)
        n_janelas += 1

    print(f" {arq_in}: {n_janelas} janelas processadas (Limite: {MAX_JANELAS}).")


# =======================================
# EXPORTAÇÃO DO DATASET CONSOLIDADO
# =======================================

df_final = pd.DataFrame(dados_totais)
caminho_out = os.path.join(DIR_PROCESSADO, "dataset_caracteristicas.csv")
df_final.to_csv(caminho_out, index=False)

print(f"\nDataset salvo com sucesso em: {caminho_out}")
print(f" Dimensão total: {df_final.shape[0]} amostras x {df_final.shape[1]} colunas.")