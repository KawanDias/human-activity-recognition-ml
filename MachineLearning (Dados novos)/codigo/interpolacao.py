# ============================
# IMPORTAÇÃO E BIBLIOTECAS
# ============================

import os
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURAÇÃO DE DIRETÓRIOS E ARQUIVOS
# ==========================================

DIR_BRUTO = "dados/bruto"
DIR_INTERPOLADO = "dados/interpolados"

os.makedirs(DIR_INTERPOLADO, exist_ok=True)

arquivos = {
    "Repouso.csv": "repouso_interpolado.csv",
    "GuardarObjeto.csv": "guardar_objeto_interpolado.csv",
    "VirarPagina.csv": "virar_pagina_interpolado.csv",
    "VirarPaginaNovo.csv": "virar_pagina_novo_interpolado.csv", 
}

# Frequência de amostragem 100hz
FS = 100


# =========================================
# PROCESSAMENTO E INTERPOLAÇÃO TEMPORAL
# =========================================

for arq_in, arq_out in arquivos.items():
    caminho_in = os.path.join(DIR_BRUTO, arq_in)
    
    if not os.path.exists(caminho_in):
        print(f"Arquivo não encontrado: {caminho_in}")
        continue

    df = pd.read_csv(caminho_in)

    # ===========================================
    # CORREÇÃO E TRATAMENTO DA ESCALA DE TEMPO
    # ===========================================

    if 'seconds_elapsed' in df.columns:
        df['tempo_sec'] = df['seconds_elapsed']
    elif 'time' in df.columns:
        t_vals = df['time'].values
        if t_vals[0] > 1e11:  
            df['tempo_sec'] = (t_vals - t_vals[0]) / 1e9
        else:
            df['tempo_sec'] = t_vals - t_vals[0]
    else:
        col_t = df.columns[0]
        df['tempo_sec'] = df[col_t] - df[col_t].min()

    col_tempo = 'tempo_sec'
    col_x = 'x' if 'x' in df.columns else 'X'
    col_y = 'y' if 'y' in df.columns else 'Y'
    col_z = 'z' if 'z' in df.columns else 'Z'

    # =========================
    # TRATAMENTO E LIMPEZA
    # =========================
    df = df.sort_values(col_tempo).drop_duplicates(subset=[col_tempo])

    t_min = df[col_tempo].min()
    t_max = df[col_tempo].max()
    duracao = t_max - t_min

    # ================================
    # REAMOSTRAGEM UNIFORME (100 Hz)
    # =================================

    num_amostras = int(duracao * FS)
    tempo_regular = np.linspace(t_min, t_max, num_amostras)

    x_interp = np.interp(tempo_regular, df[col_tempo], df[col_x])
    y_interp = np.interp(tempo_regular, df[col_tempo], df[col_y])
    z_interp = np.interp(tempo_regular, df[col_tempo], df[col_z])

    # ==========================
    # MONTAGEM E EXPORTAÇÃO
    # ==========================
    df_interp = pd.DataFrame({
        'timestamp': tempo_regular - t_min,
        'X': x_interp,
        'Y': y_interp,
        'Z': z_interp
    })

    caminho_out = os.path.join(DIR_INTERPOLADO, arq_out)
    df_interp.to_csv(caminho_out, index=False)
    print(f" {arq_out} gerado com sucesso: {len(df_interp)} amostras (~{duracao:.1f}s).")