#!/usr/bin/env python3
"""
compare_hgb_optimizer.py

Compara as predições do modelo HGB com os valores reais do otimizador
para as horas 16, 17 e 18. Utiliza cenários aleatórios com curtailment > 0.
"""

import numpy as np
import json
import pandas as pd
import os
import sqlite3
import joblib
import sys
import time
import traceback
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==================== CONFIGURAÇÕES ====================
SISTEMA = 'IEEE14'
TIPO_TESTE = 'ACOPLADO'

# Caminhos
JSON_PATH = f"DATA/output_CLAGTEE_Oficial/{SISTEMA}_BASE.json"
DB_PATH = f"DATA/output_CLAGTEE_Oficial/{SISTEMA}_RNA_DATA_ACOPF_{TIPO_TESTE}.db"
# ATENÇÃO: altere para o diretório onde estão os modelos HGB
MODELS_DIR = f"DATA/output_CLAGTEE_Oficial/{SISTEMA}_{TIPO_TESTE}_modelos_especialistas_v7_HGB"

HORAS_INTERESSE = [16, 17, 18]

BARRAS_COM_MEDICAO = [2, 3, 4, 6, 9, 14]       # barras que possuem medição real (PLOAD_medido)
LINHAS_COM_MEDICAO = ["4-7", "4-9", "5-6"]   # linhas a serem incluídas como features



# BARRAS_COM_MEDICAO = [59, 116, 90, 80, 54, 42, 15, 49, 56, 60]
# LINHAS_COM_MEDICAO = [
#     "8-5", "26-25", "30-17", "38-37",
#     "63-59", "64-61", "65-66", "81-80", "68-69"
# ]

# Controle de plotagem
SAVE_FIG = True
OUTPUT_DIR = "DATA/output_CLAGTEE_Oficial/graficos_comparacao_HGB"

CENARIO_EXISTENTE = None   # Deixe None para buscar aleatórios
# ========================================================

# Ajuste de path para módulos do projeto
current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_path = current_path[:-6]
src_path = current_path.replace("IHM", "SRC")
sys.path.append(src_path)

from UTILS.SystemLoader import SistemaLoader
from DB.DBhandler_OPF import OPF_DBHandler
from UTILS.EvaluateFactors import EvaluateFactors
from SOLVER.OPF_AC.AC_OPF_Acoplado import TimeCoupled_OPF_Result


# ========== Funções de preparação de dados ==========

def load_data(db_path, cen_id=None):
    """Carrega os dados do banco SQLite."""
    conn = sqlite3.connect(db_path)
    query = '''
        SELECT cen_id, data_simulacao, hora_simulacao, dia_semana,
               BAR_id, PLOAD_cenario, QLOAD_cenario,
               PGWIND_disponivel_cenario, PGER_UTE_result, QGER_UTE_result,
               CURTAILMENT_total_result, V_result
        FROM DBAR_results
    '''
    if cen_id is not None:
        query += " WHERE cen_id = ?"
        df = pd.read_sql_query(query, conn, params=(cen_id,))
    else:
        df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_line_data(db_path, cen_id=None):
    """Carrega dados de uso das linhas da tabela DLIN_results."""
    conn = sqlite3.connect(db_path)
    try:
        query = 'SELECT cen_id, de_barra, para_barra, LIN_usage_result FROM DLIN_results'
        if cen_id is not None:
            query += ' WHERE cen_id = ?'
            df_lines = pd.read_sql_query(query, conn, params=(cen_id,))
        else:
            df_lines = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"   Aviso: não foi possível carregar dados de linha ({e}).")
        df_lines = pd.DataFrame()
    finally:
        conn.close()
    return df_lines


def create_wide_format(df, barras_com_medicao):
    """Transforma dados longos em formato largo (pivot)."""
    df = df.copy()
    df['PLOAD_medido'] = 0.0
    df['QLOAD_medido'] = 0.0
    df['V_medido'] = 0.0

    mask_medido = df['BAR_id'].isin(barras_com_medicao)
    df.loc[mask_medido, 'PLOAD_medido'] = df.loc[mask_medido, 'PLOAD_cenario']
    df.loc[mask_medido, 'QLOAD_medido'] = df.loc[mask_medido, 'QLOAD_cenario']
    df.loc[mask_medido, 'V_medido'] = df.loc[mask_medido, 'V_result']

    pivot_cols = [
        'PGWIND_disponivel_cenario',
        'PGER_UTE_result',
        'QGER_UTE_result',
        'V_medido',
        'PLOAD_medido',
        'QLOAD_medido',
        'CURTAILMENT_total_result'
    ]
    index_cols = ['cen_id', 'data_simulacao', 'hora_simulacao', 'dia_semana']
    df_pivot = df.pivot_table(index=index_cols, columns='BAR_id', values=pivot_cols)
    df_pivot.columns = [f'{var}_BAR{bar}' for var, bar in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    return df_pivot


def carregar_modelos():
    """Carrega os pipelines e metadados dos modelos HGB."""
    models = {}
    for hora in HORAS_INTERESSE:
        model_path = os.path.join(MODELS_DIR, f"hora_{hora:02d}", "pipeline.joblib")
        metadata_path = os.path.join(MODELS_DIR, f"hora_{hora:02d}", "metadata.json")

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            print(f"   Modelo para hora {hora} não encontrado. Ignorando.")
            continue

        pipeline = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        models[hora] = {
            'pipeline': pipeline,
            'feature_names': metadata['feature_names'],
            'target_names': metadata['target_names']
        }
        print(f"   Hora {hora:02d}: {len(metadata['feature_names'])} features, "
              f"{len(metadata['target_names'])} targets")
    return models


def get_random_cenario(db_path, hora):
    """Retorna um cenário aleatório com curtailment > 0 na hora especificada."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = """
        SELECT cen_id
        FROM DBAR_results
        WHERE hora_simulacao = ?
        GROUP BY cen_id
        HAVING SUM(COALESCE(CURTAILMENT_total_result, 0)) > 0
    """
    cursor.execute(query, (hora,))
    rows = cursor.fetchall()
    conn.close()
    if rows:
        escolhido = random.choice(rows)[0]
        print(f"   Cenário escolhido: {escolhido}")
        return escolhido
    return None


def extrair_comparacao(cen_id, models, hora_alvo=None):
    """Prepara os dados e gera predições para um cenário e hora(s)."""
    print(f"\n[COMPARAÇÃO] Carregando dados do cenário {cen_id}...")
    df = load_data(DB_PATH, cen_id=cen_id)
    if df.empty:
        print("   Nenhum dado encontrado.")
        return {}

    df_wide = create_wide_format(df, BARRAS_COM_MEDICAO)
    df_lines = load_line_data(DB_PATH, cen_id=cen_id)

    if not df_lines.empty:
        # Cria identificadores de linha normalizados
        df_lines_1 = df_lines.copy()
        df_lines_1['line_id'] = df_lines_1['de_barra'].astype(str) + '-' + df_lines_1['para_barra'].astype(str)
        df_lines_2 = df_lines.copy()
        df_lines_2['line_id'] = df_lines_2['para_barra'].astype(str) + '-' + df_lines_2['de_barra'].astype(str)
        df_lines_all = pd.concat([df_lines_1, df_lines_2], ignore_index=True)

        df_lines_wide = df_lines_all.pivot_table(
            index='cen_id', columns='line_id', values='LIN_usage_result', aggfunc='first'
        )
        df_lines_wide.columns = [f'LIN_usage_result_{col}' for col in df_lines_wide.columns]
        df_lines_wide = df_lines_wide.reset_index()
        df_wide = df_wide.merge(df_lines_wide, on='cen_id', how='left')
        # Preenche NaN com 0
        lin_cols = [c for c in df_wide.columns if c.startswith('LIN_usage_result_')]
        df_wide[lin_cols] = df_wide[lin_cols].fillna(0.0)

    resultados = {}
    horas_processar = [hora_alvo] if hora_alvo is not None else HORAS_INTERESSE

    for hora in horas_processar:
        if hora not in models:
            continue

        model_data = models[hora]
        pipeline = model_data['pipeline']
        feature_names = model_data['feature_names']
        target_names = model_data['target_names']

        df_hora = df_wide[df_wide['hora_simulacao'] == hora]
        if df_hora.empty:
            print(f"   Hora {hora:02d}: sem dados.")
            continue

        if len(df_hora) > 1:
            idx = np.random.randint(len(df_hora))
            df_hora = df_hora.iloc[[idx]]
            print(f"   Hora {hora:02d}: usando amostra aleatória (índice {idx}).")
        else:
            print(f"   Hora {hora:02d}: única amostra disponível.")

        # Monta vetor de features
        X_list = []
        faltantes = []
        for feat in feature_names:
            if feat in df_hora.columns:
                X_list.append(df_hora[feat].iloc[0])
            else:
                X_list.append(0.0)
                faltantes.append(feat)
        if faltantes:
            print(f"   Aviso: {len(faltantes)} features ausentes (preenchidas com 0). Exemplos: {faltantes[:5]}")

        X = np.array(X_list).reshape(1, -1)

        # Targets reais
        y_true_list = []
        for tgt in target_names:
            if tgt in df_hora.columns:
                y_true_list.append(df_hora[tgt].iloc[0])
            else:
                print(f"   Erro: target '{tgt}' não encontrado. Pulando hora {hora}.")
                break
        else:
            y_true = np.array(y_true_list).reshape(1, -1)
            y_pred = pipeline.predict(X)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(1, -1)

            resultados[hora] = {
                'y_true': y_true,
                'y_pred': y_pred,
                'target_names': target_names
            }
            print(f"   Hora {hora:02d}: predição concluída. Shapes: X={X.shape}, y_true={y_true.shape}, y_pred={y_pred.shape}")

    return resultados


def print_detalhes_comparacao(resultados):
    """Imprime tabela detalhada de diferenças e métricas MAE/RMSE."""
    for hora in sorted(resultados.keys()):
        data = resultados[hora]
        y_true = data['y_true'][0]
        y_pred = data['y_pred'][0]
        targets = data['target_names']

        print(f"\n{'='*60}")
        print(f"DETALHES PARA HORA {hora:02d}")
        print(f"{'='*60}")

        erro_abs = np.abs(y_true - y_pred)
        # Evita divisão por zero no percentual
        with np.errstate(divide='ignore', invalid='ignore'):
            erro_pct = np.where(y_true > 1e-6, erro_abs / y_true * 100, 0.0)

        df = pd.DataFrame({
            'Target': targets,
            'Real': y_true,
            'Previsto': y_pred,
            'Diferença (abs)': erro_abs,
            'Diferença (%)': erro_pct
        })
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.float_format', '{:.6f}'.format)
        print(df.to_string(index=False))

        # MAE e RMSE apenas nas barras onde o curtailment real > 0
        mask_pos = y_true > 1e-6
        if mask_pos.any():
            mae = mean_absolute_error(y_true[mask_pos], y_pred[mask_pos])
            rmse = np.sqrt(mean_squared_error(y_true[mask_pos], y_pred[mask_pos]))
            print(f"\n   MAE (curt > 0): {mae:.4f} MW")
            print(f"   RMSE (curt > 0): {rmse:.4f} MW")
        else:
            print("\n   Nenhum curtailment positivo nesta amostra.")
        print()


def plot_comparacao_barras(resultados, hora, output_dir, save_fig):
    """Gera gráfico de barras comparando real vs previsto para curtailment."""
    if hora not in resultados:
        return

    data = resultados[hora]
    y_true = data['y_true'][0]
    y_pred = data['y_pred'][0]
    targets = data['target_names']

    # Filtra apenas targets de curtailment
    curt_indices = [i for i, t in enumerate(targets) if 'CURTAILMENT' in t]
    if not curt_indices:
        return

    true_vals = y_true[curt_indices]
    pred_vals = y_pred[curt_indices]
    barras = [t.replace('CURTAILMENT_total_result_BAR', '') for t in np.array(targets)[curt_indices]]

    fig, ax = plt.subplots(figsize=(max(6, len(barras) * 0.8), 5))
    x = np.arange(len(barras))
    width = 0.35

    ax.bar(x - width/2, true_vals, width, label='Real', color='steelblue')
    ax.bar(x + width/2, pred_vals, width, label='Previsto (HGB)', color='orange')

    ax.set_xlabel('Barra')
    ax.set_ylabel('Corte Eólico (MW)')
    ax.set_title(f'Hora {hora:02d} - CURTAILMENT')
    ax.set_xticks(x)
    ax.set_xticklabels(barras)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f'curtailment_hgb_hora_{hora:02d}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"   Gráfico salvo: {fname}")
    else:
        plt.show()
    plt.close()


def main():
    print("=" * 70)
    print("COMPARAÇÃO HGB vs OTIMIZADOR - CURTAILMENT")
    print("=" * 70)

    # 1. Carregar modelos
    print("\n[1] Carregando modelos HGB...")
    models = carregar_modelos()
    if not models:
        print("Nenhum modelo válido encontrado. Abortando.")
        return 1

    # 2. Para cada hora, obter cenário e extrair comparação
    resultados = {}
    for hora in HORAS_INTERESSE:
        print(f"\n[2] Buscando cenário para hora {hora:02d}...")
        try:
            if CENARIO_EXISTENTE:
                cen_id = CENARIO_EXISTENTE
                print(f"   Usando cenário existente: {cen_id}")
            else:
                cen_id = get_random_cenario(DB_PATH, hora)
                if cen_id is None:
                    print(f"   Nenhum cenário com curtailment > 0 para hora {hora:02d}. Pulando.")
                    continue

            res = extrair_comparacao(cen_id, models, hora_alvo=hora)
            if res:
                resultados.update(res)
        except Exception as e:
            print(f"   [!] Erro ao processar hora {hora:02d}: {e}")
            traceback.print_exc()
            continue

    if not resultados:
        print("Nenhum dado de comparação obtido. Abortando.")
        return 1

    # 3. Imprimir detalhes e métricas
    print("\n[3] Resultados detalhados:")
    print_detalhes_comparacao(resultados)

    # 4. Gráficos
    print("\n[4] Gerando gráficos comparativos...")
    for hora in HORAS_INTERESSE:
        plot_comparacao_barras(resultados, hora, OUTPUT_DIR, SAVE_FIG)

    print("\n" + "=" * 70)
    print("PROCESSO CONCLUÍDO")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())