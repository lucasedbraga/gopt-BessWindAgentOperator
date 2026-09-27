#!/usr/bin/env python3
"""
compare_rna_optimizer.py

Gera um cenário (ou utiliza um existente) e compara, para as horas 16, 17 e 18,
os valores reais (otimizador) com as predições da RNA.
Produz gráficos de barras comparativos para CURTAILMENT.
Agora também imprime uma tabela detalhada com diferenças.
Se nenhum cenário for especificado, busca um cenário aleatório com curtailment > 0 para cada hora.
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
from datetime import datetime
import secrets
from contextlib import contextmanager
import matplotlib.pyplot as plt
import random

# ==================== CONFIGURAÇÕES ====================
SISTEMA='IEEE118'
TIPO_TESTE = 'ACOPLADO'
# Caminhos
JSON_PATH = f"DATA/output_CLAGTEE_Oficial/{SISTEMA}_BASE.json"
DB_PATH = f"DATA/output_CLAGTEE_Oficial/{SISTEMA}_RNA_DATA_ACOPF_{TIPO_TESTE}.db"
MODELS_DIR = f"DATA/output_CLAGTEE_Oficial/{SISTEMA}_{TIPO_TESTE}_modelos_especialistas_v7"

# Horas para as quais existem modelos treinados e que queremos comparar
HORAS_INTERESSE = [16, 17, 18]

# Barras com medição (para create_wide_format)
BARRAS_COM_MEDICAO = [59, 116, 90, 80, 54, 42, 15, 49, 56, 60]
LINHAS_COM_MEDICAO = [
    "8-5",
    "26-25",
    "30-17",
    "38-37",
    "63-59",
    "64-61",
    "65-66",
    "81-80",
    "68-69"
]

# Parâmetros da geração de cenários (usados apenas se não houver nenhum cenário no banco)
N_ITERACOES = 5
N_DIAS = 1
N_HORAS = 24
SOC_INICIAL_FRACAO = 0.5
SOC_FINAL_FRACAO = 0.5
CONSIDERAR_PERDAS = False
SOLVER_NAME = 'highs'
TOL = 1e-4
MAX_ITER = 5
WRITE_LP = False

# Controle de plotagem
SAVE_FIG = True
OUTPUT_DIR = "DATA/output_CLAGTEE_Oficial/graficos_comparacao"

# Se você já tem um cenário no banco e quer usá-lo, defina o ID aqui.
CENARIO_EXISTENTE = None   # Exemplo: "20250310143000_00000"
# ========================================================

# Ajusta o path para encontrar os módulos do projeto
current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_path = current_path[:-6]
src_path = current_path.replace("IHM", "SRC")
sys.path.append(src_path)

from UTILS.SystemLoader import SistemaLoader
from DB.DBhandler_OPF import OPF_DBHandler
from UTILS.EvaluateFactors import EvaluateFactors
from SOLVER.OPF_AC.AC_OPF_Acoplado import TimeCoupled_OPF_Result
from SOLVER.OPF_AC.AC_OPF_Snapshot import ACOPF_Snapshot_Model


# ========== Funções de preparação de dados ==========

def load_data(db_path, cen_id=None):
    """Carrega os dados do banco SQLite. Se cen_id for fornecido, filtra por ele."""
    conn = sqlite3.connect(db_path)
    query = '''
        SELECT cen_id,
               data_simulacao,
               hora_simulacao,
               dia_semana,
               BAR_id,
               PLOAD_cenario,
               QLOAD_cenario,
               BESS_init_cenario,
               PGWIND_disponivel_cenario,
               PGER_UTE_result,
               QGER_UTE_result,
               CURTAILMENT_total_result,
               BESS_operation_result,
               V_result
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
    """
    Carrega dados de fluxo/uso das linhas.
    Ajuste os nomes da tabela e coluna conforme seu banco:
    - Tabela: 'DLIN_results' (ou 'DLIN_results')
    - Coluna identificadora: 'linha_id' (ou 'linha_id')
    """
    conn = sqlite3.connect(db_path)
    try:
        # Tente carregar da tabela DLIN_results (nome mais comum no projeto)
        query = 'SELECT cen_id, linha_id, de_barra, para_barra, LIN_usage_result FROM DLIN_results'
        if cen_id is not None:
            query += ' WHERE cen_id = ?'
            df_lines = pd.read_sql_query(query, conn, params=(cen_id,))
        else:
            df_lines = pd.read_sql_query(query, conn)
    except Exception as e:
        # Se a tabela/coluna não existir, retorna vazio
        print(f"   Aviso: não foi possível carregar dados de linha ({e}). Features de linha serão ignoradas.")
        df_lines = pd.DataFrame()
    finally:
        conn.close()
    return df_lines


def create_wide_format(df, barras_com_medicao):
    df = df.copy()

    df['PLOAD_medido'] = 0.0
    df['QLOAD_medido'] = 0.0
    df['V_medido'] = 0.0

    mask_medido = df['BAR_id'].isin(barras_com_medicao)

    df.loc[mask_medido, 'PLOAD_medido'] = df.loc[mask_medido, 'PLOAD_cenario']
    df.loc[mask_medido, 'QLOAD_medido'] = df.loc[mask_medido, 'QLOAD_cenario']
    df.loc[mask_medido, 'V_medido'] = df.loc[mask_medido, 'V_result']

    #df.loc[~mask_medido, 'PLOAD_estimado'] = df.loc[~mask_medido, 'PLOAD_cenario']
    #df.loc[~mask_medido, 'QLOAD_estimado'] = df.loc[~mask_medido, 'QLOAD_cenario']
    #df.loc[~mask_medido, 'V_estimado'] = df.loc[~mask_medido, 'V_result']
    #df.loc[~mask_medido, 'ANG_estimado'] = df.loc[~mask_medido, 'ANG_result']

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


def prepare_X_y(df_wide, remove_constants=True):
    """Separa features (X) e targets (y)."""
    feature_prefixes = ['PGWIND_disponivel_cenario',
                        'PGER_UTE_result',
                        'QGER_UTE_result', 
                        'PLOAD_medido',
                        'QLOAD_medido',
                        'V_medido'
                        ]
    
    extra_feature_cols = [col for col in df_wide.columns if 'FLUX_result' in col or 'LIN_usage_result' in col]
    feature_cols = [col for col in df_wide.columns if any(col.startswith(p) for p in feature_prefixes)] + extra_feature_cols

    df_clean = df_wide.dropna(subset=feature_cols)
    X = df_clean[feature_cols].copy()

    target_prefixes = ['CURTAILMENT_total_result']
    target_cols = [col for col in df_clean.columns if any(col.startswith(p) for p in target_prefixes)]
    y = df_clean[target_cols].copy()

    if remove_constants:
        constant_X = X.columns[X.std() == 0].tolist()
        if constant_X:
            X = X.drop(columns=constant_X)
            wind_cols = [col for col in X.columns if col.startswith('PGWIND_disponivel_cenario_')]
            curt_cols = [col.replace('PGWIND_disponivel_cenario_', 'CURTAILMENT_total_result_') for col in wind_cols]
            target_cols = curt_cols
            y = df_clean[target_cols].copy()

    print(f"   Features shape: {X.shape}, Targets shape: {y.shape}")
    return X, y


def get_target_names_from_features(feature_names):
    """Deriva os nomes dos targets a partir dos nomes das features."""
    target_names = []
    for col in feature_names:
        if '_BAR' not in col:
            continue
        base, bar = col.rsplit('_BAR', 1)
        if base.startswith('BESS_init_cenario'):
            target_names.append(f'BESS_operation_result_BAR{bar}')
        elif base.startswith('PGWIND_disponivel_cenario'):
            target_names.append(f'CURTAILMENT_total_result_BAR{bar}')
    return list(dict.fromkeys(target_names))


# ========== Função para carregar modelos ==========
def carregar_modelos():
    models = {}
    for hora in HORAS_INTERESSE:
        model_path = os.path.join(MODELS_DIR, f"hora_{hora:02d}", "pipeline.joblib")
        metadata_path = os.path.join(MODELS_DIR, f"hora_{hora:02d}", "metadata.json")

        if not os.path.exists(model_path):
            print(f"   Modelo para hora {hora} não encontrado em {model_path}. Ignorando.")
            continue
        if not os.path.exists(metadata_path):
            print(f"   Metadata para hora {hora} não encontrado. Ignorando.")
            continue

        pipeline = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        feature_names = metadata['feature_names']
        target_names = metadata['target_names']

        models[hora] = {
            'pipeline': pipeline,
            'feature_names': feature_names,
            'target_names': target_names
        }
        print(f"   Hora {hora:02d} carregada: {len(feature_names)} features, {len(target_names)} targets")
    return models


# ========== Função para obter cenário aleatório por hora (com curtailment > 0) ==========
def get_random_cenario(db_path, hora):
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

    print(f"   [DEBUG] Hora {hora:02d}: {len(rows)} cenários com curtailment positivo encontrados.")
    if rows:
        escolhido = random.choice(rows)[0]
        print(f"   Cenário escolhido: {escolhido}")
        return escolhido

    return None


# ========== Função para gerar um novo cenário ==========
def gerar_cenario_unico_ACOPLADO():
    print("\n[GERAÇÃO] Gerando um novo cenário...")
    if not os.path.exists(JSON_PATH):
        print(f"ERRO: Arquivo do sistema não encontrado: {JSON_PATH}")
        return None

    sistema = SistemaLoader(JSON_PATH)
    db_handler = OPF_DBHandler(DB_PATH)
    db_handler.create_tables()

    modelo = TimeCoupled_OPF_Result(
        sistema=sistema,
        n_horas=N_HORAS,
        n_dias=N_DIAS,
        db_handler=db_handler,
        dia_inicial=0
    )

    try:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        cen_id = f"{timestamp}_00000"

        seed = (int(time.time() * 1e6)) % 1000
        avaliador = EvaluateFactors(
            sistema=sistema,
            n_dias=N_DIAS,
            n_horas=N_HORAS,
            carga_incerteza=0.2,
            vento_variacao=0.1,
            seed=seed
        )
        fatores_carga, fatores_vento = avaliador.gerar_tudo()

        print(f"   Gerando cenário {cen_id}...")
        _ = modelo.solve_multiday(
            solver_name=SOLVER_NAME,
            fator_carga=fatores_carga,
            fator_vento=fatores_vento,
            soc_inicial=SOC_INICIAL_FRACAO,
            soc_final=SOC_FINAL_FRACAO,
            cen_id=cen_id,
            tol=TOL,
            max_iter=MAX_ITER,
            write_lp=WRITE_LP
        )
        print(f"   Cenário {cen_id} gerado com sucesso.")
        return cen_id
    except Exception as e:
        print(f"   [!] Erro na geração: {e}")
        traceback.print_exc()
        return None

def gerar_cenario_unico_SNAPSHOT():
    print("\n[GERAÇÃO] Gerando um novo cenário...")
    if not os.path.exists(JSON_PATH):
        print(f"ERRO: Arquivo do sistema não encontrado: {JSON_PATH}")
        return None

    sistema = SistemaLoader(JSON_PATH)
    db_handler = OPF_DBHandler(DB_PATH)
    db_handler.create_tables()

    modelo = ACOPF_Snapshot_Model(sistema=sistema, db_handler=db_handler)

    try:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        cen_id = f"{timestamp}_00000"

        seed = (int(time.time() * 1e6)) % 1000
        avaliador = EvaluateFactors(
            sistema=sistema,
            n_dias=1,
            n_horas=1,
            carga_incerteza=0.2,
            vento_variacao=0.1,
            seed=seed
        )
        fatores_carga_completo, fatores_vento_completo = avaliador.gerar_tudo()

        fator_carga_hora = fatores_carga_completo[0, 0, :]
        fator_vento_hora = fatores_vento_completo[0, 0, :] if sistema.NGER_GWD > 0 else 1.0
        soc_baterias = {b: 0.5 for b in sistema.BARRAS_COM_BATERIA}
        print(f"   Gerando cenário {cen_id}...")
        _ = modelo.solve_snapshot(
        solver_name='ipopt',
        fator_carga=fator_carga_hora,
        fator_vento=fator_vento_hora,
        soc_baterias=soc_baterias,
        hora=HORAS_INTERESSE,
        dia=0,
        cen_id=cen_id
    )
        print(f"   Cenário {cen_id} gerado com sucesso.")
        return cen_id
    except Exception as e:
        print(f"   [!] Erro na geração: {e}")
        traceback.print_exc()
        return None


# ========== Função para extrair dados de comparação (com mescla de linhas) ==========
def extrair_comparacao(cen_id, models, hora_alvo=None):
    """
    Se hora_alvo for int, processa apenas aquela hora; senão processa HORAS_INTERESSE.
    """
    print(f"\n[COMPARAÇÃO] Carregando dados do cenário {cen_id}...")
    df = load_data(DB_PATH, cen_id=cen_id)
    if df.empty:
        print(f"   Nenhum dado encontrado para cenário {cen_id}")
        return {}

    df_wide = create_wide_format(df, BARRAS_COM_MEDICAO)
    df_lines = load_line_data(DB_PATH, cen_id=cen_id)

    if not df_lines.empty:
        # Cria as duas orientações possíveis para cada linha (ex: "17-30" e "30-17")
        df_lines_1 = df_lines.copy()
        df_lines_1['line_id'] = df_lines_1['de_barra'].astype(str) + '-' + df_lines_1['para_barra'].astype(str)
    
        df_lines_2 = df_lines.copy()
        df_lines_2['line_id'] = df_lines_2['para_barra'].astype(str) + '-' + df_lines_2['de_barra'].astype(str)
        
        df_lines_all = pd.concat([df_lines_1, df_lines_2], ignore_index=True)
        
        # Pivoteia com a coluna line_id
        df_lines_wide = df_lines_all.pivot_table(
            index='cen_id',
            columns='line_id',
            values='LIN_usage_result',
            aggfunc='first'
        )
        df_lines_wide.columns = [f'LIN_usage_result_{col}' for col in df_lines_wide.columns]
        df_lines_wide = df_lines_wide.reset_index()
        
        # Mescla com o DataFrame principal
        df_wide = df_wide.merge(df_lines_wide, on='cen_id', how='left')
    else:
        print("   Aviso: nenhum dado de linha carregado. Features de linha ficarão zeradas.")

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
            print(f"   Hora {hora:02d}: sem dados neste cenário.")
            continue

        if len(df_hora) > 1:
            idx_aleatorio = np.random.randint(0, len(df_hora))
            df_hora = df_hora.iloc[[idx_aleatorio]]
            print(f"   Hora {hora:02d}: selecionada linha aleatória {idx_aleatorio} de {len(df_hora)} disponíveis")
        else:
            print(f"   Hora {hora:02d}: apenas 1 linha disponível")

        X_values = []
        features_faltando = []
        for feat in feature_names:
            if feat in df_hora.columns:
                X_values.append(df_hora[feat].iloc[0])
            else:
                X_values.append(0.0)
                features_faltando.append(feat)
        if features_faltando:
            print(f"   Aviso: features ausentes preenchidas com 0: {features_faltando}")

        X = np.array(X_values).reshape(1, -1)
        features_faltando = [f for f in feature_names if f not in df_hora.columns]
        print(f"   Features esperadas pelo modelo: {len(feature_names)}")
        print(f"   Features faltando (preenchidas com 0): {len(features_faltando)}")
        if features_faltando:
            print(f"   Exemplos: {features_faltando[:10]}")

        y_true_values = []
        targets_faltando = []
        for tgt in target_names:
            if tgt in df_hora.columns:
                y_true_values.append(df_hora[tgt].iloc[0])
            else:
                targets_faltando.append(tgt)
        if targets_faltando:
            print(f"   Erro: targets ausentes: {targets_faltando}. Pulando hora {hora}.")
            continue

        y_true = np.array(y_true_values).reshape(1, -1)
        print(f"   X contém NaN? {np.any(np.isnan(X))}")
        print(f"   Valores de X (primeiros 10): {X[0,:10]}")
        y_pred = pipeline.predict(X)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)

        resultados[hora] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'target_names': target_names
        }
        print(f"   Hora {hora:02d}: amostra processada. Shapes X:{X.shape} y_true:{y_true.shape} y_pred:{y_pred.shape}")

    return resultados


# ========== Função para imprimir detalhes ==========
def print_detalhes_comparacao(resultados):
    for hora in sorted(resultados.keys()):
        data = resultados[hora]
        y_true = data['y_true'][0]
        y_pred = data['y_pred'][0]
        target_names = data['target_names']

        print(f"\n{'='*60}")
        print(f"DETALHES PARA HORA {hora:02d}")
        print(f"{'='*60}")

        erro_abs = np.abs(y_true - y_pred)
        df_detalhes = pd.DataFrame({
            'Target': target_names,
            'Real': y_true,
            'Previsto': y_pred,
            'Diferença (abs)': erro_abs,
            'Diferença (%)': np.where(
                np.abs(y_true) > 1e-6,
                erro_abs / np.abs(y_true) * 100,
                0.0   # evita NaN/divisão por zero
            )
        })
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.float_format', '{:.6f}'.format)
        print(df_detalhes.to_string(index=False))
        print()


# ========== Funções de plotagem ==========
def plot_comparacao_barras(resultados, hora, output_dir, save_fig):
    if hora not in resultados:
        return

    data = resultados[hora]
    y_true = data['y_true'][0]
    y_pred = data['y_pred'][0]
    target_names = data['target_names']

    grupos = {}
    for i, name in enumerate(target_names):
        if 'BESS_operation' in name:
            key = 'BESS_operation'
        elif 'CURTAILMENT' in name:
            key = 'CURTAILMENT'
        elif 'PLOAD_estimado' in name:
            key = 'PLOAD_estimado'
        else:
            key = 'Outros'
        grupos.setdefault(key, []).append(i)

    for grupo, indices in grupos.items():
        if not indices:
            continue
        true_vals = y_true[indices]
        pred_vals = y_pred[indices]
        nomes = [target_names[i] for i in indices]

        if grupo == 'PLOAD_estimado':
            barras = [n.replace('PLOAD_estimado_BAR', '') for n in nomes]
        elif grupo == 'BESS_operation':
            barras = [n.replace('BESS_operation_result_BAR', '') for n in nomes]
        elif grupo == 'CURTAILMENT':
            barras = [n.replace('CURTAILMENT_total_result_BAR', '') for n in nomes]
        else:
            barras = nomes

        fig, ax = plt.subplots(figsize=(max(6, len(barras) * 0.8), 5))
        x = np.arange(len(barras))
        width = 0.35

        ax.bar(x - width/2, true_vals, width, label='Real', color='steelblue')
        ax.bar(x + width/2, pred_vals, width, label='Previsto', color='orange')

        ax.set_xlabel('Barra')
        ax.set_ylabel('Potência (MW)')
        if grupo == 'BESS_operation':
            titulo = f'Hora {hora:02d} - Operação da Bateria (BESS)'
        elif grupo == 'CURTAILMENT':
            titulo = f'Hora {hora:02d} - Corte Eólico (CURTAILMENT)'
        elif grupo == 'PLOAD_estimado':
            titulo = f'Hora {hora:02d} - Carga Estimada (PLOAD_estimado)'
        else:
            titulo = f'Hora {hora:02d} - {grupo}'

        ax.set_title(titulo)
        ax.set_xticks(x)
        ax.set_xticklabels(barras)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_fig:
            os.makedirs(output_dir, exist_ok=True)
            fname = os.path.join(output_dir, f'{SISTEMA}_{TIPO_TESTE}_{grupo.lower()}_comparison_hora_{hora:02d}.png')
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            print(f"   Gráfico salvo: {fname}")


# ========== Função principal ==========
def main():
    print("=" * 70)
    print("COMPARAÇÃO RNA vs OTIMIZADOR - HORAS 16, 17 e 18")
    print("=" * 70)

    # 1. Carregar modelos
    print("\n[1] Carregando modelos...")
    models = carregar_modelos()
    if not models:
        print("Nenhum modelo válido carregado. Abortando.")
        return 1

    resultados = {}  # Acumula dados de todas as horas

    # 2. Para cada hora, buscar um cenário aleatório e extrair comparação
    for hora in HORAS_INTERESSE:
        print(f"\n[2] Buscando cenário aleatório para hora {hora:02d}...")
        try:
            if CENARIO_EXISTENTE:
                cen_id = CENARIO_EXISTENTE
                print(f"   Usando cenário existente: {cen_id}")
            else:
                cen_id = get_random_cenario(DB_PATH, hora)
                if cen_id is None:
                    print(f"   Nenhum cenário com curtailment positivo para hora {hora:02d}. Pulando.")
                    continue
                print(f"   Cenário: {cen_id}")

            # Extrair comparação apenas para essa hora
            res = extrair_comparacao(cen_id, models, hora_alvo=hora)
            if res:
                resultados.update(res)
        except:
            continue

    if not resultados:
        print("Nenhum dado de comparação obtido. Abortando.")
        return 1

    # 3. Imprimir detalhes
    print("\n[3] Detalhes das predições:")
    print_detalhes_comparacao(resultados)

    # 4. Plotar gráficos
    print("\n[4] Gerando gráficos comparativos...")
    for hora in HORAS_INTERESSE:
        plot_comparacao_barras(resultados, hora, OUTPUT_DIR, SAVE_FIG)

    print("\n" + "=" * 70)
    print("PROCESSO CONCLUÍDO")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())