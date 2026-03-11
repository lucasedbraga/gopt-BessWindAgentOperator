#!/usr/bin/env python3
"""
compare_rna_optimizer.py

Gera um cenário (ou utiliza um existente) e compara, para as horas 4, 5 e 6,
os valores reais (otimizador) com as predições da RNA.
Produz gráficos de barras comparativos para BESS_operation e CURTAILMENT.
"""

import numpy as np
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

# ==================== CONFIGURAÇÕES ====================
# Caminhos
JSON_PATH = "../DATA/input/B6L8_BASE.json"
DB_PATH = "../DATA/output/RNA_resultados_PL_acoplado.db"
MODELS_DIR = "../DATA/output/modelos_especialistas_v4"

# Horas para as quais existem modelos treinados e que queremos comparar
HORAS_INTERESSE = [4, 5, 6]

# Barras com medição (para create_wide_format)
BARRAS_COM_MEDICAO = [3]

# Parâmetros da geração de cenários (usados apenas se não for fornecido um cenário existente)
N_ITERACOES = 1          # Vamos gerar apenas UM cenário
N_DIAS = 30
N_HORAS = 24
SOC_INICIAL_FRACAO = 0.5
SOC_FINAL_FRACAO = 0.5
CONSIDERAR_PERDAS = True
SOLVER_NAME = 'highs'
TOL = 1e-4
MAX_ITER = 5
WRITE_LP = False
SILENCIAR_SOLVER = True   # Suprime saída do solver durante a geração

# Controle de plotagem
SAVE_FIG = True
OUTPUT_DIR = "DATA/output/graficos_comparacao"

# Se você já tem um cenário no banco e quer usá-lo, defina o ID aqui.
# Caso contrário, deixe como None para gerar um novo.
CENARIO_EXISTENTE = None   # Exemplo: "20250310143000_00000"
# ========================================================

# Ajusta o path para encontrar os módulos do projeto
# Ajusta o path para encontrar os módulos do projeto
current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = current_path.replace("IHM", "SRC")
sys.path.append(src_path)

from UTILS.SystemLoader import SistemaLoader
from DB.DBhandler_OPF import OPF_DBHandler
from UTILS.EvaluateFactors import EvaluateFactors
from SOLVER.OPF_DC.DC_OPF_Acoplado import TimeCoupledOPFModel

# ========== Context manager para suprimir saída ==========
@contextmanager
def suppress_output():
    """Suprime toda a saída para stdout e stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# ========== Funções de preparação de dados (copiadas do script de treino) ==========

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
               BESS_init_cenario,
               PGWIND_disponivel_cenario,
               PGER_CONV_total_result,
               CURTAILMENT_total_result,
               BESS_operation_result
        FROM DBAR_results
    '''
    if cen_id is not None:
        query += " WHERE cen_id = ?"
        df = pd.read_sql_query(query, conn, params=(cen_id,))
    else:
        df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def create_wide_format(df, barras_com_medicao):
    """
    Transforma o DataFrame longo em largo:
    - Uma linha por (cen_id, data_simulacao, hora_simulacao)
    - Colunas: para cada barra, suas features e targets.
    """
    df = df.copy()
    df['PLOAD_medido'] = 0.0
    df['PLOAD_estimado'] = 0.0
    mask_medido = df['BAR_id'].isin(barras_com_medicao)
    df.loc[mask_medido, 'PLOAD_medido'] = df.loc[mask_medido, 'PLOAD_cenario']
    df.loc[~mask_medido, 'PLOAD_estimado'] = df.loc[~mask_medido, 'PLOAD_cenario']
    
    pivot_cols = [
        'BESS_init_cenario', 'PGWIND_disponivel_cenario', 'PGER_CONV_total_result',
        'PLOAD_medido', 'PLOAD_estimado',
        'CURTAILMENT_total_result', 'BESS_operation_result'
    ]
    index_cols = ['cen_id', 'data_simulacao', 'hora_simulacao', 'dia_semana']
    
    df_pivot = df.pivot_table(index=index_cols, columns='BAR_id', values=pivot_cols)
    df_pivot.columns = [f'{var}_BAR{bar}' for var, bar in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    return df_pivot

def prepare_X_y(df_wide, remove_constants=True):
    """
    Separa features (X) e targets (y) a partir do DataFrame largo.
    Retorna X e y como DataFrames.
    """
    feature_prefixes = ['BESS_init_cenario', 'PGWIND_disponivel_cenario', 
                        'PGER_CONV_total_result', 'PLOAD_medido', 'PLOAD_estimado']
    
    all_feature_cols = [col for col in df_wide.columns 
                        if any(col.startswith(prefix) for prefix in feature_prefixes)]
    
    df_clean = df_wide.dropna(subset=all_feature_cols)
    X = df_clean[all_feature_cols]
    
    if remove_constants:
        constant_features = X.columns[X.std() == 0].tolist()
        if constant_features:
            X = X.drop(columns=constant_features)
    
    # Construir targets com base nas features remanescentes
    target_cols = []
    for col in X.columns:
        if '_BAR' not in col:
            continue
        base, bar = col.rsplit('_BAR', 1)
        if base.startswith('BESS_init_cenario'):
            target_name = f'BESS_operation_result_BAR{bar}'
            if target_name in df_clean.columns:
                target_cols.append(target_name)
        elif base.startswith('PGWIND_disponivel_cenario'):
            target_name = f'CURTAILMENT_total_result_BAR{bar}'
            if target_name in df_clean.columns:
                target_cols.append(target_name)
    
    target_cols = list(dict.fromkeys(target_cols))
    
    if target_cols:
        y = df_clean[target_cols]
    else:
        y = pd.DataFrame(index=df_clean.index)
    
    return X, y

def get_target_names_from_features(feature_names):
    """
    Deriva os nomes dos targets a partir dos nomes das features, seguindo a mesma
    lógica usada no prepare_X_y.
    """
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
    """Carrega os pipelines das horas de interesse e retorna um dicionário."""
    models = {}
    for hora in HORAS_INTERESSE:
        model_path = os.path.join(MODELS_DIR, f"hora_{hora:02d}", "pipeline.joblib")
        if not os.path.exists(model_path):
            print(f"   Modelo para hora {hora} não encontrado em {model_path}. Ignorando.")
            continue
        pipeline = joblib.load(model_path)
        scaler = pipeline.named_steps['scaler']
        if hasattr(scaler, 'feature_names_in_'):
            feature_names = list(scaler.feature_names_in_)
        else:
            print(f"   Aviso: scaler não possui feature_names_in_ para hora {hora}. Modelo ignorado.")
            continue
        
        target_names = get_target_names_from_features(feature_names)
        if not target_names:
            print(f"   Aviso: não foi possível derivar targets para hora {hora}. Modelo ignorado.")
            continue
        
        models[hora] = {
            'pipeline': pipeline,
            'feature_names': feature_names,
            'target_names': target_names
        }
        print(f"   Hora {hora:02d} carregada: {len(feature_names)} features, {len(target_names)} targets")
    return models

# ========== Função para gerar um novo cenário ==========
def gerar_cenario_unico():
    """Gera um único cenário usando o otimizador e retorna o cen_id."""
    print("\n[GERAÇÃO] Gerando um novo cenário...")
    
    if not os.path.exists(JSON_PATH):
        print(f"ERRO: Arquivo do sistema não encontrado: {JSON_PATH}")
        return None
    
    sistema = SistemaLoader(JSON_PATH)
    db_handler = OPF_DBHandler(DB_PATH)
    db_handler.create_tables()
    
    modelo = TimeCoupledOPFModel(
        sistema=sistema,
        n_horas=N_HORAS,
        n_dias=N_DIAS,
        db_handler=db_handler,
        considerar_perdas=CONSIDERAR_PERDAS,
        dia_inicial=0
    )
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        cen_id = f"{timestamp}_00000"
        
        seed = secrets.randbits(32)
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
        
        if SILENCIAR_SOLVER:
            with suppress_output():
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
        else:
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

# ========== Função para extrair dados de comparação ==========
def extrair_comparacao(cen_id, models):
    """
    Para um dado cen_id e os modelos carregados, retorna um dicionário com:
        hora -> {
            'y_true': DataFrame com os valores reais (targets) - UMA ÚNICA LINHA,
            'y_pred': np.array com as predições - UMA ÚNICA LINHA,
            'target_names': lista dos nomes dos targets
        }
    """
    print(f"\n[COMPARAÇÃO] Carregando dados do cenário {cen_id}...")
    df = load_data(DB_PATH, cen_id=cen_id)
    if df.empty:
        print(f"   Nenhum dado encontrado para cenário {cen_id}")
        return {}
    
    df_wide = create_wide_format(df, BARRAS_COM_MEDICAO)
    resultados = {}
    
    for hora in HORAS_INTERESSE:
        if hora not in models:
            continue
        
        model_data = models[hora]
        pipeline = model_data['pipeline']
        feature_names = model_data['feature_names']
        target_names = model_data['target_names']
        
        # Filtra APENAS a hora específica
        df_hora = df_wide[df_wide['hora_simulacao'] == hora]
        if df_hora.empty:
            print(f"   Hora {hora:02d}: sem dados neste cenário.")
            continue
        
        # Seleciona UMA LINHA ALEATÓRIA
        if len(df_hora) > 1:
            idx_aleatorio = np.random.randint(0, len(df_hora))
            df_hora = df_hora.iloc[[idx_aleatorio]]
            print(f"   Hora {hora:02d}: selecionada linha aleatória {idx_aleatorio} de {len(df_hora)} disponíveis")
        else:
            print(f"   Hora {hora:02d}: apenas 1 linha disponível")
        
        X_raw, y_raw = prepare_X_y(df_hora, remove_constants=False)
        
        # Garantir que temos todas as features necessárias
        try:
            X = X_raw[feature_names]
        except KeyError as e:
            print(f"   Erro: Feature {e} não encontrada em X_raw para hora {hora}")
            continue
        
        # Garantir que temos todos os targets necessários
        try:
            y_true = y_raw[target_names]
        except KeyError as e:
            print(f"   Erro: Target {e} não encontrado em y_raw para hora {hora}")
            continue
        
        # Predição - reshape para 2D se necessário
        y_pred = pipeline.predict(X)
        
        # Garantir que y_pred seja 2D (1, n_targets)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)
        
        resultados[hora] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'target_names': target_names
        }
        
        print(f"   Hora {hora:02d}: 1 amostra processada.")
        print(f"   Shapes - y_true: {y_true.shape}, y_pred: {y_pred.shape}")
    
    return resultados

# ========== Funções de plotagem ==========
def plot_comparacao_barras(resultados, hora, output_dir, save_fig):
    """
    Plota gráficos de barras comparando real vs predito para BESS_operation e CURTAILMENT.
    """
    if hora not in resultados:
        return
    
    data = resultados[hora]
    y_true = data['y_true']
    y_pred = data['y_pred']
    target_names = data['target_names']
    
    # Separar targets por tipo
    bess_targets = [name for name in target_names if 'BESS_operation' in name]
    curt_targets = [name for name in target_names if 'CURTAILMENT' in name]
    
    # Extrair valores para BESS
    if len(bess_targets) > 0:
        # Pegar a primeira linha (única amostra)
        bess_true = y_true[bess_targets].iloc[0].values  # Shape: (n_barras,)
        bess_pred = y_pred[0, [target_names.index(t) for t in bess_targets]]  # Shape: (n_barras,)
        
        barras_bess = [t.replace('BESS_operation_result_BAR', '') for t in bess_targets]
        
        fig, ax = plt.subplots(figsize=(max(6, len(barras_bess) * 0.8), 5))
        x = np.arange(len(barras_bess))
        width = 0.35
        
        ax.bar(x - width/2, bess_true, width, label='Real', color='steelblue')
        ax.bar(x + width/2, bess_pred, width, label='Previsto', color='orange')
        
        ax.set_xlabel('Barra')
        ax.set_ylabel('Potência (MW)')
        ax.set_title(f'Hora {hora:02d} - Operação da Bateria (BESS)')
        ax.set_xticks(x)
        ax.set_xticklabels(barras_bess)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_fig:
            os.makedirs(output_dir, exist_ok=True)
            fname = os.path.join(output_dir, f'bess_comparison_hora_{hora:02d}.png')
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            print(f"   Gráfico salvo: {fname}")
        plt.show()
    
    # Extrair valores para CURTAILMENT
    if len(curt_targets) > 0:
        # Pegar a primeira linha (única amostra)
        curt_true = y_true[curt_targets].iloc[0].values  # Shape: (n_barras,)
        curt_pred = y_pred[0, [target_names.index(t) for t in curt_targets]]  # Shape: (n_barras,)
        
        barras_curt = [t.replace('CURTAILMENT_total_result_BAR', '') for t in curt_targets]
        
        fig, ax = plt.subplots(figsize=(max(6, len(barras_curt) * 0.8), 5))
        x = np.arange(len(barras_curt))
        width = 0.35
        
        ax.bar(x - width/2, curt_true, width, label='Real', color='steelblue')
        ax.bar(x + width/2, curt_pred, width, label='Previsto', color='orange')
        
        ax.set_xlabel('Barra')
        ax.set_ylabel('Potência (MW)')
        ax.set_title(f'Hora {hora:02d} - Corte Eólico (CURTAILMENT)')
        ax.set_xticks(x)
        ax.set_xticklabels(barras_curt)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_fig:
            os.makedirs(output_dir, exist_ok=True)
            fname = os.path.join(output_dir, f'curtailment_comparison_hora_{hora:02d}.png')
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            print(f"   Gráfico salvo: {fname}")
        plt.show()

# ========== Função principal ==========
def main():
    print("=" * 70)
    print("COMPARAÇÃO RNA vs OTIMIZADOR - HORAS 4, 5 e 6")
    print("=" * 70)

    # 1. Carregar modelos
    print("\n[1] Carregando modelos...")
    models = carregar_modelos()
    if not models:
        print("Nenhum modelo válido carregado. Abortando.")
        return 1

    # 2. Definir cenário a ser usado
    if CENARIO_EXISTENTE:
        cen_id = CENARIO_EXISTENTE
        print(f"\n[2] Usando cenário existente: {cen_id}")
    else:
        cen_id = gerar_cenario_unico()
        if cen_id is None:
            print("Falha na geração do cenário. Abortando.")
            return 1

    # 3. Extrair comparação
    resultados = extrair_comparacao(cen_id, models)
    if not resultados:
        print("Nenhum dado de comparação obtido. Abortando.")
        return 1

    # 4. Plotar gráficos
    print("\n[3] Gerando gráficos comparativos...")
    for hora in HORAS_INTERESSE:
        plot_comparacao_barras(resultados, hora, OUTPUT_DIR, SAVE_FIG)

    print("\n" + "=" * 70)
    print("PROCESSO CONCLUÍDO")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())