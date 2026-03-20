#!/usr/bin/env python3
"""
compare_rna_optimizer.py

Gera um cenário (ou utiliza um existente) e compara, para as horas 4, 5 e 6,
os valores reais (otimizador) com as predições da RNA.
Produz gráficos de barras comparativos para BESS_operation e CURTAILMENT.
Agora também imprime uma tabela detalhada com diferenças.
Se nenhum cenário for especificado, utiliza o último cenário presente no banco.
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

# ==================== CONFIGURAÇÕES ====================
# Caminhos
JSON_PATH = "DATA/input/ieee14_BASE.json"
DB_PATH = "DATA/output/RNA_resultados_PL_acoplado.db"
MODELS_DIR = "DATA/output/modelos_especialistas_v4"

# Horas para as quais existem modelos treinados e que queremos comparar
HORAS_INTERESSE = [16, 17, 18]

# Barras com medição (para create_wide_format)
#BARRAS_COM_MEDICAO = [3]
#BARRAS_COM_MEDICAO = [3, 5, 8]  # IEEE 14
BARRAS_COM_MEDICAO = [3, 5, 8, 17]  # IEEE 33

# Parâmetros da geração de cenários (usados apenas se não houver nenhum cenário no banco)
N_ITERACOES = 1          # Vamos gerar apenas UM cenário
N_DIAS = 7
N_HORAS = 24
SOC_INICIAL_FRACAO = 0.5
SOC_FINAL_FRACAO = 0.5
CONSIDERAR_PERDAS = True
SOLVER_NAME = 'highs'
TOL = 1e-4
MAX_ITER = 5
WRITE_LP = False

# Controle de plotagem
SAVE_FIG = True
OUTPUT_DIR = "../DATA/output/graficos_comparacao"

# Se você já tem um cenário no banco e quer usá-lo, defina o ID aqui.
# Caso contrário, deixe como None para buscar o último automaticamente.
CENARIO_EXISTENTE = None   # Exemplo: "20250310143000_00000"
# ========================================================

# Ajusta o path para encontrar os módulos do projeto
current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = current_path.replace("IHM", "SRC")
sys.path.append(src_path)

from UTILS.SystemLoader import SistemaLoader
from DB.DBhandler_OPF import OPF_DBHandler
from UTILS.EvaluateFactors import EvaluateFactors
from SOLVER.OPF_DC.DC_OPF_Acoplado import TimeCoupledOPFModel


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

# ========== Função para carregar modelos (agora usando metadata.json) ==========
def carregar_modelos():
    """
    Carrega os pipelines das horas de interesse a partir dos arquivos salvos,
    utilizando o metadata.json para obter os nomes exatos das features e targets.
    Retorna um dicionário com modelos.
    """
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

# ========== Função para obter o último cenário do banco ==========
def get_ultimo_cenario(db_path):
    """
    Consulta o banco e retorna o cen_id mais recente (considerando a ordenação
    decrescente do campo cen_id, que segue o padrão YYYYMMDDHHMMSS_xxxxx).
    Se não houver nenhum cenário, retorna None.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT cen_id FROM DBAR_results ORDER BY cen_id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]
    else:
        return None

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

# ========== Função para extrair dados de comparação (modificada) ==========
def extrair_comparacao(cen_id, models):
    """
    Para um dado cen_id e os modelos carregados, retorna um dicionário com:
        hora -> {
            'y_true': np.array com os valores reais (targets) - UMA ÚNICA LINHA,
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
        
        # --- Extrair features (X) ---
        X_values = []
        features_faltando = []
        for feat in feature_names:
            if feat in df_hora.columns:
                X_values.append(df_hora[feat].iloc[0])
            else:
                # Se a feature não existir, usa 0 (ou outro valor padrão)
                X_values.append(0.0)
                features_faltando.append(feat)
        if features_faltando:
            print(f"   Aviso: features ausentes preenchidas com 0: {features_faltando}")
        
        X = np.array(X_values).reshape(1, -1)  # Formato (1, n_features)
        
        # --- Extrair targets reais (y_true) ---
        y_true_values = []
        targets_faltando = []
        for tgt in target_names:
            if tgt in df_hora.columns:
                y_true_values.append(df_hora[tgt].iloc[0])
            else:
                targets_faltando.append(tgt)
        if targets_faltando:
            print(f"   Erro: targets ausentes no DataFrame: {targets_faltando}")
            print(f"   Não é possível comparar para hora {hora}. Pulando.")
            continue
        
        y_true = np.array(y_true_values).reshape(1, -1)  # Formato (1, n_targets)
        
        # --- Predição ---
        y_pred = pipeline.predict(X)  # Já deve retornar (1, n_targets)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)
        
        resultados[hora] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'target_names': target_names
        }
        
        print(f"   Hora {hora:02d}: 1 amostra processada.")
        print(f"   Shapes - X: {X.shape}, y_true: {y_true.shape}, y_pred: {y_pred.shape}")
    
    return resultados

# ========== Função para imprimir detalhes das diferenças (adaptada) ==========
def print_detalhes_comparacao(resultados):
    """
    Exibe uma tabela para cada hora com os valores reais, previstos e diferenças.
    """
    for hora in sorted(resultados.keys()):
        data = resultados[hora]
        y_true = data['y_true'][0]        # array 1D
        y_pred = data['y_pred'][0]         # array 1D
        target_names = data['target_names']
        
        print(f"\n{'='*60}")
        print(f"DETALHES PARA HORA {hora:02d}")
        print(f"{'='*60}")
        
        # Criar DataFrame para exibição
        erro_abs = np.abs(y_true - y_pred)

        df_detalhes = pd.DataFrame({
            'Target': target_names,
            'Real': y_true,
            'Previsto': y_pred,
            'Diferença (abs)': erro_abs,
            'Diferença (%)': np.where(
                np.abs(y_true) > 1e-6,
                erro_abs / np.abs(y_true) * 100,
                np.nan
            )
        })
        # Formatação
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.float_format', '{:.6f}'.format)
        print(df_detalhes.to_string(index=False))
        print()

# ========== Funções de plotagem (ajustadas para aceitar múltiplos tipos) ==========
def plot_comparacao_barras(resultados, hora, output_dir, save_fig):
    """
    Plota gráficos de barras comparando real vs predito.
    Agora separa os targets por prefixo (BESS_operation, CURTAILMENT, PLOAD_estimado)
    e gera um gráfico para cada grupo.
    """
    if hora not in resultados:
        return
    
    data = resultados[hora]
    y_true = data['y_true'][0]        # array 1D
    y_pred = data['y_pred'][0]         # array 1D
    target_names = data['target_names']
    
    # Agrupar targets por prefixo
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
    
    # Gerar um gráfico para cada grupo
    for grupo, indices in grupos.items():
        if not indices:
            continue
        
        # Extrair valores para este grupo
        true_vals = y_true[indices]
        pred_vals = y_pred[indices]
        nomes = [target_names[i] for i in indices]
        
        # Extrair número da barra (para rótulos)
        if grupo == 'PLOAD_estimado':
            barras = [n.replace('PLOAD_estimado_BAR', '') for n in nomes]
        elif grupo == 'BESS_operation':
            barras = [n.replace('BESS_operation_result_BAR', '') for n in nomes]
        elif grupo == 'CURTAILMENT':
            barras = [n.replace('CURTAILMENT_total_result_BAR', '') for n in nomes]
        else:
            barras = nomes  # fallback
        
        # Criar gráfico
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
            fname = os.path.join(output_dir, f'{grupo.lower()}_comparison_hora_{hora:02d}.png')
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
        print(f"\n[2] Usando cenário existente (fornecido): {cen_id}")
    else:
        print("\n[2] Buscando último cenário no banco de dados...")
        cen_id = get_ultimo_cenario(DB_PATH)
        if cen_id is None:
            print("Nenhum cenário encontrado no banco. Gerando um novo...")
            cen_id = gerar_cenario_unico()
            if cen_id is None:
                print("Falha na geração do cenário. Abortando.")
                return 1
        else:
            print(f"   Último cenário encontrado: {cen_id}")

    # 3. Extrair comparação
    resultados = extrair_comparacao(cen_id, models)
    if not resultados:
        print("Nenhum dado de comparação obtido. Abortando.")
        return 1

    # 4. Imprimir detalhes das diferenças
    print("\n[3] Detalhes das predições:")
    print_detalhes_comparacao(resultados)

    # 5. Plotar gráficos
    print("\n[4] Gerando gráficos comparativos...")
    for hora in HORAS_INTERESSE:
        plot_comparacao_barras(resultados, hora, OUTPUT_DIR, SAVE_FIG)

    print("\n" + "=" * 70)
    print("PROCESSO CONCLUÍDO")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())