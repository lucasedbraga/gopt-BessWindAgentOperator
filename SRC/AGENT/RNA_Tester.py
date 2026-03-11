#!/usr/bin/env python3
"""
RNA_Tester.py

Combina a geração de cenários (DATAGenerator_TimeCoupled.py) com a avaliação
dos modelos treinados (RNA_especialistas_por_horario_v4.py). 
Primeiro gera N cenários (rodando o otimizador), depois testa a RNA em todos
os cenários gerados e apresenta um resultado geral.

Uso: python RNA_Tester.py
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

# ==================== CONFIGURAÇÕES ====================
# Caminhos
JSON_PATH = "DATA/input/B6L8_BASE.json"
DB_PATH = "DATA/output/RNA_resultados_PL_acoplado.db"
MODELS_DIR = "DATA/output/modelos_especialistas_v4"

# Horas para as quais existem modelos treinados
HORAS_INTERESSE = [4, 5, 6]

# Tolerâncias para considerar acerto (mesmas do treinamento)
TOLERANCE_REL = 0.05
TOLERANCE_ABS = 0.1

# Parâmetros da geração de cenários
N_ITERACOES = 3
N_DIAS = 30
N_HORAS = 24
SOC_INICIAL_FRACAO = 0.5
SOC_FINAL_FRACAO = 0.5
CONSIDERAR_PERDAS = True
SOLVER_NAME = 'highs'
TOL = 1e-4
MAX_ITER = 5
WRITE_LP = False

# Barras com medição (para create_wide_format)
BARRAS_COM_MEDICAO = [3]

# Controle de saída do solver (silenciar ou não)
SILENCIAR_SOLVER = True   # Se True, suprime toda a saída do solver durante a geração

# ========================================================

# Ajusta o path para encontrar os módulos do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UTILS.SystemLoader import SistemaLoader
from DB.DBhandler_OPF import OPF_DBHandler
from UTILS.EvaluateFactors import EvaluateFactors
from SOLVER.OPF_DC_TimeCoupled.DC_OPF_BESS_Acoplado import TimeCoupledOPFModel

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

def calculate_correctness(y_true, y_pred, rel_tol, abs_tol):
    """
    Retorna array booleano indicando se a previsão está correta para TODOS os targets.
    """
    errors = np.abs(y_true.values - y_pred)
    tolerance = np.maximum(rel_tol * np.abs(y_true.values), abs_tol)
    correct_all = (errors <= tolerance).all(axis=1)
    return correct_all

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

# ========== Função de teste de um cenário ==========

def testar_cenario(cen_id, models, contadores):
    """
    Para um cenário já salvo no banco, testa todas as horas de interesse usando os modelos.
    Atualiza os contadores in-place.
    """
    df = load_data(DB_PATH, cen_id=cen_id)
    if df.empty:
        print(f"   Aviso: Nenhum dado encontrado para cenário {cen_id}")
        return

    df_wide = create_wide_format(df, BARRAS_COM_MEDICAO)
    
    for hora in HORAS_INTERESSE:
        model_data = models.get(hora)
        if model_data is None:
            continue
        pipeline = model_data['pipeline']
        feature_names = model_data['feature_names']
        target_names = model_data['target_names']
        
        df_hora = df_wide[df_wide['hora_simulacao'] == hora]
        if df_hora.empty:
            continue
        
        X_raw, y_raw = prepare_X_y(df_hora, remove_constants=False)
        
        try:
            X = X_raw[feature_names]
        except KeyError as e:
            print(f"   Erro: Feature {e} não encontrada em X_raw para hora {hora}")
            continue
        
        try:
            y_true = y_raw[target_names]
        except KeyError as e:
            print(f"   Erro: Target {e} não encontrado em y_raw para hora {hora}")
            continue
        
        y_pred = pipeline.predict(X)
        correct = calculate_correctness(y_true, y_pred, TOLERANCE_REL, TOLERANCE_ABS)
        n_correct = np.sum(correct)
        n_total = len(correct)
        
        contadores[hora]['total'] += n_total
        contadores[hora]['acertos'] += n_correct
        
        print(f"      Hora {hora:02d}: {n_correct}/{n_total} corretos ({n_correct/n_total*100:.1f}%)")

# ========== Função principal ==========

def main():
    print("=" * 70)
    print("TESTE DE RNAs ESPECIALISTAS COM GERAÇÃO PRÉVIA DE CENÁRIOS")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Carregar modelos treinados
    # -------------------------------------------------------------------------
    print("\n[1] Carregando modelos...")
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

    if not models:
        print("Nenhum modelo válido carregado. Abortando.")
        return 1

    # -------------------------------------------------------------------------
    # 2. Preparar sistema e modelo de otimização (para geração)
    # -------------------------------------------------------------------------
    print("\n[2] Carregando sistema e criando modelo de otimização...")
    if not os.path.exists(JSON_PATH):
        print(f"ERRO: Arquivo do sistema não encontrado: {JSON_PATH}")
        return 1
    sistema = SistemaLoader(JSON_PATH)
    print(f"   Sistema: {JSON_PATH}")
    
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
    print("   Modelo criado.")

    # -------------------------------------------------------------------------
    # 3. Gerar todos os cenários primeiro
    # -------------------------------------------------------------------------
    print(f"\n[3] Gerando {N_ITERACOES} cenários...")
    cenarios_gerados = []
    inicio_geracao = time.time()

    for i in range(N_ITERACOES):
        try:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            cen_id = f"{timestamp}_{i:05d}"

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

            print(f"\n   [{i+1:2d}/{N_ITERACOES}] Gerando cenário {cen_id}...")
            
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

            cenarios_gerados.append(cen_id)
            print(f"   Cenário {cen_id} gerado com sucesso.")

        except Exception as e:
            print(f"   [!] Erro na iteração {i}: {e}")
            traceback.print_exc()
            continue

    tempo_geracao = time.time() - inicio_geracao
    print(f"\nGeração concluída em {tempo_geracao:.2f} segundos. Total de cenários gerados: {len(cenarios_gerados)}")

    if not cenarios_gerados:
        print("Nenhum cenário foi gerado. Abortando.")
        return 1

    # -------------------------------------------------------------------------
    # 4. Inicializar contadores e testar todos os cenários gerados
    # -------------------------------------------------------------------------
    print("\n[4] Iniciando teste dos cenários com as RNAs...")
    contadores = {hora: {'total': 0, 'acertos': 0} for hora in models.keys()}
    inicio_teste = time.time()

    for idx, cen_id in enumerate(cenarios_gerados, 1):
        print(f"\n   Testando cenário {idx}/{len(cenarios_gerados)}: {cen_id}")
        testar_cenario(cen_id, models, contadores)

    tempo_teste = time.time() - inicio_teste

    # -------------------------------------------------------------------------
    # 5. Resultados finais
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTADOS FINAIS")
    print("=" * 70)
    print(f"Total de cenários gerados e testados: {len(cenarios_gerados)}")
    print(f"Tempo de geração: {tempo_geracao:.2f} s")
    print(f"Tempo de teste: {tempo_teste:.2f} s")
    print(f"Tempo total: {tempo_geracao + tempo_teste:.2f} s")
    print("\n--- Acurácia por hora ---")
    for hora in sorted(contadores.keys()):
        tot = contadores[hora]['total']
        acertos = contadores[hora]['acertos']
        if tot > 0:
            acc = acertos / tot * 100
            print(f"Hora {hora:02d}: {acertos:4d}/{tot:4d} corretos ({acc:6.2f}%)")
        else:
            print(f"Hora {hora:02d}: sem testes")
    print("=" * 70)

    # Opcional: salvar resultados em arquivo
    resumo = {
        'data': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'cenarios': len(cenarios_gerados),
        'tempo_geracao': tempo_geracao,
        'tempo_teste': tempo_teste,
        'acertos_por_hora': {h: contadores[h]['acertos'] for h in contadores},
        'total_por_hora': {h: contadores[h]['total'] for h in contadores}
    }
    # Salvar em JSON, se desejado
    # with open('resultado_teste.json', 'w') as f:
    #     json.dump(resumo, f, indent=2)

    return 0

if __name__ == "__main__":
    sys.exit(main())