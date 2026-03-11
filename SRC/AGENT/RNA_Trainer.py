#!/usr/bin/env python3
"""
RNA_especialistas_por_horario_v4.py

Para cada hora (4,5,6) treina uma MLPRegressor multi‑saída que recebe como entrada
as features de TODAS as barras (concatenadas) e prevê os targets de TODAS as barras
simultaneamente.

Features (para cada barra):
    BESS_init_cenario, PGWIND_disponivel_cenario, PGER_CONV_total_result,
    PLOAD_medido (se barra com medição) ou PLOAD_estimado (caso contrário)

Targets (para cada barra):
    CURTAILMENT_total_result, BESS_operation_result

Cada amostra corresponde a uma combinação única de (cen_id, data_simulacao, hora_simulacao).
"""

import numpy as np
import pandas as pd
import os
import sqlite3
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from tqdm import tqdm

# ==================== CONFIGURAÇÕES ====================
DB_PATH = 'DATA/output/RNA_resultados_PL_acoplado.db'
MODELS_DIR = 'DATA/output/modelos_especialistas_v4'
TEST_SIZE = 0.2
RANDOM_STATE = 42
HIDDEN_LAYERS = (128, 64)          # arquitetura da MLP
MAX_ITER = 2000
TOLERANCE_REL = 0.05                # 5% de erro relativo
TOLERANCE_ABS = 0.1                 # tolerância absoluta para valores próximos de zero
MIN_SAMPLES_PER_GROUP = 30          # número mínimo de amostras por hora
BARRAS_COM_MEDICAO = [3]            # lista de BAR_id que possuem medição real
REMOVE_CONSTANT_COLUMNS = True      # se True, remove colunas com valores constantes (ex.: tudo zero)
# ========================================================

def load_data(db_path):
    """Carrega os dados do banco SQLite."""
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
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def create_wide_format(df, barras_com_medicao):
    """
    Transforma o DataFrame longo em largo:
    - Uma linha por (cen_id, data_simulacao, hora_simulacao)
    - Colunas: para cada barra, suas features e targets.
    """
    # Criar colunas de carga medida/estimada
    df = df.copy()
    df['PLOAD_medido'] = 0.0
    df['PLOAD_estimado'] = 0.0
    mask_medido = df['BAR_id'].isin(barras_com_medicao)
    df.loc[mask_medido, 'PLOAD_medido'] = df.loc[mask_medido, 'PLOAD_cenario']
    df.loc[~mask_medido, 'PLOAD_estimado'] = df.loc[~mask_medido, 'PLOAD_cenario']
    
    # Lista de colunas que serão pivotadas (features + targets)
    pivot_cols = [
        'BESS_init_cenario', 'PGWIND_disponivel_cenario', 'PGER_CONV_total_result',
        'PLOAD_medido', 'PLOAD_estimado',
        'CURTAILMENT_total_result', 'BESS_operation_result'
    ]
    
    # Índice que identifica cada instante único
    index_cols = ['cen_id', 'data_simulacao', 'hora_simulacao', 'dia_semana']
    
    # Fazer o pivot: cada BAR_id vira uma coluna para cada variável em pivot_cols
    df_pivot = df.pivot_table(index=index_cols, columns='BAR_id', values=pivot_cols)
    
    # Achatar o MultiIndex das colunas: junta o nome da variável com o BAR_id
    df_pivot.columns = [f'{var}_BAR{bar}' for var, bar in df_pivot.columns]
    
    # Resetar o índice para ter as colunas de identificação como colunas normais
    df_pivot = df_pivot.reset_index()
    
    return df_pivot

def prepare_X_y(df_wide, remove_constants=True):
    """
    Separa features (X) e targets (y) a partir do DataFrame largo.
    Remove linhas com NaN.
    Se remove_constants=True, remove colunas de features constantes.
    Em seguida, constrói y com base nas features restantes:
        - Se existe BESS_init_cenario_BAR{id} em X, adiciona BESS_operation_result_BAR{id} em y.
        - Se existe PGWIND_disponivel_cenario_BAR{id} em X, adiciona CURTAILMENT_total_result_BAR{id} em y.
    As demais features (PLOAD_medido, PLOAD_estimado, PGER_CONV_total_result) permanecem apenas em X.
    """
    feature_prefixes = ['BESS_init_cenario', 'PGWIND_disponivel_cenario', 
                        'PGER_CONV_total_result', 'PLOAD_medido', 'PLOAD_estimado']
    
    # Todas as colunas que são features (inclusive as que podem ser removidas depois)
    all_feature_cols = [col for col in df_wide.columns 
                        if any(col.startswith(prefix) for prefix in feature_prefixes)]
    
    # Remover linhas com NaN nas features (targets ainda não foram separados)
    df_clean = df_wide.dropna(subset=all_feature_cols)
    
    # Inicialmente X com todas as features
    X = df_clean[all_feature_cols]
    
    if remove_constants:
        # Remover colunas de features constantes
        constant_features = X.columns[X.std() == 0].tolist()
        if constant_features:
            print(f"   Removendo features constantes: {constant_features}")
            X = X.drop(columns=constant_features)
    
    # Agora, construir y com base nas features restantes
    target_cols = []
    for col in X.columns:
        # Extrair o número da barra e o tipo da feature
        if '_BAR' not in col:
            continue
        base, bar = col.rsplit('_BAR', 1)
        # base pode ser algo como 'BESS_init_cenario' ou 'PGWIND_disponivel_cenario', etc.
        if base.startswith('BESS_init_cenario'):
            target_name = f'BESS_operation_result_BAR{bar}'
            if target_name in df_clean.columns:
                target_cols.append(target_name)
        elif base.startswith('PGWIND_disponivel_cenario'):
            target_name = f'CURTAILMENT_total_result_BAR{bar}'
            if target_name in df_clean.columns:
                target_cols.append(target_name)
    
    # Remover duplicatas (caso haja)
    target_cols = list(dict.fromkeys(target_cols))
    
    # Construir y
    if target_cols:
        y = df_clean[target_cols]
    else:
        y = pd.DataFrame(index=df_clean.index)  # vazio, mas com mesmo índice
    
    print(f"   Features shape: {X.shape}, Targets shape: {y.shape}")
    return X, y

def calculate_correctness(y_true, y_pred, rel_tol, abs_tol):
    """
    Retorna array booleano indicando se a previsão está correta para TODOS os targets.
    Critério: para cada target, |y_true - y_pred| <= max(rel_tol * |y_true|, abs_tol)
    """
    errors = np.abs(y_true.values - y_pred)
    tolerance = np.maximum(rel_tol * np.abs(y_true.values), abs_tol)
    correct_all = (errors <= tolerance).all(axis=1)
    return correct_all

def train_and_evaluate_for_hour(X, y, hour, models_dir):
    """
    Treina e avalia um modelo para uma hora específica.
    Retorna um dicionário com métricas ou None se falhar.
    """
    if len(X) < MIN_SAMPLES_PER_GROUP:
        print(f"   Hora {hour:02d} tem apenas {len(X)} amostras (< {MIN_SAMPLES_PER_GROUP}) - ignorado.")
        return None

    # Divisão treino-teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Pipeline com escalonamento e MLP
    mlp = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation='relu',
        solver='adam',
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=100,
        verbose=False
    )

    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('mlp', mlp)
    ])

    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"   Erro no treinamento para hora {hour:02d}: {e}")
        return None

    # Previsões
    y_pred = pipeline.predict(X_test)

    # Corrigir dimensionalidade quando há apenas um target
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Métricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

    # Acurácia por tolerância
    correctness = calculate_correctness(y_test, y_pred, TOLERANCE_REL, TOLERANCE_ABS)
    n_correct = np.sum(correctness)
    n_total = len(correctness)
    accuracy = n_correct / n_total * 100 if n_total > 0 else 0.0

    # Salvar o pipeline
    hour_dir = os.path.join(models_dir, f"hora_{hour:02d}")
    os.makedirs(hour_dir, exist_ok=True)
    model_path = os.path.join(hour_dir, 'pipeline.joblib')
    joblib.dump(pipeline, model_path)

    # Salvar previsões do teste (opcional)
    df_out = y_test.copy()
    for i, col in enumerate(y_test.columns):
        df_out[f'{col}_previsto'] = y_pred[:, i]
    df_out['correto'] = correctness
    df_out.to_csv(os.path.join(hour_dir, 'previsoes_teste.csv'), index=False)

    return {
        'hora': hour,
        'n_amostras': len(X),
        'n_treino': len(X_train),
        'n_teste': len(X_test),
        'mse': mse,
        'r2': r2,
        'acuracia': accuracy,
        'n_correct': n_correct,
        'n_total_teste': n_total,
        'model_path': model_path
    }

def plot_accuracy_bar(summary_df, save_dir):
    """Gráfico de barras da acurácia por hora."""
    if summary_df.empty:
        return
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df['hora'], summary_df['acuracia'], color='steelblue')
    plt.xlabel('Hora do Dia')
    plt.ylabel('Acurácia (%)')
    plt.title(f'Acurácia dos modelos por hora (tolerância {TOLERANCE_REL*100:.0f}% rel.)')
    plt.xticks(summary_df['hora'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'acuracia_por_hora.png')
    plt.savefig(plot_path, dpi=150)
    plt.show()
    plt.close()
    print(f"Gráfico salvo em: {plot_path}")

def main():
    print("=" * 70)
    print("TREINAMENTO DE RNAs ESPECIALISTAS POR HORA (TODAS AS BARRAS CONCATENADAS)")
    print("=" * 70)

    # 1. Carregar dados
    print("\n[1] Carregando dados do banco...")
    df = load_data(DB_PATH)
    print(f"   Total de registros: {len(df)}")
    print(f"   Colunas: {list(df.columns)}")

    # 2. Transformar para formato largo
    print("\n[2] Transformando dados para formato largo (uma linha por instante)...")
    df_wide = create_wide_format(df, BARRAS_COM_MEDICAO)
    print(f"   Instantes únicos: {len(df_wide)}")
    print(f"   Colunas após pivot: {len(df_wide.columns)}")
    print(f"   Colunas: {list(df_wide.columns)}")

    # 3. Verificar distribuição por hora
    print("\n[3] Verificando distribuição por hora...")
    horas_disponiveis = sorted(df_wide['hora_simulacao'].unique())
    print(f"   Horas disponíveis: {horas_disponiveis}")
    for h in horas_disponiveis:
        count = len(df_wide[df_wide['hora_simulacao'] == h])
        print(f"      Hora {h:02d}: {count} amostras")

    # 4. Preparar diretório de saída
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 5. Treinar modelos para horas 4,5,6
    print("\n[4] Treinando modelos para horas 4,5,6...")
    resultados = []
    horas_interesse = [4, 5, 6]
    
    for hora in horas_interesse:
        print(f"\n--- Processando hora {hora:02d} ---")
        df_hora = df_wide[df_wide['hora_simulacao'] == hora]
        X, y = prepare_X_y(df_hora, remove_constants=REMOVE_CONSTANT_COLUMNS)
        
        if X.shape[1] == 0 or y.shape[1] == 0:
            print(f"   Sem features ou targets após remoção de constantes - ignorando hora {hora}.")
            continue
        
        metrica = train_and_evaluate_for_hour(X, y, hora, MODELS_DIR)
        if metrica is not None:
            resultados.append(metrica)

    # 6. Consolidar resultados
    print("\n[5] Consolidando resultados...")
    summary_df = pd.DataFrame(resultados)
    if summary_df.empty:
        print("Nenhum modelo foi treinado.")
        return

    summary_csv = os.path.join(MODELS_DIR, 'resumo_modelos.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Resumo salvo em: {summary_csv}")

    # 7. Estatísticas
    print("\n--- Estatísticas Globais ---")
    print(f"Total de modelos treinados: {len(summary_df)}")
    print(f"Acurácia média simples: {summary_df['acuracia'].mean():.2f}%")
    print(f"R² médio: {summary_df['r2'].mean():.4f}")

    # 8. Gráfico
    print("\n[6] Gerando gráfico...")
    plot_accuracy_bar(summary_df, MODELS_DIR)

    print("\n" + "=" * 70)
    print("PROCESSO CONCLUÍDO")
    print("=" * 70)

if __name__ == '__main__':
    main()