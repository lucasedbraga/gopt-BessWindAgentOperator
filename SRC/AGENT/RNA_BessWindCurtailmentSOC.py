#!/usr/bin/env python3
"""
RNA_especialistas_por_horario.py

Para cada combinação de dia_semana e hora_simulacao, treina uma MLPRegressor multi‑saída
especialista naquele horário. Features: BAR_id (one‑hot), BESS_init_cenario,
PGWIND_disponivel_cenario, PGER_CONV_total_result. Targets: CURTAILMENT_total_result,
BESS_operation_result. Avalia acurácia com tolerância relativa/absoluta.
"""

import numpy as np
import pandas as pd
import os
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from tqdm import tqdm

# ==================== CONFIGURAÇÕES ====================
DB_PATH = 'DATA/output/resultados_PL_acoplado_RNA.db'
MODELS_DIR = 'DATA/output/modelos_especialistas'
TEST_SIZE = 0.2
RANDOM_STATE = 42
HIDDEN_LAYERS = (64, 32)          # arquitetura da MLP
MAX_ITER = 2000
TOLERANCE_REL = 0.05               # 5% de erro relativo
TOLERANCE_ABS = 0.1                 # tolerância absoluta para valores próximos de zero
MIN_SAMPLES_PER_GROUP = 50          # número mínimo de amostras para treinar um modelo
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
               BESS_init_cenario,
               PGWIND_disponivel_cenario,
               PGER_CONV_total_result,
               CURTAILMENT_total_result,
               BESS_operation_result
        FROM DBAR_results
        -- LIMIT 500000  -- removido para usar todos os dados
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def prepare_features_targets(df_group):
    """
    Para um grupo (DataFrame), retorna X (features) e y (targets).
    X contém as colunas: BAR_id (categórica) + numéricas.
    """
    feature_cols = ['BAR_id', 'BESS_init_cenario', 'PGWIND_disponivel_cenario', 'PGER_CONV_total_result']
    target_cols = ['CURTAILMENT_total_result', 'BESS_operation_result']

    # Remove linhas com NaN
    df_clean = df_group[feature_cols + target_cols].dropna()
    if len(df_clean) == 0:
        return None, None

    X = df_clean[feature_cols].copy()
    y = df_clean[target_cols].copy()
    return X, y

def calculate_correctness(y_true, y_pred, rel_tol, abs_tol):
    """
    Retorna array booleano indicando se a previsão está correta para AMBOS os targets.
    Critério: para cada target, |y_true - y_pred| <= max(rel_tol * |y_true|, abs_tol)
    """
    errors = np.abs(y_true.values - y_pred)
    tolerance = np.maximum(rel_tol * np.abs(y_true.values), abs_tol)
    correct_both = (errors <= tolerance).all(axis=1)
    return correct_both

def train_and_evaluate_for_group(X, y, group_name, models_dir):
    """
    Treina e avalia um modelo para um grupo (dia_semana, hora).
    Retorna um dicionário com métricas ou None se falhar.
    """
    if len(X) < MIN_SAMPLES_PER_GROUP:
        print(f"   Grupo {group_name} tem apenas {len(X)} amostras (< {MIN_SAMPLES_PER_GROUP}) - ignorado.")
        return None

    # Divisão treino-teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Pré-processador: one-hot para BAR_id, demais numéricas passam direto
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['BAR_id'])
        ],
        remainder='passthrough'
    )

    # MLP multi-saída
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
        ('preprocessor', preprocessor),
        ('regressor', mlp)
    ])

    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"   Erro no treinamento do grupo {group_name}: {e}")
        return None

    # Previsões
    y_pred = pipeline.predict(X_test)
    df_output = pd.DataFrame(y_pred) 
    df_output.columns = y_test.columns
    df_input = y_test.reset_index()
    df_edimar = df_input - df_output
    # Métricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

    # Acurácia por tolerância
    correctness = calculate_correctness(y_test, y_pred, TOLERANCE_REL, TOLERANCE_ABS)
    n_correct = np.sum(correctness)
    n_total = len(correctness)
    accuracy = n_correct / n_total * 100 if n_total > 0 else 0.0

    # Salvar o pipeline
    group_dir = os.path.join(models_dir, f"dia_{group_name[0]}", f"hora_{group_name[1]:02d}")
    os.makedirs(group_dir, exist_ok=True)
    model_path = os.path.join(group_dir, 'pipeline.joblib')
    joblib.dump(pipeline, model_path)

    # Opcional: salvar previsões do teste
    df_out = y_test.copy()
    df_out['CURTAILMENT_previsto'] = y_pred[:, 0]
    df_out['BESS_operation_previsto'] = y_pred[:, 1]
    df_out['correto'] = correctness
    #df_out.to_csv(os.path.join(group_dir, 'previsoes_teste.csv'), index=False)

    return {
        'dia': group_name[0],
        'hora': group_name[1],
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

def plot_accuracy_heatmap(summary_df, save_dir):
    """
    Gera um heatmap da acurácia por dia da semana (0-6) e hora (0-23).
    """
    if summary_df.empty:
        print("Sem dados para gerar heatmap.")
        return

    pivot = summary_df.pivot(index='hora', columns='dia', values='acuracia')
    # Preenche NaN com 0 (grupos sem modelo)
    pivot = pivot.fillna(0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': 'Acurácia (%)'})
    plt.title(f'Acurácia dos modelos especialistas por dia/hora (tolerância {TOLERANCE_REL*100:.0f}% rel.)')
    plt.xlabel('Dia da Semana (0=Segunda)')
    plt.ylabel('Hora do Dia')
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'heatmap_acuracia.png')
    plt.savefig(plot_path, dpi=150)
    plt.show()
    plt.close()
    print(f"Heatmap salvo em: {plot_path}")

def plot_global_accuracy_bar(summary_df, save_dir):
    """
    Gera um gráfico de barras com a acurácia média por hora (considerando todos os dias).
    """
    if summary_df.empty:
        return

    # Média por hora
    hora_means = summary_df.groupby('hora')['acuracia'].mean().reset_index()
    plt.figure(figsize=(12, 5))
    plt.bar(hora_means['hora'], hora_means['acuracia'], color='steelblue')
    plt.xlabel('Hora do Dia')
    plt.ylabel('Acurácia Média (%)')
    plt.title('Acurácia Média por Hora (todos os dias)')
    plt.xticks(range(24))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'acuracia_media_por_hora.png')
    plt.savefig(plot_path, dpi=150)
    plt.show()
    plt.close()
    print(f"Gráfico de acurácia por hora salvo em: {plot_path}")

def main():
    print("=" * 70)
    print("TREINAMENTO DE RNAs ESPECIALISTAS POR DIA DA SEMANA E HORA")
    print("=" * 70)

    # 1. Carregar dados
    print("\n[1] Carregando dados do banco...")
    df = load_data(DB_PATH)
    print(f"   Total de registros: {len(df)}")
    print(f"   Colunas: {list(df.columns)}")

    # Verificar se as colunas necessárias existem
    required_cols = ['dia_semana', 'hora_simulacao', 'BAR_id', 
                     'BESS_init_cenario', 'PGWIND_disponivel_cenario', 
                     'PGER_CONV_total_result', 'CURTAILMENT_total_result', 
                     'BESS_operation_result']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERRO: Colunas faltando no banco: {missing}")
        return

    # 2. Agrupar por dia_semana e hora_simulacao
    print("\n[2] Agrupando dados por (dia_semana, hora_simulacao)...")
    grupos = df.groupby(['dia_semana', 'hora_simulacao'])
    n_grupos = len(grupos)
    print(f"   Total de grupos: {n_grupos}")

    # 3. Preparar diretório de saída
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 4. Loop sobre os grupos
    print("\n[3] Treinando modelos para cada grupo...")
    resultados = []
    for (dia, hora), grupo_df in tqdm(grupos, desc="Processando grupos"):
        group_name = (dia, hora)
        X, y = prepare_features_targets(grupo_df)
        if X is None or len(X) == 0:
            continue

        metrica = train_and_evaluate_for_group(X, y, group_name, MODELS_DIR)
        if metrica is not None:
            resultados.append(metrica)

    # 5. Consolidar resultados
    print("\n[4] Consolidando resultados...")
    summary_df = pd.DataFrame(resultados)
    if summary_df.empty:
        print("Nenhum modelo foi treinado (verifique os dados e o limite mínimo de amostras).")
        return

    # summary_csv = os.path.join(MODELS_DIR, 'resumo_modelos.csv')
    # summary_df.to_csv(summary_csv, index=False)
    # print(f"Resumo salvo em: {summary_csv}")

    # 6. Estatísticas globais
    print("\n--- Estatísticas Globais ---")
    print(f"Total de modelos treinados: {len(summary_df)}")
    print(f"Acurácia média (ponderada pelo nº de testes): "
          f"{(summary_df['n_correct'].sum() / summary_df['n_total_teste'].sum() * 100):.2f}%")
    print(f"Acurácia média simples: {summary_df['acuracia'].mean():.2f}%")
    print(f"R² médio: {summary_df['r2'].mean():.4f}")

    # 7. Gerar gráficos
    print("\n[5] Gerando gráficos de avaliação...")
    plot_accuracy_heatmap(summary_df, MODELS_DIR)
    plot_global_accuracy_bar(summary_df, MODELS_DIR)

    print("\n" + "=" * 70)
    print("PROCESSO CONCLUÍDO")
    print("=" * 70)

if __name__ == '__main__':
    main()