# RNA_unica_multitarget_simples.py
"""
Treina uma única MLPRegressor (multi‑saída) para todas as barras.
Features:
  - data_simulacao, hora_simulacao,
  - BESS_init_cenario, PGWIND_disponivel_cenario, PGER_CONV_total_result,
  - BAR_id (codificada one‑hot)
Targets:
  - CURTAILMENT_total_result, BESS_operation_result

Após o treino, avalia a acurácia com base em uma tolerância (relativa/absoluta)
e gera um gráfico de barras simples com a contagem de acertos/erros.
"""

import numpy as np
import pandas as pd
import os
import sqlite3
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# ==================== CONFIGURAÇÕES ====================
DB_PATH = 'DATA/output/resultados_PL_RNA.db'
MODELS_DIR = 'DATA/output/modelo_unico'
TEST_SIZE = 0.2
RANDOM_STATE = 42
HIDDEN_LAYERS = (64, 32)          # arquitetura da MLP
MAX_ITER = 2000
TOLERANCE_REL = 0.05               # 5% de erro relativo
TOLERANCE_ABS = 0.1                 # tolerância absoluta para valores próximos de zero
# ========================================================

def load_data(db_path):
    """Carrega os dados do banco SQLite."""
    conn = sqlite3.connect(db_path)
    query = '''
        SELECT cen_id,
               data_simulacao,
               hora_simulacao,
               BAR_id,
               BESS_init_cenario,
               PGWIND_disponivel_cenario,
               PGER_CONV_total_result,
               CURTAILMENT_total_result,
               BESS_operation_result
        FROM DBAR_results
        LIMIT 500000
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def prepare_features_targets(df):
    """
    Prepara as features (incluindo codificação da BAR_id) e os targets (2 colunas).
    Retorna X (DataFrame) e y (DataFrame com duas colunas).
    """
    feature_cols = [
        'data_simulacao',
        'hora_simulacao',
        'BESS_init_cenario',
        'PGWIND_disponivel_cenario',
        'PGER_CONV_total_result',
        'BAR_id'                     # será codificada
    ]
    target_cols = ['CURTAILMENT_total_result', 'BESS_operation_result']

    # Remove linhas com valores ausentes nas features ou targets
    df_clean = df[feature_cols + target_cols].dropna()

    X = df_clean[feature_cols].copy()
    y = df_clean[target_cols].copy()

    return X, y

def calculate_correctness(y_true, y_pred, rel_tol, abs_tol):
    """
    Retorna um array booleano indicando se a previsão está correta para AMBOS os targets.
    Critério: para cada target, |y_true - y_pred| <= max(rel_tol * |y_true|, abs_tol)
    """
    errors = np.abs(y_true.values - y_pred)
    tolerance = np.maximum(rel_tol * np.abs(y_true.values), abs_tol)
    correct_both = (errors <= tolerance).all(axis=1)
    return correct_both

def plot_accuracy_summary(correctness, save_dir):
    """
    Gera um gráfico de barras simples mostrando a quantidade de cenários corretos e incorretos,
    e exibe a porcentagem de acerto.
    """
    os.makedirs(save_dir, exist_ok=True)

    n_correct = np.sum(correctness)
    n_incorrect = len(correctness) - n_correct
    accuracy = n_correct / len(correctness) * 100

    plt.figure(figsize=(6, 5))
    bars = plt.bar(['Corretos', 'Incorretos'], [n_correct, n_incorrect],
                   color=['green', 'red'], alpha=0.7)
    plt.ylabel('Número de cenários')
    plt.title(f'Acurácia: {accuracy:.2f}% (tolerância {TOLERANCE_REL*100:.1f}% rel. ou {TOLERANCE_ABS} abs)')

    # Adiciona os valores sobre as barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'acuracia_barras.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Gráfico de acurácia salvo em: {plot_path}")

def save_predictions_report(y_test, y_pred, correctness, save_dir):
    """Salva um CSV com os valores reais, previstos e a flag de acerto."""
    df_out = y_test.copy()
    df_out['CURTAILMENT_previsto'] = y_pred[:, 0]
    df_out['BESS_operation_previsto'] = y_pred[:, 1]
    df_out['correto'] = correctness
    csv_path = os.path.join(save_dir, 'previsoes_teste.csv')
    df_out.to_csv(csv_path, index=False)
    print(f"Relatório de previsões salvo em: {csv_path}")

def main():
    print("Carregando dados...")
    df = load_data(DB_PATH)
    print(f"Dados carregados: {df.shape[0]} registros, {df.shape[1]} colunas")

    print("Preparando features e targets...")
    X, y = prepare_features_targets(df)
    print(f"Total de amostras após remoção de NaN: {len(X)}")

    # Define as colunas categóricas e numéricas
    categorical_features = ['BAR_id']
    # As demais colunas são numéricas e já estão normalizadas (0-1), portanto serão passadas sem alteração
    numeric_features = [col for col in X.columns if col not in categorical_features]

    # Pré-processador: apenas codifica a BAR_id (OneHotEncoder) e mantém as numéricas inalteradas
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'          # as colunas numéricas passam direto
    )

    # Cria o pipeline completo (pré-processamento + MLP)
    mlp = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation='relu',
        solver='adam',
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', mlp)
    ])

    # Divisão treino-teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("Treinando a rede neural... (pode levar alguns minutos)")
    pipeline.fit(X_train, y_train)

    # Previsões
    y_pred = pipeline.predict(X_test)

    # Avaliação multi-saída
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
    print(f"\nDesempenho no teste:")
    print(f"MSE (médio) = {mse:.6f}")
    print(f"R² (médio)  = {r2:.4f}")

    # Cálculo da acurácia baseada em tolerância
    correctness = calculate_correctness(y_test, y_pred, TOLERANCE_REL, TOLERANCE_ABS)
    n_correct = np.sum(correctness)
    n_total = len(correctness)
    accuracy = n_correct / n_total * 100
    print(f"\nAcurácia (ambos os targets dentro da tolerância): {accuracy:.2f}%")
    print(f"Cenários corretos: {n_correct} de {n_total}")

    # Gráfico simples de acurácia
    plot_accuracy_summary(correctness, os.path.join(MODELS_DIR, 'graficos'))

    # Salvar relatório de previsões
    save_predictions_report(y_test, y_pred, correctness, MODELS_DIR)

    # Salvando o pipeline completo (pré-processador + modelo)
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'pipeline_modelo_unico.joblib')
    joblib.dump(pipeline, model_path)
    print(f"\nPipeline completo (pré-processador + MLP) salvo em: {model_path}")

    # Exemplo de como carregar e usar depois:
    # pipeline_carregado = joblib.load('pipeline_modelo_unico.joblib')
    # previsoes = pipeline_carregado.predict(X_novo)

if __name__ == '__main__':
    main()