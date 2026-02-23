# RNA_BessWindCurtailmentSOC_per_bar.py
"""
Treina uma rede neural MLPRegressor para cada barra (BAR_id) distinta.
Para cada barra, a rede aprende a relação entre:
  - Features: data_simulacao (dia), hora_simulacao,
              PGWIND_total_result, PCURTAILMENT_total_result,
              PLOAD_cenario, BESS_init_cenario, BESS_soc_atual_result
  - Target:   BESS_operation_result
Os modelos treinados (e opcionalmente os scalers) são salvos em disco.
Gera gráficos de dispersão comparando valores reais e previstos.
"""

import numpy as np
import pandas as pd
import os
import sqlite3
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Configurações
DB_PATH = 'DATA/output/resultados_PL_RNA.db'
MODELS_DIR = 'DATA/output/modelos_por_barra'
TEST_SIZE = 0.2
RANDOM_STATE = 42
HIDDEN_LAYERS = (32, 16)          # arquitetura da MLP
MAX_ITER = 1000
N_PLOT_SAMPLES = 50                # número de amostras para o gráfico comparativo
USE_SCALER = True                  # se True, aplica StandardScaler e salva o scaler

def load_data(db_path):
    """Carrega os dados do banco SQLite."""
    conn = sqlite3.connect(db_path)
    query = '''
        SELECT cen_id,
               data_simulacao,
               hora_simulacao,
               BAR_id,

               PLOAD_cenario,
               BESS_init_cenario,

               PGER_total_result,
               PGWIND_total_result,
               PCURTAILMENT_total_result,
               BESS_operation_result,
               BESS_soc_atual_result,
               V_result

        FROM DBAR_results
        LIMIT 500000
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def prepare_data_for_bar(df_bar):
    """
    Para um DataFrame filtrado por uma barra, retorna X (features) e y (target).
    Features: 'data_simulacao' (dia), 'hora_simulacao',
              'PGWIND_total_result', 'PCURTAILMENT_total_result',
              'PLOAD_cenario', 'BESS_init_cenario', 'BESS_soc_atual_result'
    Target: 'BESS_operation_result'
    """
    feature_cols = [
        'data_simulacao',
        'hora_simulacao',
        'PGWIND_total_result',
        'PCURTAILMENT_total_result',
        'PLOAD_cenario',
        'BESS_init_cenario',
        'BESS_soc_atual_result'
    ]
    target_col = 'BESS_operation_result'

    # Remove linhas com valores ausentes (se houver)
    df_clean = df_bar[feature_cols + [target_col]].dropna()

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    return X, y

def plot_predictions(y_test, y_pred, bar_id, num_samples=7, save_dir=None):
    """
    Gera um gráfico de barras comparando valores reais e previstos
    para um conjunto de amostras aleatórias do teste (padrão 7 amostras).
    """
    if save_dir is None:
        save_dir = os.path.join(MODELS_DIR, 'graficos')
    os.makedirs(save_dir, exist_ok=True)

    n_total = len(y_test)
    if n_total < num_samples:
        indices = np.arange(n_total)
    else:
        # Escolhe um início aleatório para um bloco de tamanho num_samples
        inicio = np.random.randint(0, n_total - num_samples + 1)
        indices = np.arange(inicio, inicio + num_samples)

    y_test_sample = y_test[indices]
    y_pred_sample = y_pred[indices]

    x = np.arange(len(indices))  # posições no eixo x

    plt.figure(figsize=(10, 6))
    width = 0.35
    plt.bar(x - width/2, y_test_sample, width, label='Real', color='blue', alpha=0.7)
    plt.bar(x + width/2, y_pred_sample, width, label='Previsto', color='orange', alpha=0.7)

    plt.xlabel('Índice da amostra (no conjunto de teste)')
    plt.ylabel('BESS_operation_result')
    plt.title(f'Barra {bar_id} - Comparação: Real vs Previsto (amostras aleatórias)')
    plt.xticks(x, indices)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plot_path = os.path.join(save_dir, f'bar_{bar_id}_comparacao.png')
    plt.show()
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de barras comparativo salvo em: {plot_path}")

def train_and_evaluate(X, y, bar_id):
    """
    Treina uma MLPRegressor para uma barra específica.
    Retorna o modelo treinado, o scaler (se utilizado) e as métricas (MSE, R2).
    """
    # Divisão treino-teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = None
    if USE_SCALER:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    mlp = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation='relu',
        solver='adam',
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"BAR {bar_id}: MSE = {mse:.6f}, R2 = {r2:.4f} (amostras: {len(X)})")

    # Gráfico comparativo
    plot_predictions(y_test, y_pred, bar_id)

    return mlp, scaler, mse, r2

def save_model_and_scaler(model, scaler, bar_id, models_dir):
    """Salva o modelo e (se existir) o scaler em arquivos .joblib."""
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"mlp_bar_{bar_id}.joblib")
    joblib.dump(model, model_path)
    print(f"Modelo da barra {bar_id} salvo em {model_path}")

    if scaler is not None:
        scaler_path = os.path.join(models_dir, f"scaler_bar_{bar_id}.joblib")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler da barra {bar_id} salvo em {scaler_path}")

def main():
    print("Carregando dados...")
    df = load_data(DB_PATH)
    print(f"Dados carregados: {df.shape[0]} registros, {df.shape[1]} colunas")

    barras = df['BAR_id'].unique()
    print(f"Barras encontradas: {barras}")

    resultados = []

    for bar_id in barras:
        print(f"\n--- Processando barra {bar_id} ---")
        df_bar = df[df['BAR_id'] == bar_id].copy()

        if df_bar.empty:
            print(f"Aviso: barra {bar_id} sem dados. Pulando.")
            continue

        # Verifica se todos os valores de BESS_operation_result são zero
        if (df_bar['BESS_operation_result'] == 0).all():
            print(f"Aviso: barra {bar_id} possui todos os BESS_operation_result iguais a zero. Pulando (sem operação da bateria).")
            continue

        # Informações sobre dias e horas (apenas para exibição)
        n_dias = df_bar['data_simulacao'].nunique()
        n_horas = df_bar['hora_simulacao'].nunique()
        print(f"   Dias: {n_dias}, Horas: {n_horas}, Total de amostras: {len(df_bar)}")

        X, y = prepare_data_for_bar(df_bar)

        if len(X) == 0:
            print(f"Aviso: barra {bar_id} não possui dados completos após remoção de NaN. Pulando.")
            continue

        try:
            modelo, scaler, mse, r2 = train_and_evaluate(X, y, bar_id)
            save_model_and_scaler(modelo, scaler, bar_id, MODELS_DIR)
            resultados.append({'bar_id': bar_id, 'amostras': len(X), 'mse': mse, 'r2': r2})
        except Exception as e:
            print(f"Erro ao treinar para barra {bar_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Resumo final
    print("\n=== RESUMO DO TREINAMENTO ===")
    res_df = pd.DataFrame(resultados)
    if not res_df.empty:
        print(res_df.to_string(index=False))
        res_df.to_csv(os.path.join(MODELS_DIR, 'resumo_treinamento.csv'), index=False)
    else:
        print("Nenhum modelo foi treinado.")

    print(f"\nTodos os modelos e scalers foram salvos em: {MODELS_DIR}")
    print(f"Gráficos comparativos salvos em: {os.path.join(MODELS_DIR, 'graficos')}")

if __name__ == '__main__':
    main()