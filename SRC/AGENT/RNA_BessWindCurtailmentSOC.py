# RNA_BessWindCurtailmentSOC.py
"""
Treina uma rede neural (MLPRegressor) para prever curtailment, carregamento e SOC da bateria
com base em cenários identificados por cen_id.
"""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Carregamento dos dados
def load_data(db_path):
    import sqlite3
    conn = sqlite3.connect(db_path)
    # Exemplo: detalhes_barras deve conter cen_id, curtailment, carregamento, soc
    query = '''
        SELECT cen_id,
               
               PGER_total_result,
               PGWIND_total_result,
               PCURTAILMENT_total_result,
               V_result,
               PLOAD_cenario,
               PBESS_init_cenario,
               
               PBESS_inst_result,
               PBESS_soc_result,
        FROM DBAR_results
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def prepare_data(df):
    X = df[['cen_id']].values  # Entrada: apenas o identificador do cenário
    y = df[['curtailment', 'carregamento', 'soc']].values  # Saídas
    return X, y

def train_mlp(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'MSE: {mse:.4f}, R2: {r2:.4f}')
    return mlp

def save_model(model, path):
    joblib.dump(model, path)

def main():
    db_path = 'DATA/output/resultados_PL.db'  # Caminho do banco
    df = load_data(db_path)
    X, y = prepare_data(df)
    mlp = train_mlp(X, y)
    save_model(mlp, 'DATA/output/mlp_curtailment_soc.joblib')
    print('Modelo treinado e salvo!')

if __name__ == '__main__':
    main()
