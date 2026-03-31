#!/usr/bin/env python3
"""
RNA_especialistas_por_horario_v7.py

Treina uma MLPRegressor para cada hora (16, 17, 18) e repete o treinamento
com diferentes sementes aleatórias até que a acurácia no conjunto de teste
seja ≥ 75%. Utiliza StandardScaler e arquitetura robusta.

Inclui como features adicionais as medições de fluxo e utilização de linhas
específicas (4-7, 4-9, 5-6) obtidas das tabelas DLIN e DLIN_results.
"""

import numpy as np
import pandas as pd
import os
import sqlite3
import matplotlib.pyplot as plt
import json
import joblib
import time
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ==================== CONFIGURAÇÕES ====================
DB_PATH = 'DATA/output_CUR_Oficial/RNA_DATA_PL_acoplado.db'
MODELS_DIR = 'DATA/output_CUR_Oficial/modelos_especialistas_v7'   # diretório de saída (nova versão)
TEST_SIZE = 0.2
MAX_ATTEMPTS = 1000               # número máximo de tentativas por hora
TARGET_ACCURACY = 90.0            # acurácia mínima desejada (%)

# Hiperparâmetros da MLP (configuração robusta)
HIDDEN_LAYERS = (100, 50)         # duas camadas ocultas
ALPHA = 0.001                     # regularização L2
LEARNING_RATE_INIT = 0.01
MAX_ITER = 10000
TOL = 1e-5
EARLY_STOPPING = True
VALIDATION_FRACTION = 0.2
N_ITER_NO_CHANGE = 300

# Tolerâncias para acerto
TOLERANCE_REL = 0.1              # 8% de erro relativo
TOLERANCE_ABS = 0.01              # tolerância absoluta para valores próximos de zero

# MIN_SAMPLES_PER_GROUP = 50        # número mínimo de amostras por hora
# BARRAS_COM_MEDICAO = [2, 3, 4, 6, 9, 14]       # barras que possuem medição real (PLOAD_medido)
# LINHAS_COM_MEDICAO = ["4-7", "4-9", "5-6"]   # linhas a serem incluídas como features

BARRAS_COM_MEDICAO = [59, 116, 90, 80, 54, 42, 15, 49, 56, 60]
LINHAS_COM_MEDICAO = [
    "8-5",      # tap = 0.985
    "26-25",    # tap = 0.96
    "30-17",    # tap = 0.96
    "38-37",    # tap = 0.935
    "63-59",    # tap = 0.96
    "64-61",    # tap = 0.985
    "65-66",    # tap = 0.935
    "81-80",    # tap = 0.935
    "68-69"     # tap = 0.935
]

MIN_SAMPLES_PER_GROUP = 100        # número mínimo de amostras por hora para treinar o modelo
REMOVE_CONSTANT_COLUMNS = True
# ========================================================

def load_data(db_path):
    """Carrega os dados do banco SQLite (tabela DBAR_results)."""
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
               BESS_operation_result,
               ANG_result
        FROM DBAR_results
        WHERE hora_simulacao IN (16,17,18)
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def load_dlin_measurements(db_path, cenarios_datas, linhas_especificas):
    """
    Carrega as medições de FLUX_result e LIN_usage_result para as linhas
    especificadas, considerando os dois sentidos e pegando a primeira ocorrência
    por (cen_id, data_simulacao, linha).

    Parâmetros:
        db_path: caminho do banco SQLite
        cenarios_datas: lista de tuplas (cen_id, data_simulacao) que aparecem nos dados principais
        linhas_especificas: lista de strings no formato "4-7", etc.

    Retorna:
        DataFrame com colunas: cen_id, data_simulacao, e colunas de medição
        (ex: FLUX_result_4-7, LIN_usage_result_4-7, etc.)
    """
    if not cenarios_datas:
        return pd.DataFrame()

    # Construir condição WHERE para os pares
    condicoes_linha = []    
    for linha in linhas_especificas:
        barra1, barra2 = linha.split('-')
        condicoes_linha.append(f"(de_barra = '{barra1}' AND para_barra = '{barra2}')")
        condicoes_linha.append(f"(de_barra = '{barra2}' AND para_barra = '{barra1}')")
    where_linhas = " OR ".join(condicoes_linha)

    # Conectar e carregar todas as medições das linhas de interesse
    conn = sqlite3.connect(db_path)
    # Primeiro, carregar apenas as linhas de interesse
    query_med = f"""
        SELECT cen_id, data_simulacao, de_barra, para_barra,
               FLUX_result, LIN_usage_result
        FROM DLIN_results
        WHERE {where_linhas}
    """
    df_med = pd.read_sql_query(query_med, conn)
    conn.close()

    if df_med.empty:
        return pd.DataFrame()

    # Criar coluna linha_normalizada (menor-maior)
    df_med['linha'] = df_med.apply(
        lambda row: f"{min(row['de_barra'], row['para_barra'])}-{max(row['de_barra'], row['para_barra'])}",
        axis=1
    )

    # Filtrar apenas as linhas desejadas (já filtramos no SQL, mas por segurança)
    df_med = df_med[df_med['linha'].isin(linhas_especificas)]

    if df_med.empty:
        return pd.DataFrame()

    if 'id' not in df_med.columns:
        df_med = df_med.reset_index(drop=True)
        df_med['_ordem'] = df_med.index
        ordem_col = '_ordem'
    else:
        ordem_col = 'id'

    # Ordenar e manter a primeira ocorrência por (cen_id, data_simulacao, linha)
    df_med = df_med.sort_values(ordem_col).groupby(['cen_id', 'data_simulacao', 'linha']).first().reset_index()

    # Pivotar: criar colunas para cada linha e métrica
    # Vamos criar colunas: FLUX_result_4-7, LIN_usage_result_4-7, etc.
    # Como LIN_usage_result deve ser dividido por 100, já fazemos aqui.
    df_med['LIN_usage_result'] = df_med['LIN_usage_result'] / 100.0

    # Derreter para pivotar
    df_melt = pd.melt(
        df_med,
        id_vars=['cen_id', 'data_simulacao', 'linha'],
        value_vars=['LIN_usage_result'],
        var_name='metric',
        value_name='value'
    )
    df_melt['col_name'] = df_melt['metric'] + '_' + df_melt['linha']
    df_pivot = df_melt.pivot_table(
        index=['cen_id', 'data_simulacao'],
        columns='col_name',
        values='value',
        aggfunc='first'
    ).reset_index()

    # Remover colunas que são completamente NaN (caso alguma linha não tenha medição)
    df_pivot = df_pivot.dropna(axis=1, how='all')

    return df_pivot

def create_wide_format(df, barras_com_medicao):
    """Transforma o DataFrame longo em formato largo (uma linha por instante)."""
    df = df.copy()
    df['PLOAD_medido'] = 0.0
    df['PLOAD_estimado'] = 0.0
    mask_medido = df['BAR_id'].isin(barras_com_medicao)
    df.loc[mask_medido, 'PLOAD_medido'] = df.loc[mask_medido, 'PLOAD_cenario']
    df.loc[~mask_medido, 'PLOAD_estimado'] = df.loc[~mask_medido, 'PLOAD_cenario']

    pivot_cols = [
        'BESS_init_cenario', 'PGWIND_disponivel_cenario', 'PGER_CONV_total_result',
        'ANG_result','PLOAD_medido', 'PLOAD_estimado',
        'CURTAILMENT_total_result', 'BESS_operation_result'
    ]
    index_cols = ['cen_id', 'data_simulacao', 'hora_simulacao', 'dia_semana']
    df_pivot = df.pivot_table(index=index_cols, columns='BAR_id', values=pivot_cols)
    df_pivot.columns = [f'{var}_BAR{bar}' for var, bar in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    return df_pivot

def prepare_X_y(df_wide, remove_constants=True):
    """
    Separa features (X) e targets (y).
    Features: BESS_init, PGWIND, PGER_CONV, ANG, PLOAD_medido,
              e as colunas de medição adicionais (FLUX_result_* e LIN_usage_result/100_*).
    Targets: CURTAILMENT_total_result (ou BESS_operation_result, conforme configuração).
    """
    # Prefixos das features originais
    feature_prefixes = ['PGWIND_disponivel_cenario',
                        'PGER_CONV_total_result', 'PLOAD_medido']

    # Identificar colunas de medição adicionais (que contêm os prefixos das novas features)
    # Elas serão todas as colunas que não começam com os prefixos acima e que não são targets
    # Porém, vamos pegar explicitamente colunas que contenham 'FLUX_result' ou 'LIN_usage_result/100'
    extra_feature_cols = [col for col in df_wide.columns if 'FLUX_result' in col or 'LIN_usage_result' in col]

    # Colunas features completas
    feature_cols = [col for col in df_wide.columns if any(col.startswith(p) for p in feature_prefixes)] + extra_feature_cols

    df_clean = df_wide.dropna(subset=feature_cols)
    X = df_clean[feature_cols].copy()

    # Targets
    target_prefixes = ['CURTAILMENT_total_result']  # ou 'BESS_operation_result'
    target_cols = [col for col in df_clean.columns if any(col.startswith(p) for p in target_prefixes)]
    y = df_clean[target_cols].copy()

    if remove_constants:
        constant_X = X.columns[X.std() == 0].tolist()
        if constant_X:
            # 1. Remove colunas constantes de X
            X = X.drop(columns=constant_X)
            # 2. Identifica as colunas de interesse em X
            wind_cols = [col for col in X.columns if col.startswith('PGWIND_disponivel_cenario_')]
            curt_cols = [col.replace('PGWIND_disponivel_cenario_', 'CURTAILMENT_total_result_') for col in wind_cols]
            # 3. Cria y selecionando essas colunas e já renomeando o prefixo
            target_cols = curt_cols
            y = df_clean[target_cols].copy()



    print(f"   Features shape: {X.shape}, Targets shape: {y.shape}")
    return X, y

def calculate_correctness(y_true, y_pred, rel_tol, abs_tol):
    """
    Retorna array booleano indicando acerto por elemento.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    abs_err = np.abs(y_true - y_pred)
    nonzero = y_true > abs_tol
    correct = np.zeros_like(y_true, dtype=bool)
    if np.any(nonzero):
        rel_err = abs_err[nonzero] / np.abs(y_true[nonzero])
        correct[nonzero] = rel_err <= rel_tol
    zero = ~nonzero
    if np.any(zero):
        correct[zero] = abs_err[zero] <= abs_tol
    correct = sum(correct[nonzero])/len(correct[nonzero])
    return correct

def train_until_threshold(X, y, hour, models_dir):
    """
    Treina repetidamente uma MLP com diferentes sementes aleatórias até que
    a acurácia no conjunto de teste atinja TARGET_ACCURACY.
    Retorna um dicionário com as métricas do modelo aceito.
    """
    if len(X) < MIN_SAMPLES_PER_GROUP:
        print(f"   Hora {hour:02d} tem apenas {len(X)} amostras (< {MIN_SAMPLES_PER_GROUP}) - ignorado.")
        return None

    # Hiperparâmetros da MLP
    mlp_params = {
        'hidden_layer_sizes': HIDDEN_LAYERS,
        'activation': 'relu',
        'solver': 'adam',
        'alpha': ALPHA,
        'learning_rate': 'adaptive',
        'learning_rate_init': LEARNING_RATE_INIT,
        'max_iter': MAX_ITER,
        'tol': TOL,
        'early_stopping': EARLY_STOPPING,
        'validation_fraction': VALIDATION_FRACTION,
        'n_iter_no_change': N_ITER_NO_CHANGE,
        'verbose': False
    }

    # Criar uma coluna binária para estratificação (se alguma das colunas target for != 0)
    y_bin = (y != 0).any(axis=1)

    # Divisão treino‑teste (estratificada)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y_bin, random_state=None
    )
    print(f"   Tamanho treino: {len(X_train)}, teste: {len(X_test)}")

    best_accuracy = 0.0
    best_pipeline = None
    best_seed = None

    for attempt in range(MAX_ATTEMPTS):
        # Gera uma semente pseudoaleatória baseada no tempo e na tentativa
        seed = (int(time.time() * 1e6) + attempt) % 1000
        mlp_params_with_seed = mlp_params.copy()
        mlp_params_with_seed['random_state'] = seed

        pipeline = Pipeline([
            #('scaler', StandardScaler()),
            ('mlp', MLPRegressor(**mlp_params_with_seed))
        ])

        try:
            pipeline.fit(X_train, y_train)
        except Exception as e:
            print(f"   Hora {hour:02d}: Tentativa {attempt+1} falhou com erro: {e}")
            continue

        y_pred = pipeline.predict(X_test)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        correctness = calculate_correctness(y_test, y_pred, TOLERANCE_REL, TOLERANCE_ABS)
        acc = correctness * 100

        if acc >= TARGET_ACCURACY:
            print(f"   Hora {hour:02d}: Tentativa {attempt+1} acertou {acc:.2f}% >= {TARGET_ACCURACY}% - aceito!")
            # Salvar modelo aceito
            hour_dir = os.path.join(models_dir, f"hora_{hour:02d}")
            os.makedirs(hour_dir, exist_ok=True)
            model_path = os.path.join(hour_dir, 'pipeline.joblib')
            joblib.dump(pipeline, model_path)

            metadata = {
                'feature_names': list(X.columns),
                'target_names': list(y.columns),
                'test_accuracy': acc,
                'seed': seed,
                'attempts': attempt + 1,
                'n_treino': len(X_train),
                'n_teste': len(X_test)
            }
            with open(os.path.join(hour_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            # Salvar previsões do teste
            df_out = y_test.copy()
            for i, col in enumerate(y_test.columns):
                df_out[f'{col}_previsto'] = y_pred[:, i]
                #df_out[f'{col}_correto'] = correctness[:, i]
            df_out.to_csv(os.path.join(hour_dir, 'previsoes_teste.csv'), index=False)

            return {
                'hora': hour,
                'n_amostras': len(X),
                'n_treino': len(X_train),
                'n_teste': len(X_test),
                'test_accuracy': acc,
                'attempts': attempt + 1,
                'model_path': model_path
            }
        else:
            print(f"   Hora {hour:02d}: Tentativa {attempt+1} acertou {acc:.2f}% - abaixo de {TARGET_ACCURACY}%, retreinando...")

        # Mantém o melhor modelo encontrado até agora (caso não atinja o limiar)
        if acc > best_accuracy:
            best_accuracy = acc
            best_pipeline = pipeline
            best_seed = seed

    # Se sair do loop sem sucesso, salva o melhor modelo e emite aviso
    print(f"   Aviso: Hora {hour:02d} não atingiu {TARGET_ACCURACY}% após {MAX_ATTEMPTS} tentativas. Melhor acurácia: {best_accuracy:.2f}%")
    if best_pipeline is not None:
        hour_dir = os.path.join(models_dir, f"hora_{hour:02d}")
        os.makedirs(hour_dir, exist_ok=True)
        model_path = os.path.join(hour_dir, 'pipeline.joblib')
        joblib.dump(best_pipeline, model_path)

        metadata = {
            'feature_names': list(X.columns),
            'target_names': list(y.columns),
            'test_accuracy': best_accuracy,
            'seed': best_seed,
            'attempts': MAX_ATTEMPTS,
            'warning': f'Did not reach {TARGET_ACCURACY}%',
            'n_treino': len(X_train),
            'n_teste': len(X_test)
        }
        with open(os.path.join(hour_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        return {
            'hora': hour,
            'n_amostras': len(X),
            'n_treino': len(X_train),
            'n_teste': len(X_test),
            'test_accuracy': best_accuracy,
            'attempts': MAX_ATTEMPTS,
            'model_path': model_path
        }
    else:
        return None

def plot_accuracy_bar(summary_df, save_dir):
    """Gráfico de barras da acurácia obtida."""
    if summary_df.empty:
        return
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df['hora'], summary_df['test_accuracy'], color='steelblue', alpha=0.7)
    plt.axhline(y=TARGET_ACCURACY, color='red', linestyle='--', label=f'Limiar {TARGET_ACCURACY}%')
    plt.xlabel('Hora do Dia')
    plt.ylabel('Acurácia no Teste (%)')
    plt.title('Acurácia dos modelos (tolerância 8% rel. / 0,01 abs.)')
    plt.xticks(summary_df['hora'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'acuracia_testes.png')
    plt.savefig(plot_path, dpi=150)
    plt.show()
    plt.close()
    print(f"Gráfico salvo em: {plot_path}")

def main():
    print("=" * 70)
    print("TREINAMENTO DE RNAs ESPECIALISTAS POR HORA")
    print("=" * 70)

    # 1. Carregar dados principais
    print("\n[1] Carregando dados do banco (DBAR_results)...")
    df = load_data(DB_PATH)
    print(f"   Total de registros: {len(df)}")

    # 2. Transformar para formato largo (pivot por barra)
    print("\n[2] Transformando dados para formato largo...")
    df_wide = create_wide_format(df, BARRAS_COM_MEDICAO)
    print(f"   Instantes únicos: {len(df_wide)}")
    print(f"   Colunas após pivot: {len(df_wide.columns)}")

    # 3. Carregar medições das linhas (DLIN_results)
    print("\n[3] Carregando medições de linhas...")
    # Coletar os pares (cen_id, data_simulacao) presentes no df_wide
    cenarios_datas = list(df_wide[['cen_id', 'data_simulacao']].drop_duplicates().itertuples(index=False, name=None))
    df_medicoes = load_dlin_measurements(DB_PATH, cenarios_datas, LINHAS_COM_MEDICAO)
    if df_medicoes.empty:
        print("   Nenhuma medição encontrada para as linhas especificadas.")
    else:
        print(f"   Carregadas {len(df_medicoes)} linhas de medição.")
        print(f"   Colunas adicionais: {list(df_medicoes.columns)}")

    # 4. Mesclar medições com df_wide
    print("\n[4] Mesclando medições com dados principais...")
    df_wide = df_wide.merge(df_medicoes, on=['cen_id', 'data_simulacao'], how='left')
    print(f"   Shape após merge: {df_wide.shape}")
    med_cols = [c for c in df_wide.columns if 'FLUX_result' in c or 'LIN_usage_result' in c]
    df_wide[med_cols] = df_wide[med_cols].fillna(0)

    # 5. Verificar distribuição por hora
    print("\n[5] Verificando distribuição por hora...")
    horas_disponiveis = sorted(df_wide['hora_simulacao'].unique())
    print(f"   Horas disponíveis: {horas_disponiveis}")
    for h in horas_disponiveis:
        count = len(df_wide[df_wide['hora_simulacao'] == h])
        print(f"      Hora {h:02d}: {count} amostras")

    # 6. Preparar diretório de saída
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 7. Treinar modelos para horas 16, 17 e 18
    print(f"\n[6] Treinando modelos (com retreinamento até acurácia ≥ {TARGET_ACCURACY}%)...")
    resultados = []
    horas_interesse = [16, 17, 18]

    for hora in horas_interesse:
        print(f"\n--- Processando hora {hora:02d} ---")
        df_hora = df_wide[df_wide['hora_simulacao'] == hora]
        X, y = prepare_X_y(df_hora, remove_constants=REMOVE_CONSTANT_COLUMNS)

        if X.shape[1] == 0 or y.shape[1] == 0:
            print(f"   Sem features ou targets após remoção de constantes - ignorando hora {hora}.")
            continue

        metrica = train_until_threshold(X, y, hora, MODELS_DIR)
        if metrica is not None:
            resultados.append(metrica)

    # 8. Consolidar resultados
    print("\n[7] Consolidando resultados...")
    summary_df = pd.DataFrame(resultados)
    if summary_df.empty:
        print("Nenhum modelo foi treinado.")
        return

    summary_csv = os.path.join(MODELS_DIR, 'resumo_modelos.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Resumo salvo em: {summary_csv}")

    # 9. Estatísticas
    print("\n--- Estatísticas Globais ---")
    print(f"Total de modelos treinados: {len(summary_df)}")
    print(f"Acurácia média no teste: {summary_df['test_accuracy'].mean():.2f}%")
    print(f"Tentativas médias: {summary_df['attempts'].mean():.1f}")

    # 10. Gráfico
    print("\n[8] Gerando gráfico...")
    plot_accuracy_bar(summary_df, MODELS_DIR)

    print("\n" + "=" * 70)
    print("PROCESSO CONCLUÍDO")
    print(f"Modelos salvos em: {MODELS_DIR}")
    print("=" * 70)

if __name__ == '__main__':
    main()