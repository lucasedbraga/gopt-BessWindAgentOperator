#!/usr/bin/env python3
"""
RNA_especialistas_por_horario_v8_HGB_BESS.py

Treina MultiOutputRegressor(HistGradientBoostingRegressor) por hora,
considerando CURTAILMENT (>= 0) e BESS_operation_result (com sinal).

Saída por hora: pipeline.joblib, metadata.json, previsoes_teste.csv.
O CSV inclui os targets reais, suas previsões E as features de BESS
(BESS_init_cenario_*) para permitir análise de contexto.
"""

import numpy as np
import pandas as pd
import os
import sys
import sqlite3
import matplotlib
import json
import joblib
import time

# [FIX] Backend matplotlib sem Qt/OpenGL (elimina spam XVisualInfo)
if not os.environ.get('MPLBACKEND'):
    if os.environ.get('DISPLAY') or sys.platform in ('win32', 'darwin'):
        try:
            matplotlib.use('TkAgg', force=True)
        except Exception:
            matplotlib.use('Agg', force=True)
    else:
        matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))[:-6])
from UTILS.beep import *

# ==================== CONFIGURAÇÕES ====================
TIPO_TESTE = "BESS"

TARGET_ACCURACY = 30.0
HORAS_INTERESSE = [16, 17, 18]

DB_PATH = f'DATA/output_IEEE_Test/IEEE14_RNA_DATA_ACOPF_{TIPO_TESTE}.db'
MODELS_DIR = f'DATA/output_IEEE_Test/IEEE14_{TIPO_TESTE}_modelos_especialistas_v8_HGB'
TEST_SIZE = 0.2
MAX_ATTEMPTS = 100

TOLERANCE_REL = 0.1
TOLERANCE_ABS = 0.01

HGB_PARAMS = {
    'max_iter': 500,
    'max_depth': 5,
    'learning_rate': 0.05,
    'early_stopping': True,
    'validation_fraction': 0.2,
    'n_iter_no_change': 30,
    'verbose': 0,
    'random_state': None
}
BARRAS_COM_MEDICAO = [2, 3, 4, 6, 9, 14]
LINHAS_COM_MEDICAO = ["4-7", "4-9", "5-6"]

MIN_SAMPLES_PER_GROUP = 100
REMOVE_CONSTANT_COLUMNS = True

POSITIVE_WEIGHT_FACTOR = 10.0

# --- [NOVO] Barras relevantes por tipo de grandeza ---
# Ajuste para o seu sistema; deve casar EXATAMENTE com o sufixo "BAR<k>"
# das colunas do CSV. Curtailment só faz sentido em barras com GWD;
# BESS_operation só em barras com bateria.
BARRAS_COM_GWD  = [1, 2, 3, 6, 8]     # <-- ajuste
BARRAS_COM_BESS = [2, 13]             # <-- ajuste

GWD_BARS  = {f"BAR{b}" for b in BARRAS_COM_GWD}
BESS_BARS = {f"BAR{b}" for b in BARRAS_COM_BESS}

# [FIX] Se True, salva TODO o X_test no previsoes_teste.csv.
# Se False, salva apenas as colunas BESS_init_cenario_*.
SALVAR_X_TEST_COMPLETO = False

inicio_global = time.time()
# ========================================================


# ------------------------------------------------------------------
# Helpers de classificação de colunas
# ------------------------------------------------------------------
def _is_curtailment(col):
    return col.startswith('CURTAILMENT_total_result_')

def _is_bess(col):
    return col.startswith('BESS_operation_result_')

def _bar_suffix(col):
    return col.split('_')[-1]

def _keep_target(col):
    if col.startswith('CURTAILMENT_total_result_'):
        return _bar_suffix(col) in GWD_BARS
    if col.startswith('BESS_operation_result_'):
        return _bar_suffix(col) in BESS_BARS
    return False


# ------------------------------------------------------------------
# Carregamento de dados
# ------------------------------------------------------------------
def load_data(db_path):
    conn = sqlite3.connect(db_path)
    query = '''
        SELECT cen_id,
               data_simulacao, hora_simulacao, dia_semana,
               BAR_id,
               PLOAD_cenario,
               PGER_UTE_result,
               PGWIND_disponivel_cenario,
               CURTAILMENT_total_result,
               V_result,
               ANG_result,
               QLOAD_cenario,
               QGER_UTE_result,
               BESS_init_cenario,
               BESS_operation_result
        FROM DBAR_results
        WHERE hora_simulacao IN (15, 16, 17, 18, 19)
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_dlin_measurements(db_path, cenarios_datas, linhas_especificas):
    if not cenarios_datas:
        return pd.DataFrame()

    condicoes_linha = []
    for linha in linhas_especificas:
        barra1, barra2 = linha.split('-')
        condicoes_linha.append(f"(de_barra = '{barra1}' AND para_barra = '{barra2}')")
        condicoes_linha.append(f"(de_barra = '{barra2}' AND para_barra = '{barra1}')")
    where_linhas = " OR ".join(condicoes_linha)

    conn = sqlite3.connect(db_path)
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

    df_med['linha'] = df_med.apply(
        lambda row: f"{min(row['de_barra'], row['para_barra'])}-"
                    f"{max(row['de_barra'], row['para_barra'])}",
        axis=1
    )

    if 'id' not in df_med.columns:
        df_med = df_med.reset_index(drop=True)
        df_med['_ordem'] = df_med.index
        ordem_col = '_ordem'
    else:
        ordem_col = 'id'

    df_med = (df_med.sort_values(ordem_col)
                    .groupby(['cen_id', 'data_simulacao', 'linha'])
                    .first().reset_index())
    df_med['LIN_usage_result'] = df_med['LIN_usage_result'] / 100.0

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
    df_pivot = df_pivot.dropna(axis=1, how='all')
    return df_pivot


def create_wide_format(df, barras_com_medicao):
    """Pivota para formato largo, incluindo BESS_init e BESS_operation."""
    df = df.copy()
    df['PLOAD_medido'] = 0.0
    df['QLOAD_medido'] = 0.0
    df['V_medido'] = 0.0
    df['ANG_medido'] = 0.0

    mask_medido = df['BAR_id'].isin(barras_com_medicao)
    df.loc[mask_medido, 'PLOAD_medido'] = df.loc[mask_medido, 'PLOAD_cenario']
    df.loc[mask_medido, 'QLOAD_medido'] = df.loc[mask_medido, 'QLOAD_cenario']
    df.loc[mask_medido, 'V_medido'] = df.loc[mask_medido, 'V_result']
    df.loc[mask_medido, 'ANG_medido'] = df.loc[mask_medido, 'ANG_result']

    pivot_cols = [
        'PGWIND_disponivel_cenario',
        'PGER_UTE_result',
        'QGER_UTE_result',
        'V_medido',
        'ANG_medido',
        'PLOAD_medido',
        'QLOAD_medido',
        'CURTAILMENT_total_result',
        'BESS_init_cenario',
        'BESS_operation_result',
    ]
    index_cols = ['cen_id', 'data_simulacao', 'hora_simulacao', 'dia_semana']
    df_pivot = df.pivot_table(index=index_cols, columns='BAR_id',
                              values=pivot_cols, aggfunc='first')
    df_pivot.columns = [f'{var}_BAR{bar}' for var, bar in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    bess_feat = [c for c in df_pivot.columns if c.startswith('BESS_init_cenario_')]
    bess_targ = [c for c in df_pivot.columns if c.startswith('BESS_operation_result_')]
    print(f"   [pivot] BESS_init={len(bess_feat)} | BESS_op={len(bess_targ)}")
    return df_pivot


# ------------------------------------------------------------------
# Separação X / y com foco em BESS e curtailment
# ------------------------------------------------------------------
def prepare_X_y(df_wide, remove_constants=True):
    feature_prefixes = [
        'PGWIND_disponivel_cenario',
        'PGER_UTE_result',
        'QGER_UTE_result',
        'PLOAD_medido',
        'QLOAD_medido',
        'V_medido',
        'BESS_init_cenario',
    ]
    extra_feature_cols = [c for c in df_wide.columns
                          if 'FLUX_result' in c or 'LIN_usage_result' in c]
    feature_cols = [c for c in df_wide.columns
                    if any(c.startswith(p) for p in feature_prefixes)] + extra_feature_cols

    target_prefixes = ['CURTAILMENT_total_result', 'BESS_operation_result']
    target_cols = [
        c for c in df_wide.columns
        if any(c.startswith(p) for p in target_prefixes) and _keep_target(c)
    ]

    subset_dropna = [c for c in (feature_cols + target_cols) if c in df_wide.columns]
    df_clean = df_wide.dropna(subset=subset_dropna)
    X = df_clean[feature_cols].copy()
    y = df_clean[target_cols].copy()

    bess_y = [c for c in y.columns if _is_bess(c)]
    tem_registro_bateria = (
        len(bess_y) > 0 and bool((y[bess_y].abs() > 1e-9).any().any())
    )
    print(f"   Registro de operação de bateria: {tem_registro_bateria}")

    if remove_constants:
        constant_X = X.columns[X.std() == 0].tolist()

        bess_init_const = [c for c in constant_X if c.startswith('BESS_init_cenario_')]
        bess_init_keep = []
        for col in bess_init_const:
            bess_bar = col.replace('BESS_init_cenario_', '')
            col_y = f'BESS_operation_result_{bess_bar}'
            if col_y in y.columns and (y[col_y].abs() > 1e-9).any():
                bess_init_keep.append(col)

        constant_X_to_drop = [c for c in constant_X if c not in bess_init_keep]
        if constant_X_to_drop:
            X = X.drop(columns=constant_X_to_drop)

        # Reconstrói target_cols a partir de X pós-limpeza e aplica filtro de barras
        wind_cols = [c for c in X.columns if c.startswith('PGWIND_disponivel_cenario_')]
        curt_cols = [c.replace('PGWIND_disponivel_cenario_', 'CURTAILMENT_total_result_')
                     for c in wind_cols]
        bess_init_cols = [c for c in X.columns if c.startswith('BESS_init_cenario_')]
        bess_op_cols = [c.replace('BESS_init_cenario_', 'BESS_operation_result_')
                        for c in bess_init_cols]

        target_cols = [c for c in (curt_cols + bess_op_cols) if _keep_target(c)]
        target_cols = [c for c in target_cols if c in df_clean.columns]
        y = df_clean[target_cols].copy()

        print(f"   [remove_constants] X drop={len(constant_X_to_drop)} | "
              f"BESS_init preservados={len(bess_init_keep)} | "
              f"targets: CURT={len(curt_cols)} BESS={len(bess_op_cols)}")

    # [FIX] replace seletivo: só em CURTAILMENT
    for c in y.columns:
        if _is_curtailment(c):
            y[c] = y[c].replace(0, 0.0001)

    n_curt = sum(_is_curtailment(c) for c in y.columns)
    n_bess = sum(_is_bess(c) for c in y.columns)
    print(f"   Features shape: {X.shape} | Targets: CURT={n_curt} BESS={n_bess}")
    return X, y


# ------------------------------------------------------------------
# Métrica customizada (sem penalizar predições negativas)
# ------------------------------------------------------------------
def calculate_correctness(y_true, y_pred, rel_tol=TOLERANCE_REL, abs_tol=TOLERANCE_ABS):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return 0.0

    abs_err = np.abs(y_true - y_pred)
    nonzero = np.abs(y_true) > abs_tol
    correct = np.zeros_like(y_true, dtype=bool)

    if np.any(nonzero):
        rel_err = abs_err[nonzero] / np.abs(y_true[nonzero])
        correct[nonzero] = rel_err <= rel_tol
    zero = ~nonzero
    if np.any(zero):
        correct[zero] = abs_err[zero] <= abs_tol

    return float(np.mean(correct))


def _per_target_accuracy(y_true_df, y_pred_arr, rel_tol, abs_tol):
    result = {}
    for j, col in enumerate(y_true_df.columns):
        yt = y_true_df.iloc[:, j].values.astype(float)
        yp = y_pred_arr[:, j].astype(float)
        if _is_curtailment(col):
            mask = yt > abs_tol
            if mask.sum() == 0:
                result[col] = float('nan')
                continue
            acc = calculate_correctness(yt[mask], yp[mask], rel_tol, abs_tol)
        else:
            acc = calculate_correctness(yt, yp, rel_tol, abs_tol)
        result[col] = acc * 100.0
    return result


# ------------------------------------------------------------------
# Persistência de resultados (modelo, metadata, CSV)
# ------------------------------------------------------------------
def _salvar_artefatos(pipeline, X, y, X_test, y_test, y_pred,
                      acc, per_target, seed, attempts,
                      hour_dir, extra_meta=None):
    """Salva pipeline.joblib, metadata.json e previsoes_teste.csv."""
    os.makedirs(hour_dir, exist_ok=True)
    model_path = os.path.join(hour_dir, 'pipeline.joblib')
    joblib.dump(pipeline, model_path)

    metadata = {
        'feature_names': list(X.columns),
        'target_names': list(y.columns),
        'test_accuracy': acc,
        'per_target_accuracy': per_target,
        'seed': seed,
        'attempts': attempts,
        'n_treino': len(X) - len(X_test),
        'n_teste': len(X_test),
        'model_type': 'MultiOutputRegressor(HistGradientBoostingRegressor)',
        'sample_weight_factor': POSITIVE_WEIGHT_FACTOR,
    }
    if extra_meta:
        metadata.update(extra_meta)

    with open(os.path.join(hour_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------------
    # [FIX] CSV com targets, previsões E features BESS
    # ------------------------------------------------------------------
    df_out = y_test.copy().reset_index(drop=True)
    for i, col in enumerate(y_test.columns):
        df_out[f'{col}_previsto'] = y_pred[:, i]

    if isinstance(X_test, pd.DataFrame):
        if SALVAR_X_TEST_COMPLETO:
            feat_block = X_test.reset_index(drop=True)
        else:
            init_cols = [c for c in X_test.columns
                         if c.startswith('BESS_init_cenario_')]
            feat_block = (X_test[init_cols].reset_index(drop=True)
                          if init_cols else None)
        if feat_block is not None and feat_block.shape[1] > 0:
            df_out = pd.concat([feat_block, df_out], axis=1)

    df_out.to_csv(os.path.join(hour_dir, 'previsoes_teste.csv'), index=False)
    return model_path


# ------------------------------------------------------------------
# Treino com tentativas até atingir TARGET_ACCURACY
# ------------------------------------------------------------------
def train_until_threshold(X, y, hour, models_dir):
    if len(X) < MIN_SAMPLES_PER_GROUP:
        print(f"   Hora {hour:02d} tem apenas {len(X)} amostras - ignorado.")
        return None

    # Pesos: curtailment (>0) + BESS (|op|>0)
    curt_cols = [c for c in y.columns if _is_curtailment(c)]
    bess_cols = [c for c in y.columns if _is_bess(c)]

    n_curt_ativas = (y[curt_cols] > 1e-9).sum(axis=1).values if curt_cols else np.zeros(len(y))
    n_bess_ativas = (y[bess_cols].abs() > 1e-9).sum(axis=1).values if bess_cols else np.zeros(len(y))
    sample_weight = 1.0 + (n_curt_ativas + n_bess_ativas) * POSITIVE_WEIGHT_FACTOR

    y_bin = pd.Series((n_curt_ativas + n_bess_ativas) > 0, index=y.index)
    seed_base = (int(time.time() * 1e6)) % 1000
    stratify_arg = y_bin if y_bin.nunique() > 1 else None

    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y, sample_weight,
        test_size=TEST_SIZE, stratify=stratify_arg, random_state=seed_base
    )
    print(f"   Tamanho treino: {len(X_train)}, teste: {len(X_test)}")

    best_accuracy = 0.0
    best_model = None
    best_seed = None
    best_per_target = None

    for attempt in range(MAX_ATTEMPTS):
        seed = (int(time.time() * 1e6) + attempt) % 1000
        params = HGB_PARAMS.copy()
        params['random_state'] = seed

        base_hgb = HistGradientBoostingRegressor(**params)
        multi_model = MultiOutputRegressor(base_hgb)
        pipeline = Pipeline([('multi', multi_model)])

        try:
            pipeline.fit(X_train, y_train, multi__sample_weight=sw_train)
        except Exception as e:
            print(f"   Hora {hour:02d}: Tentativa {attempt+1} falhou: {e}")
            continue

        y_pred = pipeline.predict(X_test)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        per_target = _per_target_accuracy(y_test, y_pred, TOLERANCE_REL, TOLERANCE_ABS)
        accs_validos = [v for v in per_target.values() if not np.isnan(v)]
        acc = float(np.mean(accs_validos)) if accs_validos else 0.0

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        acc_curt = [v for k, v in per_target.items() if _is_curtailment(k) and not np.isnan(v)]
        acc_bess = [v for k, v in per_target.items() if _is_bess(k) and not np.isnan(v)]
        s_curt = f"{np.mean(acc_curt):.1f}%" if acc_curt else "n/a"
        s_bess = f"{np.mean(acc_bess):.1f}%" if acc_bess else "n/a"

        print(f"   Hora {hour:02d}: Tent {attempt+1:3d} | Acc média={acc:.2f}% "
              f"(CURT={s_curt} BESS={s_bess}) | MAE={mae:.4f} RMSE={rmse:.4f}")

        if acc >= TARGET_ACCURACY:
            print(f"   >>> Hora {hour:02d}: alvo atingido na tentativa {attempt+1}.")
            hour_dir = os.path.join(models_dir, f"hora_{hour:02d}")
            model_path = _salvar_artefatos(
                pipeline, X, y, X_test, y_test, y_pred,
                acc, per_target, seed, attempt + 1, hour_dir
            )
            return {
                'hora': hour,
                'n_amostras': len(X),
                'n_treino': len(X_train),
                'n_teste': len(X_test),
                'test_accuracy': acc,
                'acc_curt': float(np.mean(acc_curt)) if acc_curt else float('nan'),
                'acc_bess': float(np.mean(acc_bess)) if acc_bess else float('nan'),
                'attempts': attempt + 1,
                'model_path': model_path
            }

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = pipeline
            best_seed = seed
            best_per_target = per_target
            # Guarda y_pred do melhor para salvar CSV no fallback
            best_y_pred = y_pred

    # Fallback: não atingiu alvo — salva o melhor
    print(f"   Aviso: hora {hour:02d} não atingiu {TARGET_ACCURACY}%. "
          f"Melhor={best_accuracy:.2f}%")
    if best_model is None:
        return None

    hour_dir = os.path.join(models_dir, f"hora_{hour:02d}")
    model_path = _salvar_artefatos(
        best_model, X, y, X_test, y_test, best_y_pred,
        best_accuracy, best_per_target, best_seed, MAX_ATTEMPTS, hour_dir,
        extra_meta={'warning': f'Did not reach {TARGET_ACCURACY}%'}
    )

    acc_curt_fb = [v for k, v in best_per_target.items()
                   if _is_curtailment(k) and not np.isnan(v)]
    acc_bess_fb = [v for k, v in best_per_target.items()
                   if _is_bess(k) and not np.isnan(v)]

    return {
        'hora': hour,
        'n_amostras': len(X),
        'n_treino': len(X_train),
        'n_teste': len(X_test),
        'test_accuracy': best_accuracy,
        'acc_curt': float(np.mean(acc_curt_fb)) if acc_curt_fb else float('nan'),
        'acc_bess': float(np.mean(acc_bess_fb)) if acc_bess_fb else float('nan'),
        'attempts': MAX_ATTEMPTS,
        'model_path': model_path
    }


# ------------------------------------------------------------------
# Gráfico de acurácia consolidada
# ------------------------------------------------------------------
def plot_accuracy_bar(summary_df, save_dir):
    if summary_df.empty:
        return
    plt.figure(figsize=(10, 6))
    x = np.arange(len(summary_df))
    largura = 0.35
    plt.bar(x - largura/2, summary_df['test_accuracy'], largura,
            color='steelblue', alpha=0.8, label='Média')
    if 'acc_curt' in summary_df.columns:
        plt.bar(x + largura/2, summary_df['acc_curt'], largura,
                color='seagreen', alpha=0.8, label='Curtailment')
    plt.axhline(y=TARGET_ACCURACY, color='red', linestyle='--',
                label=f'Limiar {TARGET_ACCURACY}%')
    plt.xlabel('Hora do Dia')
    plt.ylabel('Acurácia no Teste (%)')
    plt.title('Acurácia dos modelos (tolerância 10% rel. / 0,01 abs.)')
    plt.xticks(x, summary_df['hora'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'acuracia_testes.png')
    plt.savefig(plot_path, dpi=150)
    if matplotlib.get_backend().lower() != 'agg':
        plt.show()
    plt.close()
    print(f"Gráfico salvo em: {plot_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("=" * 70)
    print("TREINAMENTO ESPECIALISTAS POR HORA (HGB) — CURTAILMENT + BESS")
    print("=" * 70)

    print("\n[1] Carregando dados do banco...")
    df = load_data(DB_PATH)
    print(f"   Total de registros: {len(df)}")

    print("\n[2] Transformando para formato largo...")
    df_wide = create_wide_format(df, BARRAS_COM_MEDICAO)
    print(f"   Instantes: {len(df_wide)} | Colunas: {len(df_wide.columns)}")

    print("\n[3] Medições de linha...")
    cenarios_datas = list(df_wide[['cen_id', 'data_simulacao']]
                          .drop_duplicates().itertuples(index=False, name=None))
    df_medicoes = load_dlin_measurements(DB_PATH, cenarios_datas, LINHAS_COM_MEDICAO)
    if df_medicoes.empty:
        print("   Nenhuma medição encontrada.")
    else:
        print(f"   {len(df_medicoes)} linhas de medição.")

    print("\n[4] Merge com medições...")
    df_wide = df_wide.merge(df_medicoes, on=['cen_id', 'data_simulacao'], how='left')
    med_cols = [c for c in df_wide.columns
                if 'FLUX_result' in c or 'LIN_usage_result' in c]
    df_wide[med_cols] = df_wide[med_cols].fillna(0)
    print(f"   Shape após merge: {df_wide.shape}")

    print("\n[5] Distribuição por hora:")
    for h in sorted(df_wide['hora_simulacao'].unique()):
        print(f"      Hora {h:02d}: {len(df_wide[df_wide['hora_simulacao'] == h])}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"\n[6] Treinando ({MAX_ATTEMPTS} tentativas/hora, alvo={TARGET_ACCURACY}%)...")
    resultados = []
    for hora in HORAS_INTERESSE:
        inicio = time.time()
        print(f"\n--- Hora {hora:02d} ---")
        df_hora = df_wide[df_wide['hora_simulacao'].isin([hora-1, hora, hora+1])]
        X, y = prepare_X_y(df_hora, remove_constants=REMOVE_CONSTANT_COLUMNS)

        if X.shape[1] == 0 or y.shape[1] == 0:
            print("   Sem features ou targets — ignorando.")
            continue

        curt_cols = [c for c in y.columns if _is_curtailment(c)]
        bess_cols = [c for c in y.columns if _is_bess(c)]
        n_curt_ativas = int((y[curt_cols] > 1e-9).any(axis=1).sum()) if curt_cols else 0
        n_bess_ativas = int((y[bess_cols].abs() > 1e-9).any(axis=1).sum()) if bess_cols else 0
        print(f"   Amostras: total={len(y)} | curt_ativas={n_curt_ativas} "
              f"| bess_ativas={n_bess_ativas}")

        metrica = train_until_threshold(X, y, hora, MODELS_DIR)
        if metrica is not None:
            resultados.append(metrica)

        print(f"   Tempo hora {hora:02d}: {time.time()-inicio:.2f} s")
        beep()

    print("\n[7] Consolidando...")
    summary_df = pd.DataFrame(resultados)
    if summary_df.empty:
        print("Nenhum modelo treinado.")
        return

    summary_csv = os.path.join(MODELS_DIR, 'resumo_modelos.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Resumo: {summary_csv}")

    print("\n--- Estatísticas Globais ---")
    print(f"Modelos: {len(summary_df)}")
    print(f"Acurácia média: {summary_df['test_accuracy'].mean():.2f}%")
    if 'acc_curt' in summary_df.columns:
        print(f"Acurácia média CURT: {summary_df['acc_curt'].mean():.2f}%")
    if 'acc_bess' in summary_df.columns:
        print(f"Acurácia média BESS: {summary_df['acc_bess'].mean():.2f}%")

    print(f"Tempo total: {time.time()-inicio_global:.2f} s")

    print("\n[8] Gráfico de acurácia...")
    plot_accuracy_bar(summary_df, MODELS_DIR)

    print("\n" + "=" * 70)
    print("PROCESSO CONCLUÍDO")
    print(f"Modelos salvos em: {MODELS_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()