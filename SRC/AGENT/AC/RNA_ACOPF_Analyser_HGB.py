#!/usr/bin/env python3
"""
graficos_finais_HGB.py

Gera gráficos comparando FPO-AC (real) vs HGB (previsto) para:
  1) Wind curtailment — APENAS em barras com gerador eólico (GWD)
  2) Operação de bateria — APENAS em barras com bateria (BESS)
  3) Resíduos (Real - Previsto) por tipo
  4) Distribuição dos erros por tipo
  5) (Opcional) Restrições ativas que causaram curtailment

Requer previsoes_teste.csv em <MODELS_DIR>/hora_16, hora_17, hora_18.
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Backend Matplotlib — DEVE vir antes de importar pyplot
#   Escolhe TkAgg se houver display, senão Agg (só salva em disco).
#   Elimina o spam "No XVisualInfo for format QSurfaceFormat..."
#   causado pelo backend Qt em ambientes sem display/OpenGL.
# ------------------------------------------------------------------
import matplotlib
if not os.environ.get('MPLBACKEND'):
    if os.environ.get('DISPLAY') or sys.platform in ('win32', 'darwin'):
        try:
            matplotlib.use('TkAgg', force=True)
        except Exception:
            matplotlib.use('Agg', force=True)
    else:
        matplotlib.use('Agg', force=True)

import matplotlib.pyplot as plt


# ==================== CONFIGURAÇÕES ====================
MODELS_DIR = ("/home/lucasedbraga/repositorios/ufjf/gopt-BessWindAgentOperator/"
              "DATA/output_IEEE_Test/IEEE14_BESS_modelos_especialistas_v8_HGB")
DB_PATH = "DATA/output_IEEE_Test/IEEE14_RNA_DATA_ACOPF_BESS.db"
HORAS = (16, 17, 18)
CORES_HORA = {16: '#1f77b4', 17: '#ff7f0e', 18: '#2ca02c'}

# ------------------------------------------------------------------
# Barras relevantes por tipo de grandeza
#   As colunas nos CSVs são do tipo:
#       CURTAILMENT_total_result_BAR<k>
#       BESS_operation_result_BAR<k>
#   Apenas as barras listadas abaixo serão plotadas.
#   ATENÇÃO: ajuste estas listas para o seu sistema.  A convenção
#   de índice deve coincidir EXATAMENTE com o sufixo 'BAR<k>'
#   presente no previsoes_teste.csv (não misture 0-based com 1-based).
# ------------------------------------------------------------------
BARRAS_COM_GWD  = [3,14]     # barras com gerador eólico
BARRAS_COM_BESS = [2, 13]             # barras com bateria

GWD_BARS  = {f"BAR{b}" for b in BARRAS_COM_GWD}
BESS_BARS = {f"BAR{b}" for b in BARRAS_COM_BESS}
# ========================================================


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _maybe_show():
    """Mostra a figura se o backend tiver GUI; ignora caso contrário."""
    if matplotlib.get_backend().lower() == 'agg':
        return
    try:
        plt.show()
    except Exception as e:
        print(f"[aviso] Exibição interativa indisponível: {e}")


def _save(fig, path, dpi=200):
    """Salva a figura e respeita o backend."""
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    _maybe_show()
    plt.close(fig)
    print(f"   -> {path}")


def _load_previsoes(models_dir, hora):
    csv_path = os.path.join(models_dir, f"hora_{hora:02d}", "previsoes_teste.csv")
    if not os.path.exists(csv_path):
        print(f"Arquivo {csv_path} não encontrado. Pulando hora {hora}.")
        return None
    return pd.read_csv(csv_path)


def _bar_suffix(col):
    """CURTAILMENT_total_result_BAR8 -> 'BAR8'."""
    return col.split('_')[-1]


def _select_pairs(df, kind):
    """
    Retorna [(real_col, prev_col)] para o tipo pedido, restrito às
    barras relevantes:
      - kind='curtailment' -> CURTAILMENT_total_result_* em GWD_BARS
      - kind='bess'        -> BESS_operation_result_* em BESS_BARS
    """
    if kind == 'curtailment':
        prefix, permitidas = 'CURTAILMENT_total_result_', GWD_BARS
    elif kind == 'bess':
        prefix, permitidas = 'BESS_operation_result_', BESS_BARS
    else:
        raise ValueError(f"kind inválido: {kind}")

    reais = [
        c for c in df.columns
        if c.startswith(prefix)
        and not c.endswith('_previsto')
        and _bar_suffix(c) in permitidas
    ]
    return [(c, c + '_previsto') for c in reais
            if (c + '_previsto') in df.columns]


def _short_label(col):
    return col.split('_')[-1]


# ------------------------------------------------------------------
# 1) Curtailment por barra — APENAS GWD
# ------------------------------------------------------------------
def plot_windcurtailment(models_dir, horas=HORAS, max_points=200):
    print("\n[1] Curtailment por barra (apenas barras com GWD)")
    print(f"    Barras GWD: {sorted(GWD_BARS)}")
    for hora in horas:
        df = _load_previsoes(models_dir, hora)
        if df is None:
            continue
        pares = _select_pairs(df, kind='curtailment')
        if not pares:
            print(f"   hora {hora:02d}: nenhum par após filtro de barras GWD.")
            continue

        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(df), size=min(max_points, len(df)), replace=False)

        for real_col, prev_col in pares:
            real = df[real_col].values[idx]
            pred = df[prev_col].values[idx]

            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.plot(real, linestyle='-', linewidth=1.2, alpha=0.9, label='FPO-AC')
            ax.plot(pred, linestyle='--', linewidth=1.0, alpha=0.75,
                    label='HGB', color='red')

            bar_name = _short_label(real_col)
            ax.set_title(f'WindCurtailment {bar_name} – Hora {hora:02d}',
                         fontweight='bold')
            ax.set_xlabel('Amostra', fontweight='bold')
            ax.set_ylabel('Corte eólico (MW)', fontweight='bold')
            ax.tick_params(axis='both', labelsize=8, width=1.0)
            ax.legend(frameon=False)
            ax.grid(alpha=0.3, linewidth=0.8)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            fig.tight_layout(pad=0.5)

            path = os.path.join(models_dir,
                                f'curtailment_{bar_name}_hora_{hora:02d}.png')
            _save(fig, path, dpi=200)


# ------------------------------------------------------------------
# 2) Operação de bateria por barra — APENAS BESS
# ------------------------------------------------------------------
def plot_bess_operation(models_dir, horas=HORAS, max_points=200):
    print("\n[2] Operação de bateria por barra (apenas barras com BESS)")
    print(f"    Barras BESS: {sorted(BESS_BARS)}")
    for hora in horas:
        df = _load_previsoes(models_dir, hora)
        if df is None:
            continue
        pares = _select_pairs(df, kind='bess')
        if not pares:
            print(f"   hora {hora:02d}: nenhum par após filtro de barras BESS.")
            continue

        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(df), size=min(max_points, len(df)), replace=False)

        for real_col, prev_col in pares:
            real = df[real_col].values[idx]
            pred = df[prev_col].values[idx]

            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.plot(real, linestyle='-', linewidth=1.2, alpha=0.9, label='FPO-AC')
            ax.plot(pred, linestyle='--', linewidth=1.0, alpha=0.75,
                    label='HGB', color='red')
            ax.axhline(0, color='black', linewidth=0.6, alpha=0.6)

            bar_name = _short_label(real_col)
            ax.set_title(f'BESS operation {bar_name} – Hora {hora:02d}',
                         fontweight='bold')
            ax.set_xlabel('Amostra', fontweight='bold')
            ax.set_ylabel('Operação da bateria (pu)  [ + descarga / − carga ]',
                          fontweight='bold')
            ax.tick_params(axis='both', labelsize=8, width=1.0)
            ax.legend(frameon=False)
            ax.grid(alpha=0.3, linewidth=0.8)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            fig.tight_layout(pad=0.5)

            path = os.path.join(models_dir,
                                f'bess_{bar_name}_hora_{hora:02d}.png')
            _save(fig, path, dpi=200)


# ------------------------------------------------------------------
# 3) Resíduos — separado por tipo (curtailment | bess)
# ------------------------------------------------------------------
def plot_residuos(models_dir, kind='curtailment', horas=HORAS):
    titulo = 'Curtailment (GWD)' if kind == 'curtailment' else 'BESS'
    print(f"\n[3] Resíduos — {titulo}")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    algum = False

    for hora in horas:
        df = _load_previsoes(models_dir, hora)
        if df is None:
            continue
        pares = _select_pairs(df, kind=kind)
        if not pares:
            print(f"   hora {hora:02d}: nenhum par após filtro.")
            continue

        reais = np.concatenate([df[r].values for r, _ in pares])
        prevs = np.concatenate([df[p].values for _, p in pares])
        erro = reais - prevs

        cor = CORES_HORA.get(hora, '#333333')
        ax.plot(range(len(erro)), erro, alpha=0.8, linewidth=1.2,
                label=f'Hora {hora:02d}', color=cor)
        ax.axhline(y=np.mean(erro), linestyle='--', linewidth=1.0,
                   color=cor, alpha=0.6)
        algum = True

    if not algum:
        plt.close(fig)
        print(f"   Nada a plotar em resíduos ({kind}).")
        return

    unidade = 'MW' if kind == 'curtailment' else 'pu'
    ax.set_xlabel('Amostra de teste (barras relevantes concatenadas)',
                  fontweight='bold')
    ax.set_ylabel(f'Erro (Real - Previsto) [{unidade}]', fontweight='bold')
    ax.set_title(f'Resíduos – {titulo}', fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    fig.tight_layout(pad=0.5)

    path = os.path.join(models_dir, f'residuos_{kind}.png')
    _save(fig, path, dpi=300)


# ------------------------------------------------------------------
# 4) Distribuição dos erros — separado por tipo
# ------------------------------------------------------------------
def plot_distribuicao_erro(models_dir, kind='curtailment', horas=HORAS):
    titulo = 'Curtailment (GWD)' if kind == 'curtailment' else 'BESS'
    print(f"\n[4] Distribuição dos erros — {titulo}")
    fig, ax = plt.subplots(figsize=(8, 5))
    algum = False

    for hora in horas:
        df = _load_previsoes(models_dir, hora)
        if df is None:
            continue
        pares = _select_pairs(df, kind=kind)
        if not pares:
            print(f"   hora {hora:02d}: nenhum par após filtro.")
            continue

        reais = np.concatenate([df[r].values for r, _ in pares])
        prevs = np.concatenate([df[p].values for _, p in pares])
        erro = reais - prevs
        erro = erro[~np.isnan(erro)]
        if erro.size == 0:
            continue

        media = float(np.mean(erro))
        mediana = float(np.median(erro))
        desvio = float(np.std(erro))
        cor = CORES_HORA.get(hora, '#333333')

        ax.hist(erro, bins=50, density=True, alpha=0.5,
                color=cor, label=f'Hora {hora:02d}')
        ax.axvline(media, color=cor, linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axvline(mediana, color=cor, linestyle=':', linewidth=1.5, alpha=0.8)
        ax.axvspan(media - desvio, media + desvio, alpha=0.1, color=cor)

        print(f"   [{kind}] Hora {hora:02d}: média={media:.4f}, "
              f"mediana={mediana:.4f}, desvio={desvio:.4f}")
        algum = True

    if not algum:
        plt.close(fig)
        print(f"   Nada a plotar em distribuição ({kind}).")
        return

    unidade = 'MW' if kind == 'curtailment' else 'pu'
    ax.set_xlabel(f'Erro (Real - Previsto) [{unidade}]', fontweight='bold')
    ax.set_ylabel('Densidade de probabilidade', fontweight='bold')
    ax.set_title(f'Distribuição dos erros – {titulo}', fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    fig.tight_layout(pad=0.5)

    path = os.path.join(models_dir, f'distribuicao_erro_{kind}.png')
    _save(fig, path, dpi=300)


# ------------------------------------------------------------------
# 5) Restrições ativas que causaram curtailment (opcional)
# ------------------------------------------------------------------
def plot_restricoes_curtailment(models_dir, db_path=None, csv_path=None):
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if not {'tipo_restricao', 'elemento', 'ocorrencias'}.issubset(df.columns):
            print("CSV deve conter: tipo_restricao, elemento, ocorrencias")
            return
    elif db_path and os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        query = '''
            SELECT r.tipo_restricao, r.elemento, COUNT(*) as ocorrencias
            FROM RESTRICOES r
            INNER JOIN DBAR_results d
              ON r.cen_id = d.cen_id AND r.data_simulacao = d.data_simulacao
            WHERE d.CURTAILMENT_total_result > 0
              AND d.hora_simulacao IN (16,17,18)
            GROUP BY r.tipo_restricao, r.elemento
            ORDER BY ocorrencias DESC
        '''
        try:
            df = pd.read_sql_query(query, conn)
        except Exception as e:
            print(f"Erro ao consultar banco: {e}")
            conn.close()
            return
        conn.close()
    else:
        print("Nenhuma fonte de restrições fornecida (db_path ou csv_path).")
        return

    if df.empty:
        print("Nenhuma restrição encontrada.")
        return

    df = df.sort_values('ocorrencias', ascending=False).head(10)
    rotulos = df.apply(lambda r: f"{r['tipo_restricao']} {r['elemento']}", axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(rotulos, df['ocorrencias'], color='darkorange')
    ax.set_xlabel('Número de ocorrências')
    ax.set_title('Restrições ativas que causaram curtailment (16-18h)')
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    fig.tight_layout()

    path = os.path.join(models_dir, 'restricoes_curtailment.png')
    _save(fig, path, dpi=150)

    df.to_csv(os.path.join(models_dir, 'tabela_restricoes_curtailment.csv'),
              index=False)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Backend Matplotlib: {matplotlib.get_backend()}")
    print(f"MODELS_DIR: {MODELS_DIR}")

    print("=" * 60)
    print("1) Curtailment por barra — Real × Previsto HGB (só GWD)")
    plot_windcurtailment(MODELS_DIR)

    print("\n" + "=" * 60)
    print("2) Operação de bateria por barra — Real × Previsto HGB (só BESS)")
    plot_bess_operation(MODELS_DIR)

    print("\n" + "=" * 60)
    print("3) Resíduos — Curtailment")
    plot_residuos(MODELS_DIR, kind='curtailment')

    print("\n" + "=" * 60)
    print("4) Resíduos — BESS")
    plot_residuos(MODELS_DIR, kind='bess')

    print("\n" + "=" * 60)
    print("5) Distribuição dos erros — Curtailment")
    plot_distribuicao_erro(MODELS_DIR, kind='curtailment')

    print("\n" + "=" * 60)
    print("6) Distribuição dos erros — BESS")
    plot_distribuicao_erro(MODELS_DIR, kind='bess')

    # Se tiver a tabela de restrições, descomente:
    # plot_restricoes_curtailment(MODELS_DIR, db_path=DB_PATH)

    print("\nProcesso concluído.")