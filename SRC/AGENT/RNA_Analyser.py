#!/usr/bin/env python3
"""
graficos_finais.py

Gera:
  1) Gráfico de erro: Real (OPF) × Previsto (RNA) com uma reta de regressão
     para cada hora (16, 17, 18) e a linha identidade.
  2) (Opcional) Gráfico de barras com as principais restrições ativas que
     causaram curtailment (Tabela X).

Requer que os arquivos previsoes_teste.csv existam em
  <MODELS_DIR>/hora_16/ , <MODELS_DIR>/hora_17/ , <MODELS_DIR>/hora_18/
(gerados pelo script de treinamento RNA_especialistas_por_horario_v7.py).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
import sqlite3   # necessário se quiser consultar o banco para restrições

# ==================== CONFIGURAÇÕES ====================
# Pasta raiz onde estão as subpastas hora_16/, hora_17/, hora_18/
# MODELS_DIR_14 = "/home/lucasedbraga/repositorios/ufjf/gopt-BessWindAgentOperator/DATA/output/output_CUR_Oficial_14b/modelos_especialistas_v7"
# MODELS_DIR_118 = "/home/lucasedbraga/repositorios/ufjf/gopt-BessWindAgentOperator/DATA/output/output_CUR_Oficial_118b/modelos_especialistas_v7_TEST"
MODELS_DIR_118 = r"C:\Users\LucasBraga\Documents\repos\ufjf\gopt-BessWindAgentOperator\DATA\output\modelos_especialistas_v7"
# Caminho do banco de dados original (usado se você quiser extrair restrições do SQLite)
DB_PATH = "DATA/output/RNA_DATA_ACOPF.db"          # ajuste se necessário

# Caminho de um CSV com as restrições (mais simples, veja instruções no final)
RESTRICOES_CSV = "restricoes_curtailment.csv"            # se existir, usa este arquivo
# ========================================================


def plot_windcurtailment(models_dir, horas=(16, 17, 18),
                         threshold=0.0001, max_points=50):
    """
    Para cada hora, gera uma figura com subplots (um por barra de curtailment).
    Cada subplot compara os valores reais (FPO-CC) e previstos (RNA) daquela barra.
    O título do subplot identifica a barra (ex.: BAR3) e a hora.
    """
    for hora in horas:
        csv_path = os.path.join(models_dir, f"hora_{hora:02d}", "previsoes_teste.csv")
        if not os.path.exists(csv_path):
            print(f"Arquivo {csv_path} não encontrado. Pulando hora {hora}.")
            continue

        df = pd.read_csv(csv_path)

        # Identifica todas as colunas reais (targets)
        col_reais = [c for c in df.columns if '_previsto' not in c]
        # Monta os pares (real, previsto) que realmente existem
        pares = [(real, real + '_previsto') for real in col_reais if real + '_previsto' in df.columns]

        if not pares:
            print(f"Nenhum par real/previsto encontrado para hora {hora}.")
            continue

        n_barras = len(pares)
        ncols = 2
        nrows = -(-n_barras // ncols)  # ceil division

        # # Cria figura com altura proporcional ao número de linhas de subplots
        # fig, axes = plt.subplots(3, 1, figsize=(4.5, 10),
        #                          squeeze=False,   # garante que axes seja 2D
        #                          gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
        # fig.subplots_adjust(left=0.15, right=0.985, top=0.95, bottom=0.05,
        #             hspace=0.2, wspace=0.2)   # ajuste hspace/vspace conforme necessário
        # # Achata o array de eixos para fácil iteração
        # axes = axes.flatten()
        rng = np.random.default_rng(seed=42)  # reprodutibilidade
        indices = rng.choice(len(df), size=200, replace=True)
        for idx, (real_col, prev_col) in enumerate(pares):
            #ax = axes[idx]

            # Extrai os valores desta barra
            real = df[real_col].values
            pred = df[prev_col].values

            # # Filtra onde há curtailment significativo
            # mask = real > threshold
            # real_f = real[mask]
            # pred_f = pred[mask]

            real_f = real[indices]
            pred_f = pred[indices]
            # # Ordena pelos valores reais para melhor visualização
            # order = np.argsort(real_f)
            # real_f = real_f[order]
            # pred_f = pred_f[order]
            # Plota as duas curvas
            plt.figure(figsize=(6, 4.5))  # tamanho compacto
            plt.plot(real_f, linestyle='-', linewidth=1.2, alpha=0.9, label='FPO-CC')
            plt.plot(pred_f, linestyle='--', linewidth=1, alpha=0.7, label='RNA', color='red')

            # Título com o nome da barra
            bar_name = real_col.split('_')[-1]  # pega 'BAR3', 'BAR14' etc.
            plt.title(f'WindCurtailment {bar_name} – Hora {hora:02d}',  fontweight='bold')
            plt.xlabel('Índice da amostra', fontweight='bold')
            plt.ylabel('pu (MW)', fontweight='bold')
            plt.tick_params(axis='both', labelsize=8, width=1.0)
            plt.legend(frameon=False)
            plt.grid(alpha=0.3, linewidth=0.8)
            for spine in plt.gca().spines.values():
                spine.set_linewidth(0.5)

        # # Remove subplots vazios (se houver)
        # for j in range(idx + 1, len(axes)):
        #     axes[j].set_visible(False)


        # # Título geral da figura
        # fig.suptitle(f'Sistema IEEE 14', 
        #              fontsize=14, fontweight='bold',y=0.98)
        
        # Ajuste fino dos espaçamentos (compatível com suptitle)
            # Margens internas mínimas
            plt.subplots_adjust(left=0.15, right=0.95, top=0.93, bottom=0.12)
            # Salvamento com borda extra quase nula
            plot_path = os.path.join(models_dir, 'residuos.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
            plt.show()

            plot_path = os.path.join(models_dir, f'comparacao_barras_hora_{hora:02d}.png')
            #fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Gráfico da hora {hora:02d} (por barra) salvo em: {plot_path}")

def plot_residuos(models_dir, horas=(16, 17, 18)):
    """
    Plota os resíduos (Real - Previsto) ao longo das amostras de teste,
    com uma linha horizontal tracejada no erro médio de cada hora.
    """
        # Cores e estilos para cada hora
    cores = {16: '#1f77b4', 17: '#ff7f0e', 18: '#2ca02c'}

    fig, ax = plt.subplots(figsize=(6, 4.5))  # tamanho compacto

    for hora in horas:
        csv_path = os.path.join(models_dir, f"hora_{hora:02d}", "previsoes_teste.csv")
        if not os.path.exists(csv_path):
            print(f"Arquivo {csv_path} não encontrado. Pulando hora {hora}.")
            continue

        df = pd.read_csv(csv_path)

        # Todas as colunas de curtailment (real e previsto)
        col_real = [c for c in df.columns if '_previsto' not in c]
        col_prev = [c for c in df.columns if '_previsto' in c]

        # Achata para considerar todas as barras
        real = df[col_real].values.ravel()
        pred = df[col_prev].values.ravel()

        real= real[:200]
        pred= pred[:200]

        erro = abs(real) - abs(pred)

        # Linha de resíduos
        cor = cores.get(hora, '#333333')
        ax.plot(range(len(erro)), erro, alpha=0.8, linewidth=1.2,
                label=f'Hora {hora:02d}', color=cor)

        # # Linha do erro médio (tracejada, mesma cor)
        # erro_medio = erro.mean()
        # ax.axhline(y=erro_medio, linestyle='--', linewidth=1.0,
        #            color=cor, alpha=0.6, label=f'Erro médio Hora {hora:02d}')
    # Rótulos e título
    ax.set_xlabel('Amostra de teste', fontweight='bold')
    ax.set_ylabel('Erro Absoluto (pu)', fontweight='bold')
    # ax.set_title('Erro (FPO-CC - RNA) para hora : Sistema IEEE 14', fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, linewidth=0.8)

    # Bordas finas
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Margens internas mínimas
    fig.subplots_adjust(left=0.15, right=0.95, top=0.93, bottom=0.12)
    # Salvamento com borda extra quase nula
    plot_path = os.path.join(models_dir, 'residuos.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.show()
    print(f"Gráfico de resíduos salvo em: {plot_path}")


def plot_restricoes_curtailment(models_dir, db_path=None, csv_path=None):
    """
    Gera o gráfico de barras horizontais com as restrições mais frequentes
    que causaram curtailment eólico.

    Forma de obter os dados:
      - Se csv_path for fornecido e existir, carrega de um CSV com colunas:
            tipo_restricao, elemento, ocorrencias
      - Senão, se db_path for fornecido, tenta fazer a consulta diretamente no
        banco SQLite (requer uma tabela RESTRICOES com colunas:
            cen_id, data_simulacao, tipo_restricao, elemento).
        A consulta filtra os cenários com CURTAILMENT_total_result > 0 nas horas 16-18.
      - Se nenhum for fornecido, emite mensagem de instrução.
    """
    # Tenta carregar do CSV primeiro
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if not {'tipo_restricao', 'elemento', 'ocorrencias'}.issubset(df.columns):
            print("CSV de restrições deve conter colunas: tipo_restricao, elemento, ocorrencias")
            return
    elif db_path and os.path.exists(db_path):
        # Consulta SQL dinâmica
        conn = sqlite3.connect(db_path)
        query = '''
            SELECT r.tipo_restricao, r.elemento, COUNT(*) as ocorrencias
            FROM RESTRICOES r
            INNER JOIN DBAR_results d
              ON r.cen_id = d.cen_id AND r.data_simulacao = d.data_simulacao
            WHERE d.CURTAILMENT_total_result > 0 AND d.hora_simulacao IN (16,17,18)
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
        print("Nenhuma fonte de dados de restrições fornecida.")
        print("Crie um arquivo 'restricoes_curtailment.csv' com as colunas:")
        print("  tipo_restricao, elemento, ocorrencias")
        print("Ou forneça um banco SQLite com a tabela RESTRICOES.")
        return

    if df.empty:
        print("Nenhuma restrição encontrada.")
        return

    # Ordena e seleciona as 10 principais para o gráfico
    df = df.sort_values('ocorrencias', ascending=False).head(10)
    rotulos = df.apply(lambda row: f"{row['tipo_restricao']} {row['elemento']}", axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(rotulos, df['ocorrencias'], color='darkorange')
    ax.set_xlabel('Número de ocorrências')
    ax.set_title('Principais restrições ativas que causaram curtailment (horas 16-18)')
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    fig.tight_layout()

    plot_path = os.path.join(models_dir, 'restricoes_curtailment.png')
    fig.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Gráfico de restrições salvo em: {plot_path}")

    # Salva também a tabela
    tabela_path = os.path.join(models_dir, 'tabela_restricoes_curtailment.csv')
    df.to_csv(tabela_path, index=False)
    print(f"Tabela salva em: {tabela_path}")


if __name__ == '__main__':
    #os.makedirs(MODELS_DIR_14, exist_ok=True)
    os.makedirs(MODELS_DIR_118, exist_ok=True)

    # print("=" * 60)
    # print("1) Gráfico de erro: Real × Previsto com retas por hora")
    plot_windcurtailment(MODELS_DIR_118)

    print("\n" + "=" * 60)
    print("2) Gráfico de resíduos (erro médio por hora)")
    plot_residuos(MODELS_DIR_118)

    print("\n" + "=" * 60)
    print("2) Gráfico de resíduos (erro médio por hora)")
    plot_residuos(MODELS_DIR_118)

    print("\n" + "=" * 60)
    print("3) Tabela/Gráfico de restrições de curtailment")
    # #Tenta usar o CSV se existir; senão tenta o banco SQLite
    # plot_restricoes_curtailment(
    #     MODELS_DIR,
    #     db_path=DB_PATH,
    #     csv_path=RESTRICOES_CSV
    # )

    print("\nProcesso concluído.")