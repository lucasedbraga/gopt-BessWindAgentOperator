import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, pearsonr
import sqlite3

class RNADataAnalyzer:
    """
    Classe para análise exploratória dos dados da tabela DBAR_results,
    com opção de incluir também medições de linhas (DLIN_results).
    Focada na avaliação da adequação para treinamento de redes neurais.
    """

    def __init__(self, db_path, table_name='DBAR_results',
                 load_linhas=False, linhas_especificas=None):
        """
        Inicializa o analisador com o caminho do banco SQLite e o nome da tabela.

        Parâmetros:
        - db_path: caminho para o arquivo .db
        - table_name: nome da tabela a ser analisada (padrão 'DBAR_results')
        - load_linhas: se True, carrega também os dados de linhas e mescla
        - linhas_especificas: lista de strings no formato "4-7", etc.
                              Se None e load_linhas=True, usa padrão ["4-7","4-9","5-6"]
        """
        self.db_path = db_path
        self.table_name = table_name
        self.load_linhas = load_linhas
        self.linhas_especificas = linhas_especificas or ["4-7","4-9","5-6"]
        self.df = None
        self.load_data()

    def load_data(self):
        """
        Carrega os dados da tabela principal (DBAR_results) e, se solicitado,
        também os dados de linha (DLIN_results) e realiza o merge.
        """
        # 1. Carregar dados principais
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT * FROM {self.table_name}"
        self.df = pd.read_sql_query(query, conn)
        print(f"Dados principais carregados. Registros: {len(self.df)}")
        
        # Garantir que a coluna hora_simulacao seja inteira
        if 'hora_simulacao' in self.df.columns:
            self.df['hora_simulacao'] = self.df['hora_simulacao'].astype(int)
        
        # 2. Se solicitado, carregar e mesclar dados de linha
        if self.load_linhas:
            df_linhas = self._load_dlin_measurements(conn)
            if not df_linhas.empty:
                # Merge com os dados principais
                self.df = self.df.merge(df_linhas, on=['cen_id', 'data_simulacao'], how='left')
                print(f"Dados mesclados com linhas. Novo shape: {self.df.shape}")
                # Preencher NaNs com 0 (ou outra estratégia, conforme necessidade)
                cols_linhas = [c for c in df_linhas.columns if c not in ['cen_id', 'data_simulacao']]
                self.df[cols_linhas] = self.df[cols_linhas].fillna(0)
            else:
                print("Nenhum dado de linha encontrado. Continuando sem eles.")
        conn.close()

    def _load_dlin_measurements(self, conn):
        """
        Carrega medições das linhas especificadas a partir da tabela DLIN_results,
        considerando os dois sentidos e selecionando a primeira ocorrência por cenário e data.

        Parâmetros:
        - conn: conexão SQLite já aberta

        Retorna:
        - DataFrame com colunas: cen_id, data_simulacao, e colunas de medição
          (ex: FLUX_result_4-7, LIN_usage_result_4-7, etc.)
        """
        # Construir condição WHERE para os pares (de_barra, para_barra) nos dois sentidos
        condicoes_linha = []
        for linha in self.linhas_especificas:
            barra1, barra2 = linha.split('-')
            condicoes_linha.append(f"(de_barra = '{barra1}' AND para_barra = '{barra2}')")
            condicoes_linha.append(f"(de_barra = '{barra2}' AND para_barra = '{barra1}')")
        where_linhas = " OR ".join(condicoes_linha)
        
        query_med = f"""
            SELECT cen_id, data_simulacao, de_barra, para_barra,
                   FLUX_result, LIN_usage_result
            FROM DLIN_results
            WHERE {where_linhas}
        """
        try:
            df_med = pd.read_sql_query(query_med, conn)
        except Exception as e:
            print(f"Erro ao carregar DLIN_results: {e}")
            return pd.DataFrame()
        
        if df_med.empty:
            return df_med
        
        # Criar coluna linha_normalizada (menor-maior)
        df_med['linha'] = df_med.apply(
            lambda row: f"{min(row['de_barra'], row['para_barra'])}-{max(row['de_barra'], row['para_barra'])}",
            axis=1
        )
        # Filtrar apenas as linhas desejadas
        df_med = df_med[df_med['linha'].isin(self.linhas_especificas)]
        if df_med.empty:
            return df_med
        
        # Selecionar a primeira ocorrência por (cen_id, data_simulacao, linha)
        # Se existir coluna 'id', usá-la; caso contrário, usar a ordem do DataFrame
        if 'id' in df_med.columns:
            ordem_col = 'id'
        else:
            df_med = df_med.reset_index(drop=True)
            df_med['_ordem'] = df_med.index
            ordem_col = '_ordem'
        
        df_med = df_med.sort_values(ordem_col).groupby(
            ['cen_id', 'data_simulacao', 'linha'], as_index=False
        ).first()
        
        # Criar colunas pivotadas: FLUX_result_4-7, LIN_usage_result_4-7
        df_med['LIN_usage_result'] = df_med['LIN_usage_result'] / 100.0
        
        # Derreter para pivotar
        df_melt = pd.melt(
            df_med,
            id_vars=['cen_id', 'data_simulacao', 'linha'],
            value_vars=['FLUX_result', 'LIN_usage_result'],
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
        
        return df_pivot

    # ---------- Métodos originais (adaptados para funcionar com o df expandido) ----------
    # Eles permanecem inalterados, pois operam sobre self.df, que agora contém também as colunas das linhas.
    # Se desejar, pode-se adicionar variáveis específicas nos parâmetros.

    def get_hourly_stats(self, variables=None):
        """
        Calcula estatísticas descritivas por hora_simulacao.

        Parâmetros:
        - variables: lista de colunas a analisar (padrão: as cinco variáveis principais)
        """
        if variables is None:
            variables = ['PLOAD_cenario', 'PGWIND_disponivel_cenario',
                         'BESS_init_cenario', 'CURTAILMENT_total_result',
                         'BESS_operation_result']
        # Verificar existência das colunas
        missing = [v for v in variables if v not in self.df.columns]
        if missing:
            raise ValueError(f"Colunas não encontradas: {missing}")

        # Agrupar por hora e calcular estatísticas
        stats = self.df.groupby('hora_simulacao')[variables].agg([
            'mean', 'std', 'min', 'max', 'count'
        ])
        return stats.round(4)

    def plot_hourly_profiles(self, variables=None, save_path=None):
        """
        Plota o perfil horário médio (linha com banda de desvio padrão) para cada variável.
        """
        if variables is None:
            variables = ['PLOAD_cenario', 'PGWIND_disponivel_cenario',
                         'BESS_init_cenario', 'CURTAILMENT_total_result',
                         'BESS_operation_result']

        n_vars = len(variables)
        fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3 * n_vars), sharex=True)
        if n_vars == 1:
            axes = [axes]

        for ax, var in zip(axes, variables):
            # Calcular média e desvio por hora
            mean_vals = self.df.groupby('hora_simulacao')[var].mean()
            std_vals = self.df.groupby('hora_simulacao')[var].std()

            # Ajuste para evitar limites inferiores negativos (exceto para BESS_operation_result)
            if var != "BESS_operation_result":
                lower = (mean_vals - std_vals).clip(lower=0)
            else:
                lower = mean_vals - std_vals
            upper = mean_vals + std_vals

            ax.plot(mean_vals.index, mean_vals.values, marker='o', linestyle='-',
                    color='steelblue', label='Média')
            ax.fill_between(mean_vals.index, lower, upper,
                            alpha=0.2, color='steelblue', label='±1 desvio')
            ax.set_ylabel(var.replace('_', ' ').title())
            ax.grid(alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel('Hora do dia')
        plt.suptitle('Perfil horário médio das variáveis', fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def get_hourly_stats_by_bar(self, variables=None, bar_ids=None):
        """
        Calcula estatísticas descritivas por BAR_id e hora_simulacao.

        Parâmetros:
        - variables: lista de colunas a analisar (padrão: as cinco variáveis principais)
        - bar_ids: lista de IDs de barras a considerar (None = todas)
        """
        if variables is None:
            variables = ['PLOAD_cenario', 'PGWIND_disponivel_cenario',
                         'BESS_init_cenario', 'CURTAILMENT_total_result',
                         'BESS_operation_result']
        missing = [v for v in variables if v not in self.df.columns]
        if missing:
            raise ValueError(f"Colunas não encontradas: {missing}")

        if 'BAR_id' not in self.df.columns:
            raise ValueError("Coluna 'BAR_id' não encontrada nos dados. Não é possível realizar análise por barra.")

        if bar_ids is not None:
            df_filtered = self.df[self.df['BAR_id'].isin(bar_ids)].copy()
        else:
            df_filtered = self.df.copy()

        stats = df_filtered.groupby(['BAR_id', 'hora_simulacao'])[variables].agg([
            'mean', 'std', 'min', 'max', 'count'
        ])
        return stats.round(4)

    def plot_hourly_profiles_by_bar(self, variables=None, bar_ids=None, save_path=None):
        """
        Plota o perfil horário médio para cada variável, separadamente para cada barra.
        """
        if variables is None:
            variables = ['PLOAD_cenario', 'PGWIND_disponivel_cenario',
                         'BESS_init_cenario', 'CURTAILMENT_total_result',
                         'BESS_operation_result']

        if 'BAR_id' not in self.df.columns:
            raise ValueError("Coluna 'BAR_id' não encontrada.")

        if bar_ids is None:
            bar_ids = self.df['BAR_id'].unique()
        else:
            bar_ids = [bid for bid in bar_ids if bid in self.df['BAR_id'].unique()]
            if not bar_ids:
                raise ValueError("Nenhuma barra válida fornecida.")

        n_bars = len(bar_ids)
        n_vars = len(variables)

        fig, axes = plt.subplots(n_vars, n_bars, figsize=(4 * n_bars, 3 * n_vars), sharex=True, sharey='row')
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        if n_bars == 1:
            axes = axes.reshape(-1, 1)

        for i, var in enumerate(variables):
            for j, bid in enumerate(bar_ids):
                ax = axes[i, j]
                df_bar = self.df[self.df['BAR_id'] == bid]
                if df_bar.empty:
                    ax.text(0.5, 0.5, f'Sem dados\npara barra {bid}', ha='center', va='center')
                    ax.set_title(f'Barra {bid}')
                    continue

                mean_vals = df_bar.groupby('hora_simulacao')[var].mean()
                std_vals = df_bar.groupby('hora_simulacao')[var].std()

                if var != "BESS_operation_result":
                    lower = (mean_vals - std_vals).clip(lower=0)
                else:
                    lower = mean_vals - std_vals
                upper = mean_vals + std_vals

                ax.plot(mean_vals.index, mean_vals.values, marker='o', linestyle='-',
                        color='steelblue', label='Média')
                ax.fill_between(mean_vals.index, lower, upper,
                                alpha=0.2, color='steelblue', label='±1 desvio')
                ax.set_title(f'Barra {bid}')
                ax.grid(alpha=0.3)
                if i == 0:
                    ax.legend(loc='upper right', fontsize='small')
                if i == n_vars - 1:
                    ax.set_xlabel('Hora do dia')
                if j == 0:
                    ax.set_ylabel(var.replace('_', ' ').title())

        plt.suptitle('Perfil horário médio por barra', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def get_hourly_stats_by_bar(self, variables=None, bar_ids=None):
        """
        Calcula estatísticas descritivas por BAR_id e hora_simulacao.

        Parâmetros:
        - variables: lista de colunas a analisar (padrão: as cinco variáveis principais)
        - bar_ids: lista de IDs de barras a considerar (None = todas)

        Retorna um DataFrame com MultiIndex (BAR_id, hora_simulacao) e as estatísticas.
        """
        if variables is None:
            variables = ['PLOAD_cenario', 'PGWIND_disponivel_cenario',
                         'BESS_init_cenario', 'CURTAILMENT_total_result',
                         'BESS_operation_result']
        # Verificar existência das colunas
        missing = [v for v in variables if v not in self.df.columns]
        if missing:
            raise ValueError(f"Colunas não encontradas: {missing}")

        # Verificar se BAR_id existe
        if 'BAR_id' not in self.df.columns:
            raise ValueError("Coluna 'BAR_id' não encontrada nos dados. Não é possível realizar análise por barra.")

        # Filtrar por bar_ids, se fornecido
        if bar_ids is not None:
            df_filtered = self.df[self.df['BAR_id'].isin(bar_ids)].copy()
        else:
            df_filtered = self.df.copy()

        # Agrupar por BAR_id e hora_simulacao
        stats = df_filtered.groupby(['BAR_id', 'hora_simulacao'])[variables].agg([
            'mean', 'std', 'min', 'max', 'count'
        ])
        return stats.round(4)

    def plot_hourly_profiles_by_bar(self, variables=None, bar_ids=None, save_path=None):
        """
        Plota o perfil horário médio (linha com banda de desvio padrão) para cada variável,
        separadamente para cada barra ou comparando barras na mesma figura.

        Parâmetros:
        - variables: lista de colunas a analisar (padrão: as cinco variáveis principais)
        - bar_ids: lista de IDs de barras a considerar (None = todas)
        - save_path: caminho para salvar a figura (opcional)
        """
        if variables is None:
            variables = ['PLOAD_cenario', 'PGWIND_disponivel_cenario',
                         'BESS_init_cenario', 'CURTAILMENT_total_result',
                         'BESS_operation_result']

        # Verificar existência de BAR_id
        if 'BAR_id' not in self.df.columns:
            raise ValueError("Coluna 'BAR_id' não encontrada nos dados. Não é possível realizar análise por barra.")

        # Obter lista de barras a serem plotadas
        if bar_ids is None:
            bar_ids = self.df['BAR_id'].unique()
        else:
            bar_ids = [bid for bid in bar_ids if bid in self.df['BAR_id'].unique()]
            if not bar_ids:
                raise ValueError("Nenhuma barra válida fornecida.")

        n_bars = len(bar_ids)
        n_vars = len(variables)

        # Criar subplots: cada linha é uma variável, cada coluna é uma barra
        fig, axes = plt.subplots(n_vars, n_bars, figsize=(4 * n_bars, 3 * n_vars), sharex=True, sharey='row')
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        if n_bars == 1:
            axes = axes.reshape(-1, 1)

        for i, var in enumerate(variables):
            for j, bid in enumerate(bar_ids):
                ax = axes[i, j]
                # Filtrar dados da barra atual
                df_bar = self.df[self.df['BAR_id'] == bid]
                if df_bar.empty:
                    ax.text(0.5, 0.5, f'Sem dados\npara barra {bid}', ha='center', va='center')
                    ax.set_title(f'Barra {bid}')
                    continue

                # Calcular média e desvio por hora
                mean_vals = df_bar.groupby('hora_simulacao')[var].mean()
                std_vals = df_bar.groupby('hora_simulacao')[var].std()

                # Ajuste para evitar limites inferiores negativos (exceto para BESS_operation_result)
                if var != "BESS_operation_result":
                    lower = (mean_vals - std_vals).clip(lower=0)
                else:
                    lower = mean_vals - std_vals
                upper = mean_vals + std_vals

                # Plotar linha e banda
                ax.plot(mean_vals.index, mean_vals.values, marker='o', linestyle='-',
                        color='steelblue', label='Média')
                ax.fill_between(mean_vals.index, lower, upper,
                                alpha=0.2, color='steelblue', label='±1 desvio')
                ax.set_title(f'Barra {bid}')
                ax.grid(alpha=0.3)
                if i == 0:
                    ax.legend(loc='upper right', fontsize='small')
                if i == n_vars - 1:
                    ax.set_xlabel('Hora do dia')
                if j == 0:
                    ax.set_ylabel(var.replace('_', ' ').title())

        plt.suptitle('Perfil horário médio por barra', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajusta espaço para o título superior
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def correlation_matrix(self, variables=None, save_path=None):
        """
        Calcula e plota a matriz de correlação entre as variáveis selecionadas.
        """
        if variables is None:
            variables = ['PLOAD_cenario', 'PGWIND_disponivel_cenario',
                         'BESS_init_cenario', 'CURTAILMENT_total_result',
                         'BESS_operation_result']
        # Selecionar apenas as colunas relevantes
        corr_data = self.df[variables].copy()
        # Remover outliers extremos (opcional: usar percentis)
        for col in corr_data.columns:
            q99 = corr_data[col].quantile(0.995)
            corr_data[col] = corr_data[col].clip(upper=q99)

        corr = corr_data.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Matriz de correlação entre variáveis')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
        return corr

    def descriptive_statistics(self, variables=None):
        """
        Retorna estatísticas descritivas básicas (média, desvio, skew, kurtosis).
        """
        if variables is None:
            variables = ['PLOAD_cenario', 'PGWIND_disponivel_cenario',
                         'BESS_init_cenario', 'CURTAILMENT_total_result',
                         'BESS_operation_result']
        desc = self.df[variables].describe(percentiles=[0.25, 0.5, 0.75]).T
        # Adicionar skewness e kurtosis
        for var in variables:
            desc.loc[var, 'skew'] = skew(self.df[var].dropna())
            desc.loc[var, 'kurtosis'] = kurtosis(self.df[var].dropna())
        return desc

    def check_missing_values(self):
        """
        Verifica valores nulos por coluna.
        """
        missing = self.df.isnull().sum()
        print("Valores nulos por coluna:")
        print(missing[missing > 0])
        return missing

    def overview(self):
        """
        Exibe uma visão geral dos dados.
        """
        print(f"Total de registros: {len(self.df)}")
        print(f"Colunas disponíveis: {list(self.df.columns)}")
        print("\nPrimeiras 5 linhas:")
        print(self.df.head())
        print("\nTipos de dados:")
        print(self.df.dtypes)
        self.check_missing_values()

    def plot_raw_samples_by_bar(self, variables=None, bar_ids=None, 
                                x_col='hora_simulacao', alpha=0.5, 
                                show_mean_std=True, save_dir=None):
        """
        Para cada barra, plota os valores brutos das variáveis selecionadas ao longo do tempo.
        
        Parâmetros:
        - variables: lista de colunas a analisar (padrão: as cinco principais)
        - bar_ids: lista de IDs de barras a considerar (None = todas)
        - x_col: coluna a ser usada como eixo X (padrão: 'hora_simulacao')
        - alpha: transparência dos pontos (0 a 1)
        - show_mean_std: se True, sobrepõe a média e o desvio padrão por hora
        - save_dir: diretório para salvar as figuras (se None, exibe na tela)
        """
        if variables is None:
            variables = ['PLOAD_cenario', 'PGER_CONV_total_result', 'PGWIND_disponivel_cenario',
                        'CURTAILMENT_total_result',
                      ]
        
        # Verificar existência das colunas
        missing = [v for v in variables if v not in self.df.columns]
        if missing:
            raise ValueError(f"Colunas não encontradas: {missing}")
        
        # Verificar existência de BAR_id
        if 'BAR_id' not in self.df.columns:
            raise ValueError("Coluna 'BAR_id' não encontrada.")
        
        # Determinar barras a processar
        if bar_ids is None:
            bar_ids = self.df['BAR_id'].unique()
        else:
            bar_ids = [bid for bid in bar_ids if bid in self.df['BAR_id'].unique()]
            if not bar_ids:
                raise ValueError("Nenhuma barra válida fornecida.")
        
        # Para cada barra
        for bid in bar_ids:
            df_bar = self.df[self.df['BAR_id'] == bid].copy()
            if df_bar.empty:
                print(f"Barra {bid} sem dados. Pulando.")
                continue
            
            # Criar figura com subplots
            n_vars = len(variables)
            fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars), sharex=True)
            if n_vars == 1:
                axes = [axes]
            
            for i, var in enumerate(variables):
                ax = axes[i]
                # Plotar pontos brutos
                ax.scatter(df_bar[x_col], df_bar[var], alpha=alpha, s=10, 
                        color='steelblue', label='Amostras')
                
                # Se solicitado, calcular média e desvio por hora
                if show_mean_std:
                    mean_vals = df_bar.groupby(x_col)[var].mean()
                    std_vals = df_bar.groupby(x_col)[var].std()
                    # Ajuste para evitar limites inferiores negativos (exceto para BESS_operation_result)
                    if var != "BESS_operation_result":
                        lower = (mean_vals - std_vals).clip(lower=0)
                    else:
                        lower = mean_vals - std_vals
                    upper = mean_vals + std_vals
                    
                    ax.plot(mean_vals.index, mean_vals.values, color='red', 
                            linewidth=2, label='Média')
                    ax.fill_between(mean_vals.index, lower, upper, 
                                    alpha=0.3, color='red', label='±1 desvio')
                
                ax.set_ylabel(var.replace('_', ' ').title())
                ax.grid(alpha=0.3)
                if i == 0:
                    ax.legend(loc='upper right')
            
            axes[-1].set_xlabel(x_col.replace('_', ' ').title())
            plt.suptitle(f'Barra {bid} - Amostras brutas', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Salvar ou mostrar
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'barra_{bid}_raw_samples.png')
                plt.savefig(save_path, dpi=150)
                plt.close(fig)
                print(f"Figura salva: {save_path}")
            else:
                plt.show()

    def plot_samples_at_hour(self, hour, variables=None, bar_ids=None,
                         alpha=0.5, show_mean_std=True, save_dir=None):
        """
        Para cada barra, plota os valores brutos das variáveis selecionadas
        para uma hora específica (ex.: 17h), considerando todas as simulações.

        Parâmetros:
        - hour: inteiro (0-23) representando a hora do dia.
        - variables: lista de colunas a analisar (padrão: as cinco principais)
        - bar_ids: lista de IDs de barras a considerar (None = todas)
        - alpha: transparência dos pontos (0 a 1)
        - show_mean_std: se True, sobrepõe a média e o desvio padrão
        - save_dir: diretório para salvar as figuras (se None, exibe na tela)
        """
        if variables is None:
            variables = ['PLOAD_cenario', 'PGER_CONV_total_result', 'PGWIND_disponivel_cenario',
                        'CURTAILMENT_total_result',
                        ]
        
        # Verificar existência das colunas
        missing = [v for v in variables if v not in self.df.columns]
        if missing:
            raise ValueError(f"Colunas não encontradas: {missing}")
        
        # Verificar existência de BAR_id
        if 'BAR_id' not in self.df.columns:
            raise ValueError("Coluna 'BAR_id' não encontrada.")
        
        # Filtrar os dados pela hora especificada
        df_hour = self.df[self.df['hora_simulacao'] == hour].copy()
        if df_hour.empty:
            raise ValueError(f"Nenhum dado encontrado para a hora {hour}.")
        
        # Determinar barras a processar
        if bar_ids is None:
            bar_ids = df_hour['BAR_id'].unique()
        else:
            bar_ids = [bid for bid in bar_ids if bid in df_hour['BAR_id'].unique()]
            if not bar_ids:
                raise ValueError(f"Nenhuma barra válida fornecida para a hora {hour}.")
        
        # Para cada barra
        for bid in bar_ids:
            df_bar = df_hour[df_hour['BAR_id'] == bid].copy()
            if df_bar.empty:
                print(f"Barra {bid} sem dados para a hora {hour}. Pulando.")
                continue
            
            # Cria uma sequência de números de amostra (1-indexada)
            sample_numbers = np.arange(1, len(df_bar) + 1)
            
            # Criar figura com subplots
            n_vars = len(variables)
            fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars), sharex=False)
            if n_vars == 1:
                axes = [axes]
            
            for i, var in enumerate(variables):
                ax = axes[i]
                # Plotar pontos brutos com números de amostra sequenciais
                if sum(df_bar[var].notna()) < 0.01:
                    continue
                else:
                    ax.plot(sample_numbers, df_bar[var],# alpha=alpha, s=10,
                            color='steelblue', label='Amostras')

                    if show_mean_std:
                        mean_val = df_bar[var].mean()
                        std_val = df_bar[var].std()
                        # Adicionar linha horizontal da média e banda de desvio
                        ax.axhline(mean_val, color='red', linewidth=2, label='Média')
                        ax.fill_between([sample_numbers.min(), sample_numbers.max()],
                                        mean_val - std_val, mean_val + std_val,
                                        alpha=0.3, color='red', label='±1 desvio')
                    
                    ax.set_ylabel(var.replace('_', ' ').title())
                    ax.grid(alpha=0.3)
                    if i == 0:
                        ax.legend(loc='upper right')
            
            axes[-1].set_xlabel('Número da amostra')
            plt.suptitle(f'Barra {bid} - Hora {hour:02d}:00 - Amostras brutas', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Salvar ou mostrar
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'barra_{bid}_h{hour:02d}_raw_samples.png')
                plt.savefig(save_path, dpi=150)
                plt.close(fig)
                print(f"Figura salva: {save_path}")
            else:
                plt.show()

    def feature_target_correlation(self, target, features=None, method='pearson', plot=True, save_path=None):
        """
        Calcula correlação entre features e variável alvo.
        
        Parâmetros:
        - target: nome da coluna alvo
        - features: lista de colunas features (se None, usa todas exceto alvo e colunas não numéricas)
        - method: 'pearson' ou 'spearman'
        - plot: se True, plota barplot das correlações
        - save_path: caminho para salvar a figura
        
        Retorna:
        - DataFrame com correlações ordenadas
        """
        if features is None:
            features = [col for col in self.df.columns if col != target and self.df[col].dtype in ['int64', 'float64']]
        
        corr = self.df[features + [target]].corr(method=method)[target].drop(target).sort_values(ascending=False)
        
        if plot:
            plt.figure(figsize=(10, max(4, len(features)*0.4)))
            colors = ['red' if c < 0 else 'green' for c in corr.values]
            corr.plot(kind='barh', color=colors, edgecolor='black')
            plt.title(f'Correlação ({method}) com {target}')
            plt.xlabel(f'Correlação ({method})')
            plt.grid(alpha=0.3, axis='x')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150)
            plt.show()
        
        return corr

    def assess_predictive_potential(self, target, features=None, test_size=0.2, random_state=42):
        """
        Avalia o potencial preditivo dos dados usando modelos simples (LinearRegression e RandomForest).
        Isso serve como baseline para comparar com a RNA.
        
        Parâmetros:
        - target: nome da coluna alvo
        - features: lista de colunas features (se None, usa todas numéricas exceto target)
        - test_size: proporção de teste
        - random_state: semente para reprodutibilidade
        
        Retorna:
        - dict com métricas (R², RMSE, MAE) para cada modelo
        """
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        import warnings
        warnings.filterwarnings('ignore')
        
        if features is None:
            features = [col for col in self.df.columns if col != target and self.df[col].dtype in ['int64', 'float64']]
        
        X = self.df[features].copy()
        y = self.df[target].copy()
        
        # Remover linhas com NaN
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            raise ValueError("Não há dados completos após remoção de NaN.")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        
        def metrics(y_true, y_pred):
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            return {'R2': r2, 'RMSE': rmse, 'MAE': mae}
        
        results = {
            'LinearRegression': metrics(y_test, y_pred_lr),
            'RandomForest': metrics(y_test, y_pred_rf)
        }
        
        # Imprimir resultados
        print(f"=== Avaliação de potencial preditivo (target: {target}) ===")
        for model, metrics_dict in results.items():
            print(f"{model}: R² = {metrics_dict['R2']:.4f}, RMSE = {metrics_dict['RMSE']:.4f}, MAE = {metrics_dict['MAE']:.4f}")
        
        return results

    def check_outliers(self, variables=None, method='iqr', threshold=1.5, plot=True, save_path=None):
        """
        Detecta outliers nas variáveis usando IQR ou Z-score.
        
        Parâmetros:
        - variables: lista de colunas (se None, usa todas numéricas)
        - method: 'iqr' ou 'zscore'
        - threshold: limite (para IQR: 1.5 típico, para zscore: 3 típico)
        - plot: se True, plota boxplots
        - save_path: caminho para salvar a figura
        
        Retorna:
        - dict com contagem de outliers por variável
        """
        if variables is None:
            variables = [col for col in self.df.columns if self.df[col].dtype in ['int64', 'float64']]
        
        outliers_count = {}
        
        for var in variables:
            data = self.df[var].dropna()
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                outliers = data[(data < lower) | (data > upper)]
            elif method == 'zscore':
                z = np.abs((data - data.mean()) / data.std())
                outliers = data[z > threshold]
            else:
                raise ValueError("Método deve ser 'iqr' ou 'zscore'")
            
            outliers_count[var] = len(outliers)
        
        if plot:
            n_vars = len(variables)
            fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars))
            if n_vars == 1:
                axes = [axes]
            for ax, var in zip(axes, variables):
                self.df.boxplot(column=var, ax=ax)
                ax.set_title(f'{var} - Outliers: {outliers_count[var]}')
                ax.grid(alpha=0.3)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150)
            plt.show()
        
        return outliers_count

    def analyze_variability(self, group_col='hora_simulacao', variables=None, plot=True, save_path=None):
        """
        Analisa a variabilidade intra-grupo (por hora ou por barra) usando coeficiente de variação.
        
        Parâmetros:
        - group_col: coluna para agrupar ('hora_simulacao' ou 'BAR_id')
        - variables: lista de colunas (se None, usa as cinco principais)
        - plot: se True, plota heatmap do CV
        - save_path: caminho para salvar a figura
        
        Retorna:
        - DataFrame com coeficiente de variação (std/mean) por grupo e variável
        """
        if variables is None:
            variables = ['PLOAD_cenario', 'PGWIND_disponivel_cenario',
                         'BESS_init_cenario', 'CURTAILMENT_total_result',
                         'BESS_operation_result']
        
        # Calcular média e desvio por grupo
        grouped_mean = self.df.groupby(group_col)[variables].mean()
        grouped_std = self.df.groupby(group_col)[variables].std()
        
        # Coeficiente de variação (evitar divisão por zero)
        cv = grouped_std / (grouped_mean + 1e-10)
        
        if plot:
            plt.figure(figsize=(12, 6))
            sns.heatmap(cv.T, annot=True, fmt='.2f', cmap='viridis', cbar_kws={'label': 'CV'})
            plt.title(f'Coeficiente de Variação por {group_col}')
            plt.xlabel(group_col)
            plt.ylabel('Variável')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150)
            plt.show()
        
        return cv

    def check_distribution(self, variables=None, bins=50, plot=True, save_path=None):
        """
        Verifica a distribuição das variáveis (histogramas e teste de normalidade).
        
        Parâmetros:
        - variables: lista de colunas (se None, usa todas numéricas)
        - bins: número de bins para histograma
        - plot: se True, plota histogramas
        - save_path: caminho para salvar a figura
        
        Retorna:
        - DataFrame com skewness e kurtosis para cada variável
        """
        if variables is None:
            variables = [col for col in self.df.columns if self.df[col].dtype in ['int64', 'float64']]
        
        results = []
        for var in variables:
            data = self.df[var].dropna()
            skew_val = skew(data)
            kurt_val = kurtosis(data)
            results.append({'variável': var, 'skewness': skew_val, 'kurtosis': kurt_val})
        
        df_stats = pd.DataFrame(results)
        
        if plot:
            n_vars = len(variables)
            fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars))
            if n_vars == 1:
                axes = [axes]
            for ax, var in zip(axes, variables):
                self.df[var].hist(bins=bins, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
                ax.set_title(f'{var} - Skewness: {df_stats[df_stats["variável"]==var]["skewness"].values[0]:.2f}')
                ax.set_xlabel(var)
                ax.set_ylabel('Frequência')
                ax.grid(alpha=0.3)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150)
            plt.show()
        
        return df_stats

    def redundancy_analysis(self, features=None, threshold=0.95, plot=True, save_path=None):
        """
        Identifica variáveis redundantes (alta correlação entre features).
        
        Parâmetros:
        - features: lista de colunas (se None, usa todas numéricas)
        - threshold: limite de correlação para considerar redundante
        - plot: se True, plota matriz de correlação com destaque
        - save_path: caminho para salvar a figura
        
        Retorna:
        - lista de pares de variáveis com correlação acima do threshold
        """
        if features is None:
            features = [col for col in self.df.columns if self.df[col].dtype in ['int64', 'float64']]
        
        corr = self.df[features].corr().abs()
        # Máscara para o triângulo superior, excluindo diagonal
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        high_corr = corr.where(mask).stack()
        high_corr = high_corr[high_corr > threshold]
        
        if plot:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, mask=mask)
            plt.title('Matriz de correlação entre features')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150)
            plt.show()
        
        return high_corr.sort_values(ascending=False)

    def data_quality_report(self, target, features=None, output_dir=None):
        """
        Gera um relatório completo de qualidade dos dados para treinamento de RNA.
        
        Parâmetros:
        - target: nome da coluna alvo
        - features: lista de colunas features (se None, usa todas numéricas exceto target)
        - output_dir: diretório para salvar os gráficos (se None, apenas exibe)
        """
        if features is None:
            features = [col for col in self.df.columns if col != target and self.df[col].dtype in ['int64', 'float64']]
        
        print("="*60)
        print("RELATÓRIO DE QUALIDADE DOS DADOS PARA RNA")
        print("="*60)
        
        # 1. Estatísticas descritivas
        print("\n1. ESTATÍSTICAS DESCRITIVAS")
        print("-"*40)
        desc = self.descriptive_statistics(variables=features+[target])
        print(desc)
        
     
        
        
        print("\n" + "="*60)