"""
power_balance_plots.py
Classe para carregar dados de balanço de potência do banco SQLite
e gerar gráficos horários da operação do sistema (geração, demanda, déficit, curtailment)
para um cenário específico (cen_id).
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class PowerBalancePlotter:
    """
    Lê as tabelas resultados_opf, detalhes_barras, detalhes_geradores e bess_wind_operacao
    para um dado cen_id e produz gráficos de balanço de potência para todas as horas do cenário.
    """

    def __init__(self, db_path='DATA/output/resultados_PL.db', cen_id=None):
        """
        Parâmetros:
        -----------
        db_path : str
            Caminho para o arquivo do banco SQLite.
        cen_id : str, optional
            Identificador do cenário (formato YYYYMMddhhmmss).
            Se None, lista os cenários disponíveis e usa o primeiro.
        """
        self.db_path = db_path
        self.cen_id = cen_id
        self.df_balance = None
        self.df_geradores = None
        self.df_bateria = None
        self._load_all_data()

    def _list_cenarios(self):
        """Lista todos os cen_id disponíveis na tabela resultados_opf."""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT DISTINCT cen_id FROM resultados_opf ORDER BY cen_id", conn)
            conn.close()
            return df['cen_id'].tolist()
        except Exception as e:
            print(f"❌ Erro ao listar cenários: {e}")
            return []

    def _load_all_data(self):
        """Carrega todos os dados necessários para o cen_id especificado."""
        if self.cen_id is None:
            cenarios = self._list_cenarios()
            if not cenarios:
                print("❌ Nenhum cenário encontrado no banco.")
                return
            self.cen_id = cenarios[0]
            print(f"ℹ️  Nenhum cen_id informado. Usando o primeiro disponível: {self.cen_id}")

        print(f"📊 Carregando dados para o cenário {self.cen_id}...")

        # Carregar dados de balanço (resultados_opf)
        self.df_balance = self._load_balance_data()
        if self.df_balance is None or self.df_balance.empty:
            print("❌ Sem dados de balanço para este cenário.")
            return

        # Carregar dados de geradores (agregados por tipo)
        self.df_geradores = self._load_generator_data()

        # Carregar dados de bateria
        self.df_bateria = self._load_battery_data()

        print(f"✅ Dados carregados: {len(self.df_balance)} horas, {len(self.df_geradores) if self.df_geradores is not None else 0} tipos de geradores, {len(self.df_bateria) if self.df_bateria is not None else 0} registros de bateria.")

    def _load_balance_data(self):
        """Carrega os dados horários da tabela resultados_opf para o cen_id."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"""
                SELECT 
                    id as resultado_id,
                    data_simulacao,
                    hora_simulacao,
                    sucesso,
                    custo_total,
                    deficit_total,
                    curtailment_total,
                    perdas_total,
                    carga_total,
                    eolica_disponivel,
                    eolica_utilizada,
                    fator_vento
                FROM resultados_opf
                WHERE cen_id = '{self.cen_id}'
                ORDER BY data_simulacao, hora_simulacao
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"❌ Erro ao carregar dados de balanço: {e}")
            return None

    def _load_generator_data(self):
        """
        Carrega dados de geração por tipo de gerador para o cen_id.
        Como detalhes_geradores não tem cen_id, fazemos JOIN com resultados_opf.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"""
                SELECT 
                    dg.tipo_gerador,
                    SUM(dg.p_gerada_mw) as geracao_mw,
                    SUM(dg.p_max_mw) as capacidade_mw,
                    SUM(dg.curtailment_mw) as curtailment_mw
                FROM detalhes_geradores dg
                JOIN resultados_opf ro ON dg.resultado_id = ro.id
                WHERE ro.cen_id = '{self.cen_id}'
                GROUP BY dg.tipo_gerador
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"❌ Erro ao carregar dados de geradores: {e}")
            return None

    def _load_battery_data(self):
        """Carrega dados da tabela bess_wind_operacao para o cen_id."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"""
                SELECT 
                    data_simulacao,
                    hora_simulacao,
                    barra_bess,
                    soc_percent,
                    operacao,
                    potencia_bess_mw
                FROM bess_wind_operacao
                WHERE cen_id = '{self.cen_id}'
                ORDER BY data_simulacao, hora_simulacao, barra_bess
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            if df.empty:
                return None
            return df
        except Exception as e:
            print(f"❌ Erro ao carregar dados de bateria: {e}")
            return None

    def plot_operation_panel(self, save_fig=False, output_dir='.'):
        """
        Gera um painel 2x2 com todas as horas do cenário:
        - Custo por hora
        - Curtailment eólico por hora
        - Déficit por hora
        - Operação agregada das baterias (carga/descarga) + SOC médio
        """
        if self.df_balance is None or self.df_balance.empty:
            print("❌ Sem dados de balanço para plotar.")
            return

        df = self.df_balance.copy()
        df.sort_values(by=['data_simulacao', 'hora_simulacao'], inplace=True)

        # Preparar dados de bateria agregados por hora (soma de todas as baterias)
        bateria_disponivel = False
        if self.df_bateria is not None and not self.df_bateria.empty:
            df_bat = self.df_bateria.copy()
            # Agregar por hora: carga total (potência negativa), descarga total (potência positiva), SOC médio
            df_bat_agg = df_bat.groupby(['data_simulacao', 'hora_simulacao']).agg(
                potencia_bess_agg=('potencia_bess_mw', lambda x: x.sum()),  # soma das cargas (torna positiva)
                soc_medio=('soc_percent', 'mean')
            ).reset_index()
            df_bat_agg.sort_values(['data_simulacao', 'hora_simulacao'], inplace=True)
            bateria_disponivel = True
        else:
            print("⚠️  Sem dados de bateria. O quarto subplot será deixado em branco.")

        df['data_simulacao'] = df['data_simulacao'].astype(int)
        df_bat_agg['data_simulacao'] = df_bat_agg['data_simulacao'].astype(int)

        # Mesclar para garantir alinhamento com todas as horas
        df = df.merge(df_bat_agg, on=['data_simulacao', 'hora_simulacao'], how='left', suffixes=('', '_bat')).fillna(0)

        # Criar rótulos para o eixo X combinando dia e hora
        x_labels = [f"D{int(row.data_simulacao)} H{int(row.hora_simulacao)}" for _, row in df.iterrows()]
        x_ticks = np.arange(len(df))
        x_label_global = 'Dia e Hora'

        # Valores para os subplots
        custos = df['custo_total'].values
        curtailments = df['curtailment_total'].values
        deficits = df['deficit_total'].values
        if bateria_disponivel:
            potencia_bess_agg = df['potencia_bess_agg'].values
            soc = df['soc_medio'].values

        # Criar figura com 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Gráfico 1: Custo por hora
        ax1.bar(x_ticks, custos, alpha=0.7, color='steelblue')
        ax1.set_xlabel(x_label_global)
        ax1.set_ylabel('Custo (USD)')
        ax1.set_title('Custo de Operação por Hora')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(x_ticks[::max(1, len(x_ticks)//12)])  # Ajusta número de ticks
        ax1.set_xticklabels(x_labels[::max(1, len(x_ticks)//12)], rotation=45, ha='right')

        # Gráfico 2: Curtailment
        ax2.bar(x_ticks, curtailments, alpha=0.7, color='orange')
        ax2.set_xlabel(x_label_global)
        ax2.set_ylabel('Curtailment (MW)')
        ax2.set_title('Curtailment de Geração Eólica por Hora')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(x_ticks[::max(1, len(x_ticks)//12)])
        ax2.set_xticklabels(x_labels[::max(1, len(x_ticks)//12)], rotation=45, ha='right')

        # Gráfico 3: Déficit
        ax3.bar(x_ticks, deficits, alpha=0.7, color='red')
        ax3.set_xlabel(x_label_global)
        ax3.set_ylabel('Déficit (MW)')
        ax3.set_title('Déficit de Carga por Hora')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(x_ticks[::max(1, len(x_ticks)//12)])
        ax3.set_xticklabels(x_labels[::max(1, len(x_ticks)//12)], rotation=45, ha='right')

        # Gráfico 4: Operação da Bateria
        if bateria_disponivel:
            width = 0.35
            # Subtrair width/2 funciona agora porque x_ticks é numpy array
            ax4.bar(x_ticks + width/2, potencia_bess_agg, width, color='red', alpha=0.7, label='BESS_Agregado (MW)')
            ax4.set_xlabel(x_label_global)
            ax4.set_ylabel('Potência (MW)')
            ax4.set_title('Operação Agregada das Baterias')
            ax4.grid(True, alpha=0.3)
            ax4.set_xticks(x_ticks[::max(1, len(x_ticks)//12)])
            ax4.set_xticklabels(x_labels[::max(1, len(x_ticks)//12)], rotation=45, ha='right')

        else:
            ax4.text(0.5, 0.5, 'Sem dados de bateria', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Operação das Baterias (indisponível)')
            ax4.set_xlabel(x_label_global)
            ax4.set_xticks(x_ticks[::max(1, len(x_ticks)//12)])
            ax4.set_xticklabels(x_labels[::max(1, len(x_ticks)//12)], rotation=45, ha='right')

        plt.suptitle(f'Resumo da Operação - Cenário {self.cen_id}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_fig:
            import os
            os.makedirs(output_dir, exist_ok=True)
            fname = os.path.join(output_dir, f'painel_operacao_{self.cen_id}.png')
            plt.savefig(fname, dpi=150)
            print(f"✅ Figura salva: {fname}")

        plt.show()


if __name__ == '__main__':
    # Exemplo de uso: lista cenários e plota o primeiro ou um específico
    plotter = PowerBalancePlotter()  # Usará o primeiro cenário encontrado
    # Para um cenário específico: plotter = PowerBalancePlotter(cen_id='20250213143000')
    plotter.plot_operation_panel(save_fig=False)