"""
power_balance_plots.py
Classe para carregar dados de balanço de potência do banco SQLite
e gerar gráficos horários da operação do sistema (geração, demanda, déficit, curtailment)
para um cenário específico (cen_id), agora com foco em cada barra.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class BarraPowerPlotter:
    """
    Lê a tabela DBAR_results para um dado cen_id e produz,
    para cada barra, um conjunto de gráficos horários:
    - Geração convencional e eólica (empilhadas)
    - Curtailment e déficit
    - Operação da bateria (se houver)
    """

    def __init__(self, db_path='../DATA/output/resultados_PL_acoplado.db', cen_id=None):
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
        self.df_barras = None
        self.barras_com_bateria = set()
        self._load_data()

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

    def _load_data(self):
        """Carrega todos os dados da tabela DBAR_results para o cen_id especificado."""
        if self.cen_id is None:
            cenarios = self._list_cenarios()
            if not cenarios:
                print("❌ Nenhum cenário encontrado no banco.")
                return
            self.cen_id = cenarios[0]
            print(f"ℹ️  Nenhum cen_id informado. Usando o primeiro disponível: {self.cen_id}")

        print(f"📊 Carregando dados da tabela DBAR_results para o cenário {self.cen_id}...")

        try:
            conn = sqlite3.connect(self.db_path)
            query = f"""
                SELECT 
                    data_simulacao,
                    hora_simulacao,
                    BAR_id,
                    BAR_tipo,
                    PLOAD_cenario,
                    PGER_CONV_total_result,
                    PGWIND_total_result,
                    CURTAILMENT_total_result,
                    BESS_operation_result,
                    BESS_soc_atual_result,
                    PLOSS_result,
                    PDEF_result
                FROM DBAR_results
                WHERE cen_id = '{self.cen_id}'
                ORDER BY data_simulacao, hora_simulacao, BAR_id
            """
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                print("❌ Nenhum dado encontrado na tabela DBAR_results para este cenário.")
                return

            # Identificar barras com bateria (aquelas que têm BESS_operation_result não nulo em alguma hora)
            # Como o banco pode ter NULL ou 0, vamos considerar que se algum valor for diferente de zero, tem bateria.
            bess_active = df.groupby('BAR_id')['BESS_operation_result'].apply(lambda x: (x != 0).any())
            self.barras_com_bateria = set(bess_active[bess_active].index)

            # Converter colunas para numérico (algumas podem vir como string)
            numeric_cols = ['PGER_CONV_total_result', 'PGWIND_total_result', 'CURTAILMENT_total_result',
                            'BESS_operation_result', 'BESS_soc_atual_result', 'PLOSS_result', 'PDEF_result']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            self.df_barras = df
            print(f"✅ Dados carregados: {len(df)} registros, {df['BAR_id'].nunique()} barras.")
            print(f"   Barras com bateria: {sorted(self.barras_com_bateria)}")

        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            self.df_barras = None

    def plot_all_barras(self, save_fig=False, output_dir='.'):
        """
        Gera uma figura para cada barra, com subplots:
        - Geração por tipo (convencional + eólica) empilhada
        - Curtailment e Déficit
        - Operação da bateria (se houver)
        """
        if self.df_barras is None or self.df_barras.empty:
            print("❌ Sem dados para plotar.")
            return

        # Dimensões desejadas em centímetros (convertidas para polegadas)
        largura_cm = 20
        altura_cm = 15
        largura_in = largura_cm / 2.54
        altura_in = altura_cm / 2.54

        for bar_id in sorted(self.df_barras['BAR_id'].unique()):
            df_bar = self.df_barras[self.df_barras['BAR_id'] == bar_id].copy()
            df_bar.sort_values(['data_simulacao', 'hora_simulacao'], inplace=True)

            if (df_bar['PGER_CONV_total_result'].sum() == 0 and
                df_bar['PGWIND_total_result'].sum() == 0 and
                df_bar['BESS_soc_atual_result'].sum() == 0):
                continue

            # Criar rótulos para o eixo X
            x_labels = [f"D{int(row.data_simulacao)} H{int(row.hora_simulacao)}" for _, row in df_bar.iterrows()]
            x_ticks = np.arange(len(df_bar))
            horas = len(df_bar)

            # Determinar número de subplots
            tem_bateria = bar_id in self.barras_com_bateria
            n_subplots = 3 if tem_bateria else 2

            # Criar figura com o tamanho desejado (5cm x 10cm)
            fig, axes = plt.subplots(n_subplots, 1, figsize=(largura_in, altura_in), sharex=True)

            # Garantir que axes seja sempre uma lista (para facilitar o loop)
            if n_subplots == 1:
                axes = [axes]
            # Para 2 subplots, axes já é uma lista de 2 elementos

            # 1. Gráfico de geração empilhada
            ax1 = axes[0]
            convencional = df_bar['PGER_CONV_total_result'].values
            eolica = df_bar['PGWIND_total_result'].values
            ax1.bar(x_ticks, convencional, label='Convencional', alpha=0.7, color='steelblue')
            ax1.bar(x_ticks, eolica, bottom=convencional, label='Eólica', alpha=0.7, color='lightgreen')
            ax1.set_ylabel('Potência (pu)')
            ax1.set_title('Geração por Tipo')
            ax1.legend(loc='upper right', fontsize='small')
            ax1.grid(True, alpha=0.3)

            # 2. Curtailment e Déficit
            ax2 = axes[1]
            width = 0.35
            curtail = df_bar['CURTAILMENT_total_result'].values
            deficit = df_bar['PDEF_result'].values
            ax2.bar(x_ticks - width/2, curtail, width, label='Curtailment', alpha=0.7, color='orange')
            ax2.bar(x_ticks + width/2, deficit, width, label='Déficit', alpha=0.7, color='red')
            ax2.set_ylabel('Potência (pu)')
            ax2.set_title('Curtailment e Déficit')
            ax2.legend(loc='upper right', fontsize='small')
            ax2.grid(True, alpha=0.3)

            # 3. Bateria (se houver)
            if tem_bateria:
                ax3 = axes[2]
                potencia = np.array(df_bar['BESS_operation_result'].values)
                soc = np.array(df_bar['BESS_soc_atual_result'].values)

                max_val = max(potencia.max(), soc.max())
                ax3.bar(x_ticks, potencia, label='Operação (pu)', alpha=0.7, color='purple')
                ax3.set_ylabel('Potência (pu)', color='purple')
                ax3.tick_params(axis='y', labelcolor='purple')

                # Criar eixo y secundário para o SOC
                #ax3b = ax3.twinx()
                ax3.plot(x_ticks, soc, label='SOC', color='darkgreen', marker='o', markersize=2, linewidth=1)
                ax3.set_ylabel('Potência (pu)', color='black')
                ax3.tick_params(axis='y', labelcolor='black')
                ax3.set_ylim(min(potencia.min()*1.5,0), max(soc.max()*1.5, 1))  # Ajuste conforme necessário

                # Combinar legendas (usando linhas de ambos os eixos)
                lines1, labels1 = ax3.get_legend_handles_labels()
                #lines2, labels2 = ax3.get_legend_handles_labels()
                ax3.legend(lines1 , labels1, loc='upper right', fontsize='small')
                ax3.grid(True, alpha=0.3)

                ax3.set_title('Operação da Bateria')

            # Configurar eixo X (apenas no último subplot)
            axes[-1].set_xlabel('Período (Dia Hora)')
            axes[-1].set_xticks(x_ticks[::max(1, horas//24)])  # Mostra no máximo 8 ticks
            axes[-1].set_xticklabels(x_labels[::max(1, horas//24)], rotation=45, ha='right', fontsize='small')

            # Título geral
            titulo = f'Barra {bar_id} - IEEE 14 Barras'
            plt.suptitle(titulo, fontsize=10, fontweight='bold')

            # Ajustar layout para caber no tamanho pequeno
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Deixa espaço para o suptitle

            if save_fig:
                import os
                os.makedirs(output_dir, exist_ok=True)
                fname = os.path.join(output_dir, f'barra_{bar_id}_{self.cen_id}.png')
                # Para salvar com a resolução adequada, use dpi para controlar o tamanho em pixels
                plt.savefig(fname, dpi=150)  # 150 dpi resulta em 150*largura_in x 150*altura_in pixels
                print(f"✅ Figura salva: {fname}")

            plt.show()

    def plot_resumo_global(self, save_fig=False, output_dir='.'):
        """
        (Opcional) Gera um gráfico resumo global com totais por hora:
        - Geração total (convencional + eólica)
        - Curtailment total
        - Déficit total
        - Carga total
        """
        if self.df_barras is None or self.df_barras.empty:
            print("❌ Sem dados para plotar.")
            return

        # Agregar por hora
        df_hora = self.df_barras.groupby(['data_simulacao', 'hora_simulacao']).agg({
            'PGER_total_result': 'sum',
            'PGWIND_total_result': 'sum',
            'PCURTAILMENT_total_result': 'sum',
            'PDEF_result': 'sum',
            'PLOAD_cenario': 'sum'
        }).reset_index()
        df_hora.sort_values(['data_simulacao', 'hora_simulacao'], inplace=True)

        x_labels = [f"D{int(row.data_simulacao)} H{int(row.hora_simulacao)}" for _, row in df_hora.iterrows()]
        x_ticks = np.arange(len(df_hora))

        fig, ax = plt.subplots(figsize=(14, 6))
        width = 0.2
        ax.bar(x_ticks - 1.5*width, df_hora['PGER_total_result'], width, label='Convencional', color='steelblue')
        ax.bar(x_ticks - 0.5*width, df_hora['PGWIND_total_result'], width, label='Eólica', color='lightgreen')
        ax.bar(x_ticks + 0.5*width, df_hora['PCURTAILMENT_total_result'], width, label='Curtailment', color='orange')
        ax.bar(x_ticks + 1.5*width, df_hora['PDEF_result'], width, label='Déficit', color='red')
        ax.plot(x_ticks, df_hora['PLOAD_cenario'], label='Carga', color='black', marker='o', linewidth=2)

        ax.set_xlabel('Dia e Hora')
        ax.set_ylabel('Potência (MW)')
        ax.set_title(f'Resumo Global por Hora - Cenário {self.cen_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x_ticks[::max(1, len(x_ticks)//12)])
        ax.set_xticklabels(x_labels[::max(1, len(x_ticks)//12)], rotation=45, ha='right')

        plt.tight_layout()
        if save_fig:
            import os
            os.makedirs(output_dir, exist_ok=True)
            fname = os.path.join(output_dir, f'resumo_global_{self.cen_id}.png')
            plt.savefig(fname, dpi=150)
            print(f"✅ Figura salva: {fname}")
        plt.show()


if __name__ == '__main__':
    # Exemplo de uso
    plotter = BarraPowerPlotter(db_path='DATA/output/resultados_PL_acoplado.db') 
    # Para um cenário específico: plotter = BarraPowerPlotter(cen_id='20250213143000')
    plotter.plot_all_barras(save_fig=False)
    # plotter.plot_resumo_global(save_fig=False)