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
        - (Opcional) Perdas
        """
        if self.df_barras is None or self.df_barras.empty:
            print("❌ Sem dados para plotar.")
            return

        # Agrupar por barra
        for bar_id in sorted(self.df_barras['BAR_id'].unique()):
            df_bar = self.df_barras[self.df_barras['BAR_id'] == bar_id].copy()
            df_bar.sort_values(['data_simulacao', 'hora_simulacao'], inplace=True)

            # Criar rótulos para o eixo X
            x_labels = [f"D{int(row.data_simulacao)} H{int(row.hora_simulacao)}" for _, row in df_bar.iterrows()]
            x_ticks = np.arange(len(df_bar))
            horas = len(df_bar)

            # Determinar número de subplots (3 ou 4 dependendo da bateria)
            tem_bateria = bar_id in self.barras_com_bateria
            if tem_bateria:
                n_subplots = 3
                fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            else:
                n_subplots = 2
                fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
                # Ajustar para que axes seja sempre uma lista
                if n_subplots == 2:
                    axes = [axes[0], axes[1]]

            # 1. Gráfico de geração empilhada
            ax1 = axes[0]
            # Preparar dados: convencional, eólica
            convencional = df_bar['PGER_CONV_total_result'].values
            eolica = df_bar['PGWIND_total_result'].values
            # Se a bateria estiver descarregando (positivo), podemos incluir como parte da geração? 
            # Mas para manter a clareza, não incluiremos no empilhamento. Ficará apenas convencional+eólica.
            ax1.bar(x_ticks, convencional, label='Convencional', alpha=0.7, color='steelblue')
            ax1.bar(x_ticks, eolica, bottom=convencional, label='Eólica', alpha=0.7, color='lightgreen')
            ax1.set_ylabel('Geração (MW)')
            ax1.set_title(f'Barra {bar_id} - Geração por Tipo')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)

            # 2. Curtailment e Déficit
            ax2 = axes[1]
            width = 0.35
            curtail = df_bar['CURTAILMENT_total_result'].values
            deficit = df_bar['PDEF_result'].values
            x_pos = x_ticks
            ax2.bar(x_pos - width/2, curtail, width, label='Curtailment', alpha=0.7, color='orange')
            ax2.bar(x_pos + width/2, deficit, width, label='Déficit', alpha=0.7, color='red')
            ax2.set_ylabel('Potência (MW)')
            ax2.set_title('Curtailment e Déficit')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)

            # 3. Bateria (se houver)
            if tem_bateria:
                ax3 = axes[2]
                potencia = df_bar['BESS_operation_result'].values
                soc = df_bar['BESS_soc_atual_result'].values
                # Plotar potência (barras) e SOC (linha no eixo secundário)
                ax3.bar(x_ticks, potencia, label='Potência (MW)', alpha=0.7, color='purple')
                ax3.set_ylabel('Potência (MW)', color='purple')
                ax3.tick_params(axis='y', labelcolor='purple')
                ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)

                ax3b = ax3.twinx()
                ax3b.plot(x_ticks, soc, label='SOC (%)', color='darkgreen', marker='o', markersize=3, linewidth=1.5)
                ax3b.set_ylabel('SOC (%)', color='darkgreen')
                ax3b.tick_params(axis='y', labelcolor='darkgreen')
                ax3b.set_ylim(0, 1)

                # Legendas combinadas (usar linhas do ax3 e ax3b)
                lines1, labels1 = ax3.get_legend_handles_labels()
                lines2, labels2 = ax3b.get_legend_handles_labels()
                ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

                ax3.set_title('Operação da Bateria')

            # Configurar eixo X para todos os subplots
            for ax in axes:
                ax.set_xlabel('')
                ax.set_xticks(x_ticks[::max(1, horas//12)])
                ax.set_xticklabels(x_labels[::max(1, horas//12)], rotation=45, ha='right')

            # Ajustar título geral
            plt.suptitle(f'Barra {bar_id} - Cenário {self.cen_id}', fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.97])

            if save_fig:
                import os
                os.makedirs(output_dir, exist_ok=True)
                fname = os.path.join(output_dir, f'barra_{bar_id}_{self.cen_id}.png')
                plt.savefig(fname, dpi=150)
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
    plotter = BarraPowerPlotter()  # Usará o primeiro cenário encontrado
    # Para um cenário específico: plotter = BarraPowerPlotter(cen_id='20250213143000')
    plotter.plot_all_barras(save_fig=False)
    # plotter.plot_resumo_global(save_fig=False)