"""
battery_plots.py
Classe para carregar dados de operação de baterias do banco SQLite
a partir das novas tabelas (resultados_opf, DBAR_results) e gerar gráficos de barras horários.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class BatteryPlotter:
    """
    Lê as tabelas DBAR_results e resultados_opf para extrair informações de bateria
    e produz gráficos de barras para cada bateria.
    """

    def __init__(self, db_path='DATA/output/resultados_PL.db', cen_id=None):
        """
        Parâmetros:
        -----------
        db_path : str
            Caminho para o arquivo do banco SQLite.
        cen_id : str, optional
            Se informado, filtra apenas os registros com esse cen_id.
        """
        self.db_path = db_path
        self.cen_id = cen_id
        self.df = self._load_data()
        self.baterias = self.df['barra_bess'].unique() if self.df is not None else []

    def _load_data(self):
        """Carrega os dados de bateria da tabela DBAR_results e adiciona coluna de operação."""
        try:
            conn = sqlite3.connect(self.db_path)
            # Consulta para obter registros de barras que possuem dados de bateria (PBESS_inst_result não nulo)
            query = """
                SELECT 
                    cen_id,
                    data_simulacao,
                    hora_simulacao,
                    BAR_id AS barra_bess,
                    PBESS_inst_result AS potencia_bess_mw,
                    PBESS_soc_result AS soc_percent
                FROM DBAR_results
                WHERE PBESS_inst_result IS NOT NULL OR PBESS_soc_result IS NOT NULL
            """
            if self.cen_id is not None:
                query += f" AND cen_id = '{self.cen_id}'"
            query += " ORDER BY cen_id, data_simulacao, hora_simulacao, BAR_id;"

            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                print("⚠️  Nenhum dado de bateria encontrado nas tabelas.")
                return None

            # Derivar a operação a partir do sinal da potência
            def classificar_operacao(pot):
                if pot > 0.01:
                    return 'charge'
                elif pot < -0.01:
                    return 'discharge'
                else:
                    return 'idle'

            df['operacao'] = df['potencia_bess_mw'].apply(classificar_operacao)

            # Converter SOC para percentual (já deve estar, mas garantir)
            # Se necessário, pode-se assegurar que está entre 0 e 100
            df['soc_percent'] = df['soc_percent'].clip(0, 100)

            return df

        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return None

    def plot_battery(self, barra_bess, save_fig=False, output_dir='.'):
        """
        Gera gráfico de barras para uma bateria específica.
        """
        if self.df is None:
            print("❌ Sem dados para plotar.")
            return

        df_bat = self.df[self.df['barra_bess'] == barra_bess].copy()
        if df_bat.empty:
            print(f"⚠️  Nenhum registro para a bateria na barra {barra_bess}.")
            return

        # Ordenar por hora (dentro de cada dia, se houver múltiplos dias, pode ser necessário concatenar)
        # Para simplificar, vamos considerar apenas as primeiras 24 horas do primeiro dia,
        # mas se houver mais dias, podemos criar um eixo temporal contínuo.
        # O código original limitava a 24 pontos. Vamos manter assim para compatibilidade.
        df_bat = df_bat.head(24)  # Ajuste conforme necessidade

        horas = df_bat['hora_simulacao'].values
        potencia = df_bat['potencia_bess_mw'].values
        soc = df_bat['soc_percent'].values
        operacao = df_bat['operacao'].values

        # Mapeamento de cores (mantido igual)
        cor_map = {
            'discharge': 'red',
            'charge': 'green',
            'idle': 'gray',
            'carregando?': 'red',
            'descarregando?': 'green',
            'parado?': 'gray'
        }
        cores = [cor_map.get(str(op).lower(), 'blue') for op in operacao]

        # Figura e eixo principal (potência)
        fig, ax1 = plt.subplots(figsize=(14, 6))

        # Barras de potência
        bars = ax1.bar(horas, potencia, color=cores, alpha=0.7, label='Potência (MW)')
        ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax1.set_ylabel('Potência (MW)', fontsize=12)
        ax1.set_xlabel('Hora do dia', fontsize=12)
        ax1.set_title(f'Bateria na Barra {barra_bess} - Operação Horária', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(0, 24, 1))
        ax1.grid(True, linestyle=':', alpha=0.6)

        # Eixo secundário para SOC (twinx)
        ax2 = ax1.twinx()
        ax2.plot(horas, soc, color='gray', marker='o', linestyle='-', linewidth=2, markersize=4, label='SOC (%)')
        ax2.set_ylabel('SOC (%)', fontsize=12)
        #ax2.set_ylim(0, 105)

        # Legendas combinadas
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()

        if save_fig:
            import os
            os.makedirs(output_dir, exist_ok=True)
            fname = os.path.join(output_dir, f'bateria_barra_{barra_bess}.png')
            plt.savefig(fname, dpi=150)
            print(f"✅ Figura salva: {fname}")

        plt.show()

    def plot_all_batteries(self, save_fig=False, output_dir='.'):
        """Gera gráficos para todas as baterias presentes no DataFrame."""
        if self.df is None:
            return
        for barra in self.baterias:
            self.plot_battery(barra, save_fig=save_fig, output_dir=output_dir)


if __name__ == '__main__':
    # Exemplo de teste rápido (substitua pelo cen_id desejado)
    plotter = BatteryPlotter(cen_id='20250220120000')  # Exemplo
    plotter.plot_all_batteries()