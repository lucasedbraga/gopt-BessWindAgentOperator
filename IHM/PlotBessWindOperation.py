"""
battery_plots.py
Classe para carregar dados de operação de baterias do banco SQLite
e gerar gráficos de barras horários.
Agora com suporte a data_simulacao como inteiro (dia da simulação).
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

class BatteryPlotter:
    """
    Lê a tabela bess_wind_operacao do banco resultados_PL.db
    e produz gráficos de barras para cada bateria.
    """

    def __init__(self, db_path='DATA/output/resultados_PL.db', resultado_id=None, data_base='2026-01-01'):
        """
        Parâmetros:
        -----------
        db_path : str
            Caminho para o arquivo do banco SQLite.
        resultado_id : int, optional
            Se informado, filtra apenas os registros com esse resultado_id.
        data_base : str
            Data de referência para converter os dias em timestamp.
        """
        self.db_path = db_path
        self.resultado_id = resultado_id
        self.data_base = pd.to_datetime(data_base)
        self.df = self._load_data()
        self.baterias = self.df['barra_bess'].unique() if self.df is not None else []

    def _load_data(self):
        """Carrega os dados da tabela bess_wind_operacao e converte para timestamp."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM bess_wind_operacao"
            if self.resultado_id is not None:
                query += f" WHERE resultado_id = {self.resultado_id}"
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                print("⚠️  Nenhum dado encontrado na tabela bess_wind_operacao.")
                return None

            df.sort_values(by=['resultado_id','data_simulacao','hora_simulacao'], inplace=True)
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

        df_bat = df_bat.head(24)  # Garantir que só pegue as primeiras 24 horas (ajustável)
        # Ordenar por timestamp (já deve estar ordenado, mas garantia)
        df_bat.sort_values(by=['resultado_id','data_simulacao','hora_simulacao'], inplace=True)

        # Preparar dados
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
        ax1.set_xticks(range(0, 24, 1))  # ticks de 2 em 2 horas (ajustável)
        ax1.grid(True, linestyle=':', alpha=0.6)

        # Eixo secundário para SOC (twinx)
        ax2 = ax1.twinx()
        ax2.plot(horas, soc, color='gray', marker='o', linestyle='-', linewidth=2, markersize=4, label='SOC (%)')
        ax2.set_ylabel('SOC (%)', fontsize=12)
        ax2.set_ylim(0, 105)

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
    # Exemplo de teste rápido
    plotter = BatteryPlotter()
    plotter.plot_all_batteries()