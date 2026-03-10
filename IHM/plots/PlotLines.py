"""
plot_line.py
Classe para visualizar o grafo do sistema elétrico com cores indicando carregamento das linhas.
Versão com suporte a múltiplos instantes e anotação dos valores de fluxo e carregamento nas arestas.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
from datetime import datetime

class PlotLine:
    """
    Visualiza o grafo do sistema elétrico a partir dos dados da tabela DLIN_results,
    colorindo as linhas conforme o carregamento (LIN_usage_result).
    """

    def __init__(self,
                 db_path: str = 'DATA/output/resultados_PL.db',
                 cen_id: Optional[str] = None,
                 data_simulacao: Optional[str] = None,
                 hora_simulacao: Optional[int] = None,
                 pos_barras: Optional[Dict[int, Tuple[float, float]]] = None,
                 thresholds: Tuple[float, float] = (50.0, 80.0)):
        """
        Parâmetros:
        -----------
        db_path : str
            Caminho para o banco SQLite.
        cen_id : str, optional
            Identificador do cenário. Se None, lista os disponíveis e usa o primeiro.
        data_simulacao : str, optional
            Data da simulação (formato YYYYMMDD). Se None, usa a primeira disponível.
        hora_simulacao : int, optional
            Hora da simulação (0-23). Se None, usa a primeira disponível.
        pos_barras : dict, optional
            Dicionário com coordenadas das barras: {bar_id: (x, y)}.
            Se None, gera um layout automático (spring_layout).
        thresholds : tuple
            Limiares para cores: (limite_verde_amarelo, limite_amarelo_vermelho) em percentual.
            Padrão: (50.0, 80.0)
        """
        self.db_path = db_path
        self.cen_id = cen_id
        self.data_simulacao = data_simulacao
        self.hora_simulacao = hora_simulacao
        self.pos_barras = pos_barras
        self.thresholds = thresholds
        self.df_linhas = None
        self.graph = None
        self._load_data()

    def _list_cenarios(self) -> List[str]:
        """Lista todos os cen_id disponíveis na tabela resultados_opf."""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT DISTINCT cen_id FROM resultados_opf ORDER BY cen_id", conn)
            conn.close()
            return df['cen_id'].tolist()
        except Exception as e:
            print(f" Erro ao listar cenários: {e}")
            return []

    def _list_available_hours(self) -> pd.DataFrame:
        """Lista combinações de data_simulacao e hora_simulacao para o cenário selecionado."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"""
                SELECT DISTINCT data_simulacao, hora_simulacao
                FROM DLIN_results
                WHERE cen_id = '{self.cen_id}'
                ORDER BY data_simulacao, hora_simulacao
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f" Erro ao listar horas disponíveis: {e}")
            return pd.DataFrame()

    def get_available_timestamps(self) -> List[Tuple[str, int]]:
        """
        Retorna uma lista ordenada de tuplas (data_simulacao, hora_simulacao)
        disponíveis para o cenário atual.
        """
        df = self._list_available_hours()
        if df.empty:
            return []
        return list(zip(df['data_simulacao'], df['hora_simulacao']))

    def _load_data(self):
        """Carrega dados da tabela DLIN_results para o cenário e instante especificados."""
        # Determinar cen_id se não fornecido
        if self.cen_id is None:
            cenarios = self._list_cenarios()
            if not cenarios:
                print(" Nenhum cenário encontrado no banco.")
                return
            self.cen_id = cenarios[0]
            print(f"  Nenhum cen_id informado. Usando o primeiro disponível: {self.cen_id}")

        # Se data/hora não informadas, buscar a primeira disponível
        if self.data_simulacao is None or self.hora_simulacao is None:
            df_hours = self._list_available_hours()
            if df_hours.empty:
                print(f" Nenhum dado encontrado para o cenário {self.cen_id} na tabela DLIN_results.")
                return
            self.data_simulacao = df_hours.iloc[0]['data_simulacao']
            self.hora_simulacao = int(df_hours.iloc[0]['hora_simulacao'])
            print(f"  Usando primeiro instante disponível: data={self.data_simulacao}, hora={self.hora_simulacao:02d}")

        print(f" Carregando dados das linhas para cenário {self.cen_id}, data {self.data_simulacao}, hora {self.hora_simulacao:02d}...")

        try:
            conn = sqlite3.connect(self.db_path)
            query_lin = f"""
                SELECT 
                    linha_id,
                    de_barra,
                    para_barra,
                    PLIM_FLUX,
                    FLUX_result,
                    PLOSS_result,
                    LIN_usage_result
                FROM DLIN_results
                WHERE cen_id = '{self.cen_id}'
                  AND data_simulacao = '{self.data_simulacao}'
                  AND hora_simulacao = {self.hora_simulacao}
                ORDER BY linha_id
            """
            df_lin = pd.read_sql_query(query_lin, conn)
            conn.close()

            if df_lin.empty:
                print(" Nenhum dado encontrado na tabela DLIN_results para este instante.")
                return

            # Garantir tipos numéricos
            numeric_cols = ['PLIM_FLUX', 'FLUX_result', 'PLOSS_result', 'LIN_usage_result']
            for col in numeric_cols:
                df_lin[col] = pd.to_numeric(df_lin[col], errors='coerce').fillna(0)

            self.df_linhas = df_lin
            print(f" Dados carregados: {len(df_lin)} linhas.")

            # Construir grafo
            self._build_graph()

        except Exception as e:
            print(f" Erro ao carregar dados: {e}")
            self.df_linhas = None

    def _build_graph(self):
        """Constrói o grafo networkx a partir do DataFrame de linhas."""
        self.graph = nx.Graph()
        for _, row in self.df_linhas.iterrows():
            de = int(row['de_barra'])
            para = int(row['para_barra'])
            usage = row['LIN_usage_result']
            fluxo = row['FLUX_result']
            limite = row['PLIM_FLUX']
            perdas = row['PLOSS_result']
            self.graph.add_edge(de, para, usage=usage, fluxo=fluxo, limite=limite, perdas=perdas, linha_id=row['linha_id'])

    def _get_color(self, usage: float) -> str:
        """Retorna a cor baseada no carregamento."""
        verde_amarelo, amarelo_vermelho = self.thresholds
        if usage < verde_amarelo:
            return 'green'
        elif usage < amarelo_vermelho:
            return 'yellow'
        else:
            return 'red'

    def plot(self, save_fig: bool = False, output_dir: str = '.',
             figsize: Tuple[int, int] = (12, 8),
             node_size: int = 500,
             with_labels: bool = True,
             edge_width_factor: float = 2.0,
             show_colorbar: bool = True,
             show_edge_values: bool = True,
             edge_text_color: str = 'black',
             edge_text_bgcolor: str = 'white',
             edge_text_fontsize: int = 8,
             edge_text_format: str = "{fluxo:.1f}MW\n{usage:.1f}%"):
        """
        Gera o plot do grafo para o instante atualmente configurado.

        Parâmetros:
        -----------
        save_fig : bool
            Se True, salva a figura em arquivo.
        output_dir : str
            Diretório para salvar a figura (se save_fig=True).
        figsize : tuple
            Tamanho da figura.
        node_size : int
            Tamanho dos nós.
        with_labels : bool
            Se True, exibe os IDs das barras nos nós.
        edge_width_factor : float
            Fator para escalar a largura das arestas (largura = carregamento/100 * factor).
        show_colorbar : bool
            Se True, exibe uma barra de cores para referência.
        show_edge_values : bool
            Se True, exibe valores de fluxo e carregamento nas arestas.
        edge_text_color : str
            Cor do texto nas arestas.
        edge_text_bgcolor : str
            Cor de fundo do texto (pode ser 'none' para sem fundo).
        edge_text_fontsize : int
            Tamanho da fonte do texto nas arestas.
        edge_text_format : str
            Formato do texto. Use {fluxo}, {usage}, {limite}, {perdas} como placeholders.
        """
        if self.graph is None or self.graph.number_of_edges() == 0:
            print(" Grafo vazio. Não é possível plotar.")
            return

        # Definir posições dos nós
        if self.pos_barras is not None:
            pos = {node: self.pos_barras.get(node, (0, 0)) for node in self.graph.nodes()}
        else:
            # Usar layout fixo para consistência
            if not hasattr(self, '_fixed_pos'):
                self._fixed_pos = nx.spring_layout(self.graph, seed=42)
            pos = self._fixed_pos

        # Extrair atributos das arestas
        usages = [self.graph[u][v]['usage'] for u, v in self.graph.edges()]
        colors = [self._get_color(u) for u in usages]
        widths = [max(1.0, u / 100 * edge_width_factor) for u in usages]

        # Criar figura
        plt.figure(figsize=figsize)
        ax = plt.gca()

        # Desenhar nós
        nx.draw_networkx_nodes(self.graph, pos, node_size=node_size, node_color='lightblue',
                               edgecolors='black', linewidths=1, ax=ax)

        # Desenhar arestas com cores baseadas no carregamento
        nx.draw_networkx_edges(self.graph, pos, edge_color=colors, width=widths, alpha=0.7, ax=ax)

        # Rótulos dos nós
        if with_labels:
            nx.draw_networkx_labels(self.graph, pos, font_size=10, font_weight='bold', ax=ax)

        # Adicionar valores nas arestas
        if show_edge_values:
            for (u, v, data) in self.graph.edges(data=True):
                # Ponto médio da aresta
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                xm = (x0 + x1) / 2
                ym = (y0 + y1) / 2
                
                # Formatar texto
                text = edge_text_format.format(
                    fluxo=data['fluxo'],
                    usage=data['usage'],
                    limite=data['limite'],
                    perdas=data['perdas']
                )
                
                # Adicionar texto com fundo branco (opcional)
                bbox_props = dict(boxstyle="round,pad=0.3", facecolor=edge_text_bgcolor, alpha=0.7) if edge_text_bgcolor != 'none' else None
                ax.text(xm, ym, text, fontsize=edge_text_fontsize,
                        color=edge_text_color, ha='center', va='center',
                        bbox=bbox_props)

        # Título
        plt.title(f"Sistema Elétrico - Carregamento das Linhas\nCenário: {self.cen_id} | Data: {self.data_simulacao} Hora: {self.hora_simulacao:02d}")

        # Barra de cores (manual)
        if show_colorbar:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize, ListedColormap
            cmap = ListedColormap(['green', 'yellow', 'red'])
            norm = Normalize(vmin=0, vmax=100)
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label('Carregamento da Linha (%)')
            # Marcar thresholds
            cbar.ax.axhline(y=self.thresholds[0]/100, color='black', linestyle='--', linewidth=0.8)
            cbar.ax.axhline(y=self.thresholds[1]/100, color='black', linestyle='--', linewidth=0.8)

        plt.tight_layout()

        if save_fig:
            os.makedirs(output_dir, exist_ok=True)
            fname = os.path.join(output_dir, f"grafo_{self.cen_id}_{self.data_simulacao}_h{self.hora_simulacao:02d}.png")
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            print(f" Figura salva: {fname}")

        plt.show()

    def plot_all_hours(self, save_fig: bool = True, output_dir: str = './grafos_horarios',
                       figsize: Tuple[int, int] = (10, 5),
                       node_size: int = 500,
                       with_labels: bool = True,
                       edge_width_factor: float = 2.0,
                       show_colorbar: bool = True,
                       show_edge_values: bool = True,
                       edge_text_format: str = "{fluxo:.1f}MW\n{usage:.1f}%",
                       verbose: bool = True,
                       **kwargs):
        """
        Gera plots para TODAS as horas disponíveis no banco para o cenário atual.
        Para cada combinação (data_simulacao, hora_simulacao), recarrega os dados e gera uma figura.

        Parâmetros:
        -----------
        save_fig : bool
            Se True, salva as figuras (recomendado True para não exibir centenas de plots).
        output_dir : str
            Diretório base onde as subpastas por data serão criadas.
        figsize, node_size, with_labels, edge_width_factor, show_colorbar, show_edge_values,
        edge_text_format, kwargs:
            Repassados para o método plot.
        verbose : bool
            Se True, imprime o progresso.
        """
        # Obter todos os instantes disponíveis
        timestamps = self.get_available_timestamps()
        if not timestamps:
            print(" Nenhum instante disponível para este cenário.")
            return

        print(f" Gerando gráficos para {len(timestamps)} instantes...")

        for i, (data, hora) in enumerate(timestamps, 1):
            if verbose:
                print(f"\n[{i}/{len(timestamps)}] Processando data {data}, hora {hora:02d}...")

            # Atualizar o instante e recarregar dados
            self.data_simulacao = data
            self.hora_simulacao = hora
            self._load_data()

            if self.df_linhas is not None:
                # Definir subdiretório por data para organização
                data_dir = os.path.join(output_dir, data)
                self.plot(save_fig=save_fig, output_dir=data_dir,
                          figsize=figsize, node_size=node_size,
                          with_labels=with_labels,
                          edge_width_factor=edge_width_factor,
                          show_colorbar=show_colorbar,
                          show_edge_values=show_edge_values,
                          edge_text_format=edge_text_format,
                          **kwargs)
                if not save_fig:
                    plt.close()  # Fecha a figura para não acumular
            else:
                if verbose:
                    print(f"  Sem dados para {data} H{hora:02d}")

        print(f"\n Processamento concluído. Figuras salvas em: {output_dir}")


if __name__ == '__main__':
    # Exemplo de uso: gerar todos os gráficos para o primeiro cenário encontrado
    plotter = PlotLine()  # Usa o primeiro cenário
    plotter.plot_all_hours(save_fig=False, output_dir='./grafos_completos')

    # Caso queira um cenário específico:
    # plotter = PlotLine(cen_id='20250213143000')
    # plotter.plot_all_hours(save_fig=True)