#!/usr/bin/env python3
"""
DATAGenerator.py

Gera múltiplos cenários de simulação multi-dia OPF para alimentar a base de dados.
Executa a simulação N vezes (padrão 13200), cada vez com fatores de vento/carga
aleatórios e SOC inicial variável, salvando os resultados no banco SQLite.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time
import traceback

# Adiciona o diretório SRC ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UTILS.systemLoader import SistemaLoader
from DB.OPF_DBhandler import OPF_DBHandler
from OPT.DC_OPF_BESS_Desacoplado import MultiDayOPFModel

# Configurações
JSON_PATH = "DATA/input/B6L8_BASE.json"          # arquivo do sistema (ajuste se necessário)
WIND_FILE = "/home/lucasedbraga/repositorios/ufjf/mestrado_luedsbr/SRC/SOLVER/DB/getters/intermittent-renewables-production-france.csv"
DB_PATH = "DATA/output/resultados_PL_RNA.db"

N_ITERACOES = 100         # número total de cenários a gerar
N_DIAS = 7                   # dias por simulação
N_HORAS = 24                 # horas por dia

# Para reprodutibilidade, comente a linha abaixo se quiser aleatoriedade total
# np.random.seed(42)

def carregar_dados_vento(caminho):
    """Carrega e processa o arquivo de vento, retornando DataFrame com fatores."""
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo de vento não encontrado: {caminho}")
    df = pd.read_csv(caminho)
    df['DateTime'] = pd.to_datetime(df['Date and Hour'].str.slice(stop=-6))
    df = df.sort_values('DateTime')
    df_wind = df[df['Source'] == 'Wind'].copy()
    if len(df_wind) == 0:
        raise ValueError("Nenhum dado de vento encontrado.")
    max_prod = df_wind['Production'].max()
    df_wind['Factor'] = df_wind['Production'] / max_prod if max_prod > 0 else 0
    return df_wind[['Factor']].rename(columns={'Factor': 'wind_factor'})

def gerar_fatores_aleatorios(df_vento, n_dias, n_horas):
    """
    Gera matrizes de fatores de vento e carga.
    Vento: amostra com reposição do histórico.
    Carga: distribuição normal com média 1 e desvio 0.3.
    """
    # Vento
    amostras = df_vento.sample(n_dias * n_horas, replace=True)['wind_factor'].values
    fator_vento = amostras.reshape((n_dias, n_horas))

    # Carga
    fator_carga = np.random.normal(1, 0.3, size=(n_dias, n_horas))

    return fator_vento, fator_carga

def main():
    print("=" * 70)
    print("GERADOR DE DADOS PARA TREINAMENTO DA REDE NEURAL")
    print(f"Total de iterações: {N_ITERACOES}")
    print("=" * 70)

    # 1. Carregar sistema (uma única vez)
    print("\n[1] Carregando sistema...")
    if not os.path.exists(JSON_PATH):
        print(f"ERRO: Arquivo do sistema não encontrado: {JSON_PATH}")
        return 1
    sistema = SistemaLoader(JSON_PATH)
    print(f"   Sistema: {JSON_PATH}")
    print(f"   Barras: {sistema.NBAR}, Geradores: {sistema.NGER_ORIGINAL}")

    # 2. Carregar dados de vento
    print("\n[2] Carregando dados de vento...")
    try:
        df_vento = carregar_dados_vento(WIND_FILE)
        print(f"   {len(df_vento)} registros de fator de vento carregados.")
    except Exception as e:
        print(f"ERRO ao carregar vento: {e}")
        return 1

    # 3. Preparar banco de dados
    print("\n[3] Conectando ao banco de dados...")
    db_handler = OPF_DBHandler(DB_PATH)
    db_handler.create_tables()  # garante que as tabelas existem
    print(f"   Banco: {DB_PATH}")

    print(f"\n[4] Iniciando geração de {N_ITERACOES} cenários...")
    inicio_global = time.time()

    # SOC inicial fixo (50% da capacidade). Se quiser variar, mova para dentro do loop.
    soc_inicial_fixo = 0.5 * sistema.BATTERY_CAPACITY

    for i in range(N_ITERACOES):
        try:
            # Criar um ID único para o cenário (timestamp + contador)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            cen_id = f"{timestamp}_{i:05d}"

            # Gerar fatores aleatórios
            fator_vento, fator_carga = gerar_fatores_aleatorios(df_vento, N_DIAS, N_HORAS)

            # (Opcional) Variar SOC inicial aleatoriamente entre 0 e capacidade total
            # soc_inicial = np.random.uniform(0, sistema.BATTERY_CAPACITY)
            soc_inicial = soc_inicial_fixo  # ou use a linha acima para aleatoriedade

            # Criar modelo multi-dia SEM salvamento automático (db_handler=None)
            modelo = MultiDayOPFModel(sistema, n_horas=N_HORAS, n_dias=N_DIAS, db_handler=None)

            # Resolver o problema multi-dia
            opf_result = modelo.solve_multiday_sequencial(
                solver_name='glpk',
                fator_carga=fator_carga,
                fator_vento=fator_vento,
                soc_inicial=soc_inicial,
                cen_id=cen_id
            )

            # Salvar cada snapshot manualmente
            for snapshot in opf_result.snapshots:
                db_handler.save_hourly_result(
                    resultado=snapshot,
                    sistema=sistema,
                    hora=snapshot.hora,
                    perfil_carga=fator_carga[snapshot.dia, snapshot.hora],
                    perfil_eolica=fator_vento[snapshot.dia, snapshot.hora],
                    solver_name='glpk',
                    dia=f"{snapshot.dia+1}",
                    cen_id=cen_id
                )

            # Verificar sucesso da simulação (opcional)
            n_sucesso = sum(1 for s in opf_result.snapshots if s.sucesso)
            print(f"   [{i+1:5d}/{N_ITERACOES}] Cenário {cen_id} concluído - {n_sucesso}/{N_DIAS*N_HORAS} snaps OK")

        except Exception as e:
            print(f"   [!] Erro na iteração {i}: {e}")
            traceback.print_exc()
            # Continua para a próxima iteração

    # Estatísticas finais
    tempo_total = time.time() - inicio_global
    print("\n" + "=" * 70)
    print("GERAÇÃO CONCLUÍDA")
    print(f"Total de iterações processadas: {N_ITERACOES}")
    print(f"Tempo total: {tempo_total:.2f} s")
    print(f"Média por iteração: {tempo_total/N_ITERACOES:.2f} s")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())