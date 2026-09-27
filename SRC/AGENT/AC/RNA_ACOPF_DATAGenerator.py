#!/usr/bin/env python3
"""
DATAGenerator_ACOPF_TimeCoupled.py

Gera múltiplos cenários de simulação multi‑dia utilizando o modelo ACOPF_TimeCoupled
(AC OPF não‑linear acoplado no tempo) com Pyomo + IPOPT.
Considera todos os períodos em um único problema de otimização, incluindo restrições
de bateria, balanço de potência AC e perdas reais.
Cada cenário é salvo no banco SQLite.
"""

import sys
import os
import time
import traceback
import numpy as np
from datetime import datetime
import secrets

# Ajusta o path para encontrar os módulos do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))[:-6])
from UTILS.SystemLoader import SistemaLoader
from UTILS.beep import *
from DB.DBhandler_OPF import OPF_DBHandler
from UTILS.EvaluateFactors import EvaluateFactors

# Importa a classe do modelo AC acoplado (ajuste o caminho se necessário)
from SOLVER.OPF_AC.AC_OPF_Snapshot import ACOPF_Snapshot_Model
from SOLVER.OPF_AC.AC_OPF_Acoplado import ACOPF_TimeCoupled

# ==============================================================================================
# Configurações
# ==============================================================================================
JSON_PATH = "DATA/input/ieee14_BESS.json"        # arquivo do sistema
TIPO="BESS"
DB_PATH = f"DATA/output_IEEE_Test/IEEE14_RNA_DATA_ACOPF_{TIPO}.db"

N_ITERACOES = 1000      # número total de cenários
N_DIAS = 1              # dias por simulação
N_HORAS = 24            # horas por dia
HORA_DESEJADA = 18


# Parâmetros da bateria
SOC_INICIAL_FRACAO = 0.5
SOC_FINAL_FRACAO = 0.5

# Opções do solver
SOLVER_NAME = 'ipopt'   # obrigatório para AC OPF não‑linear
TEE = False             # não exibe logs do solver durante a geração em lote
MAX_ITER = 5
WRITE_LP = False         

def GERA_DADOS_ACOPF_snapshot():
    print("=" * 70)
    print("GERADOR DE DADOS (AC OPF Snapshot – Não‑Linear)")
    print(f"Total de iterações: {N_ITERACOES}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Carregar sistema
    # -------------------------------------------------------------------------
    print("\n[1] Carregando sistema...")
    if not os.path.exists(JSON_PATH):
        print(f"ERRO: Arquivo do sistema não encontrado: {JSON_PATH}")
        return 1
    sistema = SistemaLoader(JSON_PATH)
    print(f"   Sistema: {JSON_PATH}")
    print(f"   Barras: {sistema.NBAR}, Geradores: {sistema.NGER_UTE}")
    cap_str = ', '.join([f"{c:.2f}" for c in sistema.BATTERY_CAPACITY])
    print(f"   Capacidade bateria: {cap_str} MWh")

    # -------------------------------------------------------------------------
    # 2. Conectar ao banco de dados
    # -------------------------------------------------------------------------
    print("\n[2] Conectando ao banco de dados...")
    db_handler = OPF_DBHandler(DB_PATH)
    db_handler.create_tables()
    print(f"   Banco: {DB_PATH}")

    # -------------------------------------------------------------------------
    # 3. Criar o modelo AC acoplado
    # -------------------------------------------------------------------------
    print("\n[3] Criando modelo integrado AC...")
    modelo = ACOPF_Snapshot_Model(sistema=sistema, db_handler=db_handler)
    print("   Modelo criado.")

    # -------------------------------------------------------------------------
    # 4. Loop principal de geração de cenários
    # -------------------------------------------------------------------------
    print(f"\n[4] Iniciando geração de {N_ITERACOES} cenários...")
    inicio_global = time.time()

    for i in range(N_ITERACOES):
        try:
            # ID único do cenário (timestamp + contador)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            cen_id = f"{timestamp}_{i:05d}"

            # Gerar fatores de carga e vento para este cenário
            seed = secrets.randbits(32)
            hora_desejada = HORA_DESEJADA

            modelo = ACOPF_Snapshot_Model(sistema=sistema, db_handler=db_handler)

            avaliador = EvaluateFactors(sistema=sistema, n_dias=1, n_horas=1,
                                        carga_incerteza=0.2, vento_variacao=0.1, seed=seed)
            
            fatores_carga_completo, fatores_vento_completo = avaliador.gerar_tudo()

            fator_carga_hora = fatores_carga_completo[0, 0, :]
            fator_vento_hora = fatores_vento_completo[0, 0, :] if sistema.NGER_GWD > 0 else 1.0

            soc_baterias = {b: 0.5 for b in sistema.BARRAS_COM_BATERIA}
            print(f"   SOC inicial das baterias: {soc_baterias}")

            # Resolver o problema integrado AC
            _ = modelo.solve_snapshot(
                solver_name='ipopt',
                fator_carga=fator_carga_hora,
                fator_vento=fator_vento_hora,
                soc_baterias=soc_baterias,
                hora=hora_desejada,
                dia=0,
                cen_id=cen_id
            )

            print(f"   [{i+1:5d}/{N_ITERACOES}] Cenário {cen_id} concluído")

        except Exception as e:
            print(f"   [!] Erro na iteração {i}: {e}")
            traceback.print_exc()
            # Continua para a próxima iteração

    # -------------------------------------------------------------------------
    # Estatísticas finais
    # -------------------------------------------------------------------------
    tempo_total = time.time() - inicio_global
    print("\n" + "=" * 70)
    beep()
    print("GERAÇÃO CONCLUÍDA")
    print(f"Total de iterações processadas: {N_ITERACOES}")
    print(f"Tempo total: {tempo_total:.2f} s")
    print(f"Média por iteração: {tempo_total/N_ITERACOES:.2f} s")
    print("=" * 70)

    return 0

def GERA_DADOS_ACOPF_acoplado():
    print("=" * 70)
    print("GERADOR DE DADOS (AC OPF Acoplado – Não‑Linear)")
    print(f"Total de iterações: {N_ITERACOES}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Carregar sistema
    # -------------------------------------------------------------------------
    print("\n[1] Carregando sistema...")
    if not os.path.exists(JSON_PATH):
        print(f"ERRO: Arquivo do sistema não encontrado: {JSON_PATH}")
        return 1
    sistema = SistemaLoader(JSON_PATH)
    print(f"   Sistema: {JSON_PATH}")
    print(f"   Barras: {sistema.NBAR}, Geradores: {sistema.NGER_UTE}")
    cap_str = ', '.join([f"{c:.2f}" for c in sistema.BATTERY_CAPACITY])
    print(f"   Capacidade bateria: {cap_str} MWh")

    # -------------------------------------------------------------------------
    # 2. Conectar ao banco de dados
    # -------------------------------------------------------------------------
    print("\n[2] Conectando ao banco de dados...")
    db_handler = OPF_DBHandler(DB_PATH)
    db_handler.create_tables()
    print(f"   Banco: {DB_PATH}")

    # -------------------------------------------------------------------------
    # 3. Criar o modelo AC acoplado
    # -------------------------------------------------------------------------
    print("\n[3] Criando modelo integrado AC...")
    print("\n[3] Criando modelo integrado...")
    modelo = ACOPF_TimeCoupled(
        sistema=sistema,
        n_horas=N_HORAS,
        n_dias=N_DIAS,
        db_handler=db_handler,
        dia_inicial=0
    )

    print("   Modelo criado.")

    # -------------------------------------------------------------------------
    # 4. Loop principal de geração de cenários
    # -------------------------------------------------------------------------

    print(f"\n[4] Iniciando geração de {N_ITERACOES} cenários...")
    inicio_global = time.time()

    for i in range(N_ITERACOES):
        try:
            # Criar ID único para o cenário (timestamp + contador)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            cen_id = f"{timestamp}_{i:05d}"

            # -----------------------------------------------------------------
            # Gerar fatores de carga e vento para este cenário
            # -----------------------------------------------------------------
            seed = secrets.randbits(32) 
            avaliador = EvaluateFactors(
                sistema=sistema,
                n_dias=N_DIAS,
                n_horas=N_HORAS,
                carga_incerteza=0.2,
                vento_variacao=0.1,
                seed=seed
            )
            fatores_carga, fatores_vento = avaliador.gerar_tudo()

            #Variar SOC inicial/final aleatoriamente
            SOC_inicial = 0.5
            SOC_final = 0.5

            # -----------------------------------------------------------------
            # Resolver o problema integrado
            # -----------------------------------------------------------------
            _ = modelo.solve_timecoupled(
                solver_name='ipopt',
                fator_carga=fatores_carga,
                fator_vento=fatores_vento,
                soc_inicial=SOC_inicial,
                soc_final=SOC_final,
                cen_id=cen_id,
                tee=True
            )

            print(f"   [{i+1:5d}/{N_ITERACOES}] Cenário {cen_id} concluído")

        except Exception as e:
            print(f"   [!] Erro na iteração {i}: {e}")
            traceback.print_exc()
            # Continua para a próxima iteração

    # -------------------------------------------------------------------------
    # Estatísticas finais
    # -------------------------------------------------------------------------
    tempo_total = time.time() - inicio_global
    print("\n" + "=" * 70)
    print("GERAÇÃO CONCLUÍDA")
    beep()
    print(f"Total de iterações processadas: {N_ITERACOES}")
    print(f"Tempo total: {tempo_total:.2f} s")
    print(f"Média por iteração: {tempo_total/N_ITERACOES:.2f} s")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    #sys.exit(GERA_DADOS_ACOPF_snapshot())
    sys.exit(GERA_DADOS_ACOPF_acoplado())