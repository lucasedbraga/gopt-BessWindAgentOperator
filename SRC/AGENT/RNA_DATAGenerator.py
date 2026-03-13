#!/usr/bin/env python3
"""
DATAGenerator_TimeCoupled.py

Gera múltiplos cenários de simulação multi-dia utilizando o modelo integrado no tempo
(TimeCoupledOPFModelPyOptInterface), que considera todos os períodos em um único problema de otimização
com restrições de bateria e perdas iterativas. Cada cenário é salvo no banco SQLite.
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

from UTILS.SystemLoader import SistemaLoader
from DB.DBhandler_OPF import OPF_DBHandler
from UTILS.EvaluateFactors import EvaluateFactors

# Import da classe do modelo acoplado (versão PyOptInterface)
from SOLVER.OPF_DC.DC_OPF_Acoplado import TimeCoupledOPFModel

# ==============================================================================================

# Configurações
JSON_PATH = "DATA/input/ieee33_BASE.json"        # arquivo do sistema
DB_PATH = "DATA/output/RNA_resultados_PL_acoplado.db"

N_ITERACOES = 1000      # número total de cenários
N_DIAS = 7             # dias por simulação
N_HORAS = 24            # horas por dia

# Parâmetros da bateria 
SOC_INICIAL_FRACAO = 0.5   
SOC_FINAL_FRACAO = 0.5     

# Opções do modelo
CONSIDERAR_PERDAS = True
SOLVER_NAME = 'highs'      
TOL = 1e-4
MAX_ITER = 5
WRITE_LP = False           


def main():
    print("=" * 70)
    print("GERADOR DE DADOS COM MODELO INTEGRADO NO TEMPO (DC OPF) - PyOptInterface")
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
    print(f"   Barras: {sistema.NBAR}, Geradores: {sistema.NGER_CONV}")
    cap_str = ', '.join([f"{c:.2f}" for c in sistema.BATTERY_CAPACITY])
    print(f"   Capacidade bateria: {cap_str} MWh")

    # -------------------------------------------------------------------------
    # 2. Preparar banco de dados
    # -------------------------------------------------------------------------
    print("\n[2] Conectando ao banco de dados...")
    db_handler = OPF_DBHandler(DB_PATH)
    db_handler.create_tables() 
    print(f"   Banco: {DB_PATH}")

    # -------------------------------------------------------------------------
    # 3. Criar o modelo
    # -------------------------------------------------------------------------
    print("\n[3] Criando modelo integrado...")
    modelo = TimeCoupledOPFModel(
        sistema=sistema,
        n_horas=N_HORAS,
        n_dias=N_DIAS,
        db_handler=db_handler,         
        considerar_perdas=CONSIDERAR_PERDAS,
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
            soc_inicial_frac = SOC_INICIAL_FRACAO  
            soc_final_frac = SOC_FINAL_FRACAO

            # -----------------------------------------------------------------
            # Resolver o problema integrado
            # -----------------------------------------------------------------
            _ = modelo.solve_multiday(
                solver_name=SOLVER_NAME,
                fator_carga=fatores_carga,
                fator_vento=fatores_vento,
                soc_inicial=soc_inicial_frac,
                soc_final=soc_final_frac,
                cen_id=cen_id,
                tol=TOL,
                max_iter=MAX_ITER,
                write_lp=WRITE_LP
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
    print(f"Total de iterações processadas: {N_ITERACOES}")
    print(f"Tempo total: {tempo_total:.2f} s")
    print(f"Média por iteração: {tempo_total/N_ITERACOES:.2f} s")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())