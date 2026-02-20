#!/usr/bin/env python3
"""
Script principal para resolver um único snapshot de OPF (multi-dia)
com integração ao banco de dados SQLite.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Adiciona o diretório SRC ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UTILS.systemLoader import SistemaLoader
from DB.OPF_DBhandler import OPF_DBHandler
from SRC.OPT.DC_OPF_BESS_Desacoplado import MultiDayOPFModel

def main():
    print("=" * 70)
    print("SISTEMA DE OTIMIZAÇÃO DE FLUXO DE POTÊNCIA (OPF) - MULTI-DIA")
    print("=" * 70)
    try:
        # ------------------------------------------------------------------
        # 1. CARREGAR SISTEMA
        # ------------------------------------------------------------------
        print("\n1. Carregando dados do sistema...")
        #json_path = "DATA/input/3barras_BASE.json"
        json_path = "DATA/input/B6L8_BASE.json"
        #json_path = "DATA/input/ieee14_BASE.json"
        # json_path = "DATA/input/ieee118_BASE.json"

        if not os.path.exists(json_path):
            print(f"ERRO: Arquivo não encontrado: {json_path}")
            print("Por favor, verifique se o arquivo existe em DATA/input/")
            return 1

        sistema_loader = SistemaLoader(json_path)
        sistema = sistema_loader

        print(f"   ✓ Sistema carregado: {json_path}")
        print(f"   ✓ Potência base: {sistema.SB:.1f} MVA")
        print(f"   ✓ Barras: {sistema.NBAR}")
        print(f"   ✓ Linhas: {sistema.NLIN}")
        print(f"   ✓ Geradores: {sistema.NGER_ORIGINAL} originais + {sistema.NGER_CURTAILMENT} curtailment + {sistema.NGER_DEFICIT} déficit")
        print(f"   ✓ Geradores eólicos (GWD): {len(sistema.BAR_GWD)}")
        print(f"   ✓ Carga total: {np.sum(sistema.PLOAD):.3f} pu ({np.sum(sistema.PLOAD) * sistema.SB:.1f} MW)")

        # ------------------------------------------------------------------
        # 2. CARREGAR DADOS DE VENTO (FATORES)
        # ------------------------------------------------------------------
        print("\n2. Carregando dados de vento...")
        filepath = r"/home/lucasedbraga/repositorios/ufjf/mestrado_luedsbr/SRC/SOLVER/DB/getters/intermittent-renewables-production-france.csv"
        if not os.path.exists(filepath):
            print(f"ERRO: Arquivo de vento não encontrado: {filepath}")
            return 1

        df = pd.read_csv(filepath)
        df['DateTime'] = pd.to_datetime(df['Date and Hour'].str.slice(stop=-6))
        df = df.sort_values('DateTime')
        df_wind = df[df['Source'] == 'Wind'].copy()
        if len(df_wind) == 0:
            print("ERRO: Nenhum dado de vento encontrado no arquivo.")
            return 1

        max_prod = df_wind['Production'].max()
        if max_prod > 0:
            df_wind['Factor'] = df_wind['Production'] / max_prod
        else:
            df_wind['Factor'] = 0

        cols = ['DateTime', 'Production', 'Factor']
        df_wind = df_wind[cols].copy()
        df_wind.columns = ['timestamp', 'wind_production_mw', 'wind_factor']
        print(f"   ✓ Dados carregados: {len(df_wind)} registros")

        # ------------------------------------------------------------------
        # 3. CONFIGURAR SIMULAÇÃO MULTI-DIA
        # ------------------------------------------------------------------
        n_dias = 2
        n_horas = 24
        print(f"\n3. Configurando simulação: {n_dias} dias, {n_horas} horas/dia")

        # SOC inicial das baterias (exemplo: 50% da capacidade)
        SOC_inicial_bateria = 0.5 * sistema.BATTERY_CAPACITY

        # Gerar matriz de fatores de vento (dias x horas) por amostragem aleatória
        # Garantindo que seja um array 2D de floats
        amostras_vento = df_wind["wind_factor"].sample(n_dias * n_horas, replace=True).values
        fator_vento = amostras_vento.reshape((n_dias, n_horas))

        # Gerar matriz de fatores de carga (distribuição normal em torno de 1)
        fator_carga = np.random.normal(1, 0.3, size=(n_dias, n_horas))

        print(f"   ✓ Fatores de vento (exemplo primeiro dia): {fator_vento[0, :5]} ...")
        print(f"   ✓ Fatores de carga (exemplo primeiro dia): {fator_carga[0, :5]} ...")

        # ------------------------------------------------------------------
        # 4. CRIAR MODELO E RESOLVER (COM SALVAMENTO AUTOMÁTICO OPCIONAL)
        # ------------------------------------------------------------------
        print("\n4. Iniciando simulação multi-dia...")

        # Opção A: Usar salvamento automático (recomendado)
        usar_salvamento_automatico = True

        if usar_salvamento_automatico:
            # Criar handler e tabelas
            db_handler = OPF_DBHandler('DATA/output/resultados_PL.db')
            db_handler.create_tables()
            # Criar modelo passando o handler
            multi_model = MultiDayOPFModel(sistema, n_horas=n_horas, n_dias=n_dias, db_handler=db_handler)
            cen_id = datetime.now().strftime('%Y%m%d%H%M%S')
            print(f"   ✓ Cenário ID: {cen_id}")
            # Resolver – o modelo salvará automaticamente cada snapshot
            opf_result = multi_model.solve_multiday_sequencial(
                solver_name='glpk',
                fator_carga=fator_carga,
                fator_vento=fator_vento,
                soc_inicial=SOC_inicial_bateria,
                cen_id=cen_id          # necessário para o salvamento automático
            )
        else:
            # Opção B: Salvamento manual (como no script original)
            multi_model = MultiDayOPFModel(sistema, n_horas=n_horas, n_dias=n_dias, db_handler=None)
            opf_result = multi_model.solve_multiday_sequencial(
                solver_name='glpk',
                fator_carga=fator_carga,
                fator_vento=fator_vento,
                soc_inicial=SOC_inicial_bateria
            )
            # Agora salvar manualmente
            db_handler = OPF_DBHandler('DATA/output/resultados_PL.db')
            db_handler.create_tables()
            cen_id = datetime.now().strftime('%Y%m%d%H%M%S')
            print(f"   ✓ Cenário ID: {cen_id}")
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

        # ------------------------------------------------------------------
        # 5. EXIBIR RESUMO DOS RESULTADOS
        # ------------------------------------------------------------------
        print("\n5. Resumo dos resultados:")
        for snapshot in opf_result.snapshots:
            status = "✓" if snapshot.sucesso else "✗"
            print(f"   Dia {snapshot.dia+1}, Hora {snapshot.hora:02d}:00 {status}  "
                  f"Custo: {snapshot.custo_total:8.2f}  "
                  f"Déficit: {snapshot.deficit_total:6.2f} MW")

        print("\n" + "=" * 70)
        print("EXECUÇÃO FINALIZADA COM SUCESSO")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ ERRO durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())