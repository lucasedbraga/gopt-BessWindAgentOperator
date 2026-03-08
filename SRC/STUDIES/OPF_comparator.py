#!/usr/bin/env python3
"""
Script de comparação entre o modelo DC OPF desacoplado e o modelo com RNA.
Ambos são executados com os mesmos dados de entrada (fatores de vento, carga, SOC inicial).
Resultados são comparados hora a hora, com ênfase em curtailment e operação da bateria.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

# Adiciona diretório SRC ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SRC.UTILS.SystemLoader import SistemaLoader


# =============================================================================
# Função para executar modelo desacoplado e retornar lista de resultados (MW)
# =============================================================================
def run_modelo_desacoplado(sistema, fator_vento, fator_carga, soc_inicial_mwh):
    """
    Executa MultiDayOPFModel e retorna uma lista de dicionários com resultados por hora.
    Os valores são convertidos para MW (multiplicando por sistema.SB).
    """
    from SRC.SOLVER.OPF_DC_Snapshot.OPFDC_Sequential import MultiDayOPFModel

    # --- Configurar SOC inicial no sistema (necessário para o modelo) ---
    if not hasattr(sistema, 'SOC_init'):
        sistema.SOC_init = np.zeros(sistema.NBAR)
    else:
        # Se já existir, garantir que seja um array de tamanho correto
        if len(sistema.SOC_init) != sistema.NBAR:
            sistema.SOC_init = np.zeros(sistema.NBAR)

    barras_bateria = getattr(sistema, 'BATTERIES', [])
    soc_inicial_pu = soc_inicial_mwh / sistema.SB   # converter para pu (energia)
    for b in barras_bateria:
        if 0 <= b < sistema.NBAR:
            sistema.SOC_init[b] = soc_inicial_pu

    n_dias, n_horas = fator_vento.shape
    modelo = MultiDayOPFModel(sistema, n_horas=n_horas, n_dias=n_dias, db_handler=None)
    resultado_global = modelo.solve_multiday_sequencial(
        solver_name='glpk',
        fator_carga=fator_carga,
        fator_vento=fator_vento,
        soc_inicial=soc_inicial_pu   # o método espera SOC em pu (energia)
    )

    resultados_hora = []
    for snap in resultado_global.snapshots:
        # Converter valores para MW (assumindo que os atributos estão em pu)
        pger_mw = [p * sistema.SB for p in snap.PGER] if snap.PGER else []
        pgwind_mw = [w * sistema.SB for w in snap.PGWIND] if hasattr(snap, 'PGWIND') and snap.PGWIND else []
        curt_mw = [c * sistema.SB for c in snap.CURTAILMENT] if hasattr(snap, 'CURTAILMENT') and snap.CURTAILMENT else []
        bess_op_mw = [op * sistema.SB for op in snap.BESS_operation] if hasattr(snap, 'BESS_operation') and snap.BESS_operation else []
        deficit_mw = [d * sistema.SB for d in snap.DEFICIT] if hasattr(snap, 'DEFICIT') and snap.DEFICIT else []
        soc_atual_mwh = [s * sistema.SB for s in snap.SOC_atual] if hasattr(snap, 'SOC_atual') and snap.SOC_atual else []

        resultados_hora.append({
            'dia': snap.dia,
            'hora': snap.hora,
            'sucesso': snap.sucesso,
            'PGER_total_mw': sum(pger_mw),
            'PGER_mw': pger_mw,
            'PGWIND_total_mw': sum(pgwind_mw),
            'CURTAILMENT_total_mw': sum(curt_mw),
            'CURTAILMENT_mw': curt_mw,
            'BESS_operation_mw': bess_op_mw,
            'DEFICIT_total_mw': sum(deficit_mw),
            'SOC_atual_mwh': soc_atual_mwh,
            'custo_total': sum(snap.CUSTO) if hasattr(snap, 'CUSTO') else 0,
            'timestamp': snap.timestamp
        })
    return resultados_hora

# =============================================================================
# Função para executar modelo RNA e retornar lista de resultados (MW)
# =============================================================================
def run_modelo_rna(sistema, fator_vento, fator_carga, soc_inicial_mwh, pipeline):
    """
    Executa DC_OPF_RNA_Model sequencialmente, hora a hora, e retorna lista de resultados.
    """
    n_dias, n_horas = fator_vento.shape

    # Mapeamento de geradores eólicos para barras
    barra_eol_por_gerador = getattr(sistema, 'bus_wind', list(range(sistema.NGER_EOL)))

    # Barras com bateria
    barras_bateria = getattr(sistema, 'BATTERIES', [])

    # Potência máxima eólica por gerador (pu)
    pmax_eol = getattr(sistema, 'PMAX_EOL', [1.0]*sistema.NGER_EOL)

    # Capacidade instalada de geradores convencionais por barra (MW)
    capacidade_conv_por_barra = np.zeros(sistema.NBAR)
    for g in range(sistema.NGER_CONV):
        barra = getattr(sistema, 'bus_gen', [0]*sistema.NGER_CONV)[g]
        capacidade_conv_por_barra[barra] += sistema.PMAX[g] * sistema.SB
    capacidade_conv_total_mw = capacidade_conv_por_barra.sum()

    # SOC atual (MWh) – soc_inicial_mwh já é escalar
    soc_atual = {barra: soc_inicial_mwh for barra in barras_bateria}

    resultados_hora = []

    for dia in range(n_dias):
        for hora in range(n_horas):
            f_v = fator_vento[dia, hora]
            f_c = fator_carga[dia, hora]

            # Disponibilidade eólica por barra (MW)
            disp_eolica_barra_mw = np.zeros(sistema.NBAR)
            for g in range(sistema.NGER_EOL):
                barra = barra_eol_por_gerador[g]
                disp_eolica_barra_mw[barra] += pmax_eol[g] * f_v * sistema.SB

            # Carga total (MW)
            carga_total_mw = np.sum(sistema.PLOAD) * f_c * sistema.SB

            # Geração convencional total necessária (MW) – estimativa simples
            disp_eolica_total_mw = disp_eolica_barra_mw.sum()
            pger_total_mw = max(0.0, carga_total_mw - disp_eolica_total_mw)

            # Montar DataFrame (uma linha por barra)
            rows = []
            for barra in range(sistema.NBAR):
                # SOC da bateria (MWh)
                soc_val = soc_atual.get(barra, 0.0)
                if isinstance(soc_val, np.ndarray):
                    soc_val = float(soc_val[0]) if soc_val.size > 0 else 0.0

                # Disponibilidade eólica na barra (MW)
                disp_eolica_mw = disp_eolica_barra_mw[barra]

                # Geração convencional rateada (MW) – estimativa para a RNA
                if capacidade_conv_total_mw > 0:
                    pger_barra_mw = pger_total_mw * (capacidade_conv_por_barra[barra] / capacidade_conv_total_mw)
                else:
                    pger_barra_mw = 0.0

                rows.append({
                    'data_simulacao': dia + 1,
                    'hora_simulacao': hora,
                    'BESS_init_cenario': soc_val,
                    'PGWIND_disponivel_cenario': disp_eolica_mw,
                    'PGER_CONV_total_result': pger_barra_mw,
                    'BAR_id': barra + 1
                })

            df_measures = pd.DataFrame(rows).sort_values('BAR_id').reset_index(drop=True)

            # Construir modelo RNA (sem perdas)
            modelo_rna = DC_OPF_RNA_Model()
            modelo_rna.build(sistema, df_measures, pipeline, considerar_perdas=False)

            # Fixar disponibilidade eólica real para cada gerador (MW)
            for g in range(sistema.NGER_EOL):
                pgwind_disp_mw = pmax_eol[g] * f_v * sistema.SB
                modelo_rna.model.PGWIND_disponivel[g].fix(pgwind_disp_mw)

            # Resolver
            from pyomo.opt import SolverFactory
            solver = SolverFactory('glpk')
            results = solver.solve(modelo_rna.model, tee=False)
            resultado = modelo_rna.extract_results(results)

            # Atualizar SOC das baterias (resultado já está em MW)
            for barra in barras_bateria:
                if barra < len(resultado.BESS_operation):
                    op_mw = resultado.BESS_operation[barra]
                    delta_energia = op_mw * 1.0   # 1h -> MWh
                    soc_atual[barra] += delta_energia
                    if hasattr(sistema, 'BATTERY_CAPACITY'):
                        cap = sistema.BATTERY_CAPACITY
                        # Garantir que cap seja escalar
                        if isinstance(cap, np.ndarray):
                            cap = cap.item() if cap.size == 1 else cap[0]
                        soc_atual[barra] = max(0.0, min(soc_atual[barra], cap))

            # Guardar resultados (já em MW)
            resultados_hora.append({
                'dia': dia,
                'hora': hora,
                'sucesso': resultado.sucesso,
                'PGER_total_mw': sum(resultado.PGER),
                'PGER_mw': resultado.PGER,
                'PGWIND_total_mw': sum(resultado.PGWIND),
                'CURTAILMENT_total_mw': sum(resultado.CURTAILMENT),
                'CURTAILMENT_mw': resultado.CURTAILMENT,
                'BESS_operation_mw': resultado.BESS_operation,
                'DEFICIT_total_mw': sum(resultado.DEFICIT),
                'SOC_atual_mwh': resultado.SOC_atual,
                'custo_total': sum(resultado.CUSTO),
                'timestamp': resultado.timestamp
            })

    return resultados_hora

# =============================================================================
# Função principal de comparação
# =============================================================================
def main():
    print("=" * 70)
    print("COMPARAÇÃO ENTRE MODELO DESACOPLADO E MODELO COM RNA")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. CARREGAR SISTEMA (fixo para ambos)
    # ------------------------------------------------------------------
    print("\n1. Carregando dados do sistema...")
    json_path = "DATA/input/3barras_BASE.json"
    if not os.path.exists(json_path):
        print(f"ERRO: Arquivo não encontrado: {json_path}")
        return 1

    sistema = SistemaLoader(json_path)
    print(f"   ✓ Sistema carregado: {json_path}")
    print(f"   ✓ Potência base: {sistema.SB:.1f} MVA")
    print(f"   ✓ Barras: {sistema.NBAR}")
    print(f"   ✓ Geradores eólicos: {sistema.NGER_EOL}")
    print(f"   ✓ Baterias: {len(getattr(sistema, 'BATTERIES', []))}")
    print(f"   ✓ Carga total base: {np.sum(sistema.PLOAD):.3f} pu ({np.sum(sistema.PLOAD)*sistema.SB:.1f} MW)")

    # ------------------------------------------------------------------
    # 2. CARREGAR DADOS DE VENTO (base para amostragem)
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
    max_prod = df_wind['Production'].max()
    df_wind['Factor'] = df_wind['Production'] / max_prod if max_prod > 0 else 0
    df_wind = df_wind[['DateTime', 'Factor']].rename(columns={'DateTime': 'timestamp', 'Factor': 'wind_factor'})
    print(f"   ✓ Dados carregados: {len(df_wind)} registros")

    # ------------------------------------------------------------------
    # 3. CONFIGURAR SIMULAÇÃO (parâmetros comuns)
    # ------------------------------------------------------------------
    n_dias = 1
    n_horas = 24
    print(f"\n3. Configurando simulação: {n_dias} dias, {n_horas} horas/dia")

    # Extrair capacidade da bateria de forma robusta
    battery_capacity = getattr(sistema, 'BATTERY_CAPACITY', None)
    if battery_capacity is None:
        battery_capacity = 1.0
    elif isinstance(battery_capacity, (np.ndarray, list, tuple)):
        # Se for coleção, pegar o primeiro elemento (assumindo uma bateria)
        battery_capacity = battery_capacity[0] if len(battery_capacity) > 0 else 1.0
    # Garantir que é float
    battery_capacity = float(battery_capacity)
    soc_inicial_mwh = 0.5 * battery_capacity
    print(f"   ✓ SOC inicial: {soc_inicial_mwh:.2f} MWh (capacidade: {battery_capacity:.2f} MWh)")

    # ------------------------------------------------------------------
    # 4. GERAR FATORES COM SEMENTE FIXA (garantir mesmos números)
    # ------------------------------------------------------------------
    np.random.seed(42)   # semente para reprodutibilidade

    # Fatores de vento (amostragem aleatória)
    amostras_vento = df_wind["wind_factor"].sample(n_dias * n_horas, replace=True).values
    fator_vento = abs(amostras_vento.reshape((n_dias, n_horas)))

    # Perfil horário de carga com incerteza
    nivel_horario = np.array([
        0.6,0.6,0.6,0.6,0.6,0.6,
        0.7,0.8,0.9,1.0,1.0,1.0,
        1.1,1.0,1.0,1.0,1.0,1.1,
        1.2,1.3,1.2,1.1,1.0,0.8
    ])
    gu = 0.2
    lim_inf = nivel_horario - gu/2
    lim_sup = nivel_horario + gu/2
    fator_carga = np.random.uniform(low=lim_inf, high=lim_sup, size=(n_dias, n_horas))

    print(f"   ✓ Fatores de vento (primeiras 5h): {fator_vento[0, :5]}")
    print(f"   ✓ Fatores de carga (primeiras 5h): {fator_carga[0, :5]}")

    # ------------------------------------------------------------------
    # 5. CARREGAR PIPELINE DA RNA (necessário apenas para o modelo RNA)
    # ------------------------------------------------------------------
    pipeline_path = '/home/lucasedbraga/repositorios/ufjf/gopt-BessWindAgentOperator/DATA/output/modelo_unico/pipeline_modelo_unico.joblib'
    pipeline = joblib.load(pipeline_path)
    print("   ✓ Pipeline RNA carregado.")

    # ------------------------------------------------------------------
    # 6. EXECUTAR MODELO DESACOPLADO
    # ------------------------------------------------------------------
    print("\n4. Executando modelo desacoplado...")
    resultados_desacoplado = run_modelo_desacoplado(sistema, fator_vento, fator_carga, soc_inicial_mwh)
    print(f"   ✓ Modelo desacoplado concluído. {len(resultados_desacoplado)} horas processadas.")

    # ------------------------------------------------------------------
    # 7. EXECUTAR MODELO RNA
    # ------------------------------------------------------------------
    print("\n5. Executando modelo com RNA...")
    resultados_rna = run_modelo_rna(sistema, fator_vento, fator_carga, soc_inicial_mwh, pipeline)
    print(f"   ✓ Modelo RNA concluído. {len(resultados_rna)} horas processadas.")

    # ------------------------------------------------------------------
    # 8. COMPARAR RESULTADOS HORA A HORA
    # ------------------------------------------------------------------
    print("\n6. Comparando resultados hora a hora...")
    comparacao = []
    for i in range(len(resultados_desacoplado)):
        d = resultados_desacoplado[i]
        r = resultados_rna[i]

        diff_curtail = r['CURTAILMENT_total_mw'] - d['CURTAILMENT_total_mw']
        diff_bess_total = sum(r['BESS_operation_mw']) - sum(d['BESS_operation_mw'])
        diff_pger = r['PGER_total_mw'] - d['PGER_total_mw']
        diff_custo = r['custo_total'] - d['custo_total']

        comparacao.append({
            'dia': d['dia'],
            'hora': d['hora'],
            'curtail_desacoplado_mw': d['CURTAILMENT_total_mw'],
            'curtail_rna_mw': r['CURTAILMENT_total_mw'],
            'diff_curtail_mw': diff_curtail,
            'bess_op_desacoplado_mw': sum(d['BESS_operation_mw']),
            'bess_op_rna_mw': sum(r['BESS_operation_mw']),
            'diff_bess_mw': diff_bess_total,
            'pger_desacoplado_mw': d['PGER_total_mw'],
            'pger_rna_mw': r['PGER_total_mw'],
            'diff_pger_mw': diff_pger,
            'custo_desacoplado': d['custo_total'],
            'custo_rna': r['custo_total'],
            'diff_custo': diff_custo,
            'sucesso_desacoplado': d['sucesso'],
            'sucesso_rna': r['sucesso']
        })

    df_comp = pd.DataFrame(comparacao)

    # Exibir estatísticas
    print("\n--- Estatísticas das diferenças (RNA - Desacoplado) ---")
    print(f"Média da diferença de curtailment (MW): {df_comp['diff_curtail_mw'].mean():.4f}")
    print(f"Desvio padrão da diferença de curtailment: {df_comp['diff_curtail_mw'].std():.4f}")
    print(f"Média da diferença de operação BESS (MW): {df_comp['diff_bess_mw'].mean():.4f}")
    print(f"Desvio padrão da diferença de operação BESS: {df_comp['diff_bess_mw'].std():.4f}")
    print(f"Média da diferença de geração convencional (MW): {df_comp['diff_pger_mw'].mean():.4f}")
    print(f"Média da diferença de custo: {df_comp['diff_custo'].mean():.4f}")

    # Mostrar primeiras horas
    print("\n--- Primeiras 5 horas da comparação ---")
    print(df_comp[['hora', 'curtail_desacoplado_mw', 'curtail_rna_mw', 'diff_curtail_mw',
                   'bess_op_desacoplado_mw', 'bess_op_rna_mw', 'diff_bess_mw']].head())

    # Salvar CSV
    csv_path = "DATA/output/comparacao_modelos.csv"
    df_comp.to_csv(csv_path, index=False)
    print(f"\n   ✓ Comparação salva em: {csv_path}")

    print("\n" + "=" * 70)
    print("COMPARAÇÃO FINALIZADA")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())