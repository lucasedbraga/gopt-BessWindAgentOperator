#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal para OPF com RNAs especialistas.
Versão com todas as previsões da RNA fixadas como parâmetros.
CORRIGIDO: salva valores em MW, garantindo relação PGWIND_disponivel = PGWIND + CURTAILMENT.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from UTILS.SystemLoader import SistemaLoader
from DB.DBhandler_OPF import OPF_DBHandler
from pyomo.environ import *
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import joblib

# =============================================================================
# Dataclass de resultado
# =============================================================================
@dataclass
class OPFResult:
    sucesso: bool
    PLOAD: List[float] = field(default_factory=list)
    PGER: List[float] = field(default_factory=list)
    PGWIND_disponivel: List[float] = field(default_factory=list)
    PGWIND: List[float] = field(default_factory=list)
    CURTAILMENT: List[float] = field(default_factory=list)
    SOC_init: List[float] = field(default_factory=list)
    BESS_operation: List[float] = field(default_factory=list)
    SOC_atual: List[float] = field(default_factory=list)
    DEFICIT: List[float] = field(default_factory=list)
    V: List[float] = field(default_factory=list)
    ANG: List[float] = field(default_factory=list)
    FLUXO_LIN: List[float] = field(default_factory=list)
    CUSTO: List[float] = field(default_factory=list)
    CMO: List[float] = field(default_factory=list)
    PERDAS_BARRA: List[float] = field(default_factory=list)
    mensagem: str = ""
    dia_semana: Optional[int] = None
    timestamp: Optional[datetime] = None
    tempo_execucao: float = 0.0

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# =============================================================================
# Modelo DC OPF com RNA
# =============================================================================
@dataclass
class DC_OPF_RNA_Model:
    def __init__(self):
        self.model = None
        self.sistema = None
        self.considerar_perdas = True
        self.pipeline = None

    def build(self, sistema, df_measures, pipeline, fator_carga,
              pgwind_disponivel_por_gerador, soc_atual_dict, considerar_perdas=False):
        """
        Constrói o modelo de otimização.
        - pgwind_disponivel_por_gerador: lista com a disponibilidade real (MW) para cada gerador eólico
        - soc_atual_dict: dicionário com SOC inicial por barra (usado apenas para feature, não para restrição)
        """
        self.sistema = sistema
        self.considerar_perdas = considerar_perdas
        self.pipeline = pipeline

        m = ConcreteModel()
        self.model = m
        s = self.sistema

        m.clear()

        # === CONJUNTOS ===
        m.BUSES = Set(initialize=range(s.NBAR))
        m.CONV_GENERATORS = Set(initialize=range(s.NGER_CONV))
        m.LINES = Set(initialize=range(s.NLIN))

        if s.NGER_EOL > 0:
            m.WIND_GENERATORS = Set(initialize=range(s.NGER_EOL))
        else:
            m.WIND_GENERATORS = Set(initialize=[])

        if hasattr(s, 'BATTERIES') and len(s.BATTERIES) > 0:
            m.BATTERIES = Set(initialize=s.BATTERIES)
        else:
            m.BATTERIES = Set(initialize=[])

        # === PARÂMETROS (carga) ===
        # Carga base (pu) convertida para MW
        m.PLOAD_BASE = Param(m.BUSES, initialize=lambda m, b: s.PLOAD[b] * s.SB)
        m.FATOR_CARGA = Param(initialize=fator_carga)

        # === PARÂMETROS para eólica e bateria (fixados pela RNA) ===
        # Disponibilidade eólica (já conhecida) - em MW
        m.PGWIND_disponivel = Param(m.WIND_GENERATORS, initialize=lambda m, g: pgwind_disponivel_por_gerador[g])

        # Previsão da RNA
        feature_cols = ['BAR_id', 'BESS_init_cenario', 'PGWIND_disponivel_cenario', 'PGER_CONV_total_result']
        # Garantir que df_measures contém essas colunas
        df_measures = df_measures[feature_cols].dropna().sort_values('BAR_id').reset_index(drop=True)

        if self.pipeline is not None:
            X = df_measures[feature_cols].copy()
            pred = self.pipeline.predict(X)
        else:
            pred = np.zeros((len(df_measures), 2))

        # Mapear predições por BAR_id
        pred_dict = {}
        for idx, row in df_measures.iterrows():
            bar_id = int(row['BAR_id'])
            pred_dict[bar_id] = {
                'curtailment': pred[idx, 0],
                'bess_op': pred[idx, 1]
            }

        # Parâmetros: curtailment e operação da bateria
        barra_por_gerador = getattr(s, 'bus_wind', list(range(s.NGER_EOL)))
        m.CURTAILMENT = Param(m.WIND_GENERATORS, initialize=lambda m, g: 0.0, mutable=True)
        for g in m.WIND_GENERATORS:
            barra = barra_por_gerador[g]
            bar_id = barra + 1
            if bar_id in pred_dict:
                curt_pred = pred_dict[bar_id]['curtailment']
                # Garantir que não ultrapasse a disponibilidade
                curt_final = max(0, min(curt_pred, pgwind_disponivel_por_gerador[g]))
                m.CURTAILMENT[g] = curt_final
            else:
                m.CURTAILMENT[g] = 0.0

        # PGWIND resultante = disponivel - curtailment (também como Parâmetro)
        m.PGWIND = Param(m.WIND_GENERATORS, initialize=lambda m, g: m.PGWIND_disponivel[g] - m.CURTAILMENT[g])

        # Operação da bateria (fixada)
        m.BatteryOperation = Param(m.BATTERIES, initialize=0.0, mutable=True)
        for b in m.BATTERIES:
            bar_id = b + 1
            if bar_id in pred_dict:
                m.BatteryOperation[b] = pred_dict[bar_id]['bess_op']
            else:
                m.BatteryOperation[b] = 0.0

        # === VARIÁVEIS ===
        m.PGER = Var(m.CONV_GENERATORS, within=NonNegativeReals)                # MW
        m.DEFICIT = Var(m.BUSES, within=NonNegativeReals)                       # MW

        # Variáveis elétricas (pu)
        m.V = Var(m.BUSES, within=NonNegativeReals, bounds=(0.95, 1.05))
        m.ANG = Var(m.BUSES, within=Reals, bounds=(-3.14, 3.14))
        m.FLUXO_LIN = Var(m.LINES, within=Reals)

        # Parâmetros para perdas (se considerar)
        if considerar_perdas:
            if not hasattr(m, 'PERDAS_LINHA'):
                m.PERDAS_LINHA = Param(m.LINES, mutable=True, initialize=0.0)
            if not hasattr(m, 'PERDAS_BARRA'):
                m.PERDAS_BARRA = Param(m.BUSES, mutable=True, initialize=0.0)

        # Fixações iniciais
        for bus in m.BUSES:
            m.V[bus].fix(1.0)
        m.ANG[s.slack_idx].fix(0.0)

        # --- Restrições ---
        self.add_constraints()

        # --- Objetivo ---
        self.add_objective()

        return m

    def add_constraints(self):
        from SOLVER.OPF_DC_Snapshot.RES.EletricConstraints import DCElectricConstraints
        from SOLVER.OPF_DC_Snapshot.RES.TermicGeneratorConstraint import TermicGeneratorConstraints

        m = self.model
        s = self.sistema

        # Limites dos geradores convencionais
        TermicGeneratorConstraints.add_generator_limits_constraints(m, s)

        # Déficit
        DCElectricConstraints.add_deficit_constraints(m, s)

        # Fluxo nas linhas
        DCElectricConstraints.add_line_flow_constraints(m, s)

        # Balanço de potência personalizado (usa os parâmetros fixos de eólica e bateria)
        def power_balance_rule(m, b):
            carga = m.PLOAD_BASE[b] * m.FATOR_CARGA
            ger_conv = sum(m.PGER[g] for g in m.CONV_GENERATORS if s.BARPG_CONV[g] == b)
            ger_eol = sum(m.PGWIND[g] for g in m.WIND_GENERATORS if s.BARPG_EOL[g] == b)
            bat = sum(m.BatteryOperation[bat] for bat in m.BATTERIES if bat == b)
            deficit = m.DEFICIT[b]
            fluxo_inj = sum(m.FLUXO_LIN[e] for e in m.LINES if s.line_to[e] == b) - \
                        sum(m.FLUXO_LIN[e] for e in m.LINES if s.line_fr[e] == b)
            perdas = m.PERDAS_BARRA[b] if hasattr(m, 'PERDAS_BARRA') else 0.0
            return ger_conv + ger_eol + bat + fluxo_inj + deficit == carga + perdas
        m.PowerBalance = Constraint(m.BUSES, rule=power_balance_rule)

    def add_objective(self):
        m = self.model
        s = self.sistema

        # Custo dos geradores convencionais
        if hasattr(s, 'CPG') and len(s.CPG) == s.NGER_CONV:
            custo_conv = sum(m.PGER[g] * s.CPG[g] for g in m.CONV_GENERATORS)
        else:
            custo_conv = sum(m.PGER[g] * 10.0 for g in m.CONV_GENERATORS)

        # Penalidade por déficit
        if hasattr(s, 'CPG_DEFICIT'):
            custo_def = sum(m.DEFICIT[b] * s.CPG_DEFICIT for b in m.BUSES)
        else:
            custo_def = sum(m.DEFICIT[b] * 1000.0 for b in m.BUSES)

        m.OBJ = Objective(expr=custo_conv + custo_def, sense=minimize)

    # -------------------------------------------------------------------------
    # Métodos para Perdas
    # -------------------------------------------------------------------------
    def update_losses(self, perdas_barra: np.ndarray):
        if not self.considerar_perdas:
            return
        m = self.model
        for i in m.BUSES:
            if i < len(perdas_barra):
                m.PERDAS_BARRA[i] = perdas_barra[i]

    def calculate_losses(self) -> np.ndarray:
        s = self.sistema
        perdas_barra = np.zeros(s.NBAR)
        for e in self.model.LINES:
            i = s.line_fr[e]
            j = s.line_to[e]
            fluxo_val = value(self.model.FLUXO_LIN[e]) if hasattr(self.model, 'FLUXO_LIN') else 0.0
            r = s.r_line[e]
            perdas_linha = r * (fluxo_val ** 2)
            self.model.PERDAS_LINHA[e] = perdas_linha
            perdas_barra[i] += perdas_linha / 2
            perdas_barra[j] += perdas_linha / 2
        return perdas_barra

    def solve_iterative(self, solver='glpk', tol=1e-4, max_iter=20, **solver_args):
        from pyomo.opt import SolverFactory, TerminationCondition
        import numpy as np

        if not self.considerar_perdas:
            solver_instance = SolverFactory(solver)
            results = solver_instance.solve(self.model, **solver_args)
            return self.extract_results(results)

        m = self.model
        s = self.sistema
        ang_prev = np.zeros(s.NBAR)

        for it in range(max_iter):
            solver_instance = SolverFactory(solver)
            results = solver_instance.solve(m, **solver_args)

            if results.solver.termination_condition != TerminationCondition.optimal:
                return self.extract_results(results)

            ang_curr = np.array([value(m.ANG[b]) for b in m.BUSES])
            perdas_barra = self.calculate_losses()
            self.update_losses(perdas_barra)

            diff = np.max(np.abs(ang_curr - ang_prev))
            if diff < tol:
                print(f"Convergência na iteração {it+1} (diff = {diff:.6f})")
                break
            ang_prev = ang_curr.copy()

        return self.extract_results(results)

    def extract_results(self, results) -> OPFResult:
        m = self.model
        s = self.sistema
        from pyomo.opt import TerminationCondition

        if results.solver.termination_condition != TerminationCondition.optimal:
            return OPFResult(
                sucesso=False,
                PLOAD=[0.0]*s.NBAR,
                PGER=[0.0]*s.NGER_CONV,
                PGWIND_disponivel=[0.0]*s.NGER_EOL,
                PGWIND=[0.0]*s.NGER_EOL,
                CURTAILMENT=[0.0]*s.NGER_EOL,
                SOC_init=[0.0]*s.NBAR,
                BESS_operation=[0.0]*s.NBAR,
                SOC_atual=[0.0]*s.NBAR,
                DEFICIT=[0.0]*s.NBAR,
                V=[0.0]*s.NBAR,
                ANG=[0.0]*s.NBAR,
                FLUXO_LIN=[0.0]*s.NLIN,
                CUSTO=[0.0]*s.NBAR,
                CMO=[0.0]*s.NBAR,
                PERDAS_BARRA=[0.0]*s.NBAR,
                mensagem=f"Solver: {results.solver.termination_condition}",
                timestamp=datetime.now()
            )

        PGER = [value(m.PGER[g]) for g in m.CONV_GENERATORS]

        PGWIND_disponivel = [value(m.PGWIND_disponivel[g]) for g in m.WIND_GENERATORS]
        CURTAILMENT = [value(m.CURTAILMENT[g]) for g in m.WIND_GENERATORS]
        PGWIND = [value(m.PGWIND[g]) for g in m.WIND_GENERATORS]

        BESS_operation = [0.0]*s.NBAR
        if len(m.BATTERIES) > 0:
            for b in m.BATTERIES:
                BESS_operation[b] = value(m.BatteryOperation[b])

        DEFICIT = [value(m.DEFICIT[b]) for b in m.BUSES]
        V = [value(m.V[b]) for b in m.BUSES]
        ANG = [value(m.ANG[b]) for b in m.BUSES]
        FLUXO_LIN = [value(m.FLUXO_LIN[e]) for e in m.LINES]

        # Carga real (MW)
        PLOAD = [value(m.PLOAD_BASE[b] * m.FATOR_CARGA) for b in m.BUSES]

        CMO = 0.0
        if hasattr(m, 'dual') and hasattr(m, 'PowerBalance'):
            try:
                CMO = m.dual[m.PowerBalance[s.slack_idx]]
            except:
                pass

        PERDAS_BARRA = 0.0
        if self.considerar_perdas:
            PERDAS_BARRA = sum(value(m.PERDAS_BARRA[b]) for b in m.BUSES)

        return OPFResult(
            sucesso=True,
            PLOAD=PLOAD,
            PGER=PGER,
            PGWIND_disponivel=PGWIND_disponivel,
            PGWIND=PGWIND,
            CURTAILMENT=CURTAILMENT,
            SOC_init=[0.0]*s.NBAR,
            BESS_operation=BESS_operation,
            SOC_atual=[0.0]*s.NBAR,
            DEFICIT=DEFICIT,
            V=V,
            ANG=ANG,
            FLUXO_LIN=FLUXO_LIN,
            CUSTO=[0.0]*s.NBAR,
            CMO=CMO,
            PERDAS_BARRA=PERDAS_BARRA,
            mensagem="",
            timestamp=datetime.now(),
            tempo_execucao=results.solver.time if hasattr(results.solver, 'time') else 0.0
        )


# =============================================================================
# MAIN – SIMULAÇÃO HORÁRIA
# =============================================================================
def main():
    print("=" * 70)
    print("SISTEMA DE OTIMIZAÇÃO DE FLUXO DE POTÊNCIA (OPF) COM RNAs ESPECIALISTAS")
    print("=" * 70)
    try:
        # ------------------------------------------------------------------
        # 1. CARREGAR SISTEMA
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
        print(f"   ✓ Linhas: {sistema.NLIN}")
        print(f"   ✓ Geradores: {sistema.NGER_CONV} CONVENCIONAIS")
        print(f"   ✓ Geradores eólicos (GWD): {sistema.NGER_EOL} ")
        print(f"   ✓ Baterias: {len(getattr(sistema, 'BATTERIES', []))}")
        print(f"   ✓ Carga total base: {np.sum(sistema.PLOAD):.3f} pu ({np.sum(sistema.PLOAD)*sistema.SB:.1f} MW)")

        # ------------------------------------------------------------------
        # 2. CARREGAR DADOS DE VENTO
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
        # 3. CONFIGURAR SIMULAÇÃO
        # ------------------------------------------------------------------
        n_dias = 1
        n_horas = 24
        print(f"\n3. Configurando simulação: {n_dias} dias, {n_horas} horas/dia")

        # Data base para determinar dia da semana
        data_base = datetime(2026, 3, 9)  # segunda-feira

        # SOC inicial das baterias (apenas para feature, não restrição)
        if hasattr(sistema, 'BATTERY_CAPACITY'):
            if isinstance(sistema.BATTERY_CAPACITY, (list, np.ndarray)):
                soc_inicial_bateria = {b: 0.5 * cap for b, cap in enumerate(sistema.BATTERY_CAPACITY) if b in sistema.BATTERIES}
            else:
                soc_inicial_bateria = {b: 0.5 * sistema.BATTERY_CAPACITY for b in sistema.BATTERIES}
        else:
            soc_inicial_bateria = {b: 0.5 for b in sistema.BATTERIES}

        # Fatores de vento (aleatórios do histórico)
        amostras_vento = df_wind["wind_factor"].sample(n_dias*n_horas, replace=True).values
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

        # ------------------------------------------------------------------
        # 4. PREPARAR ESTRUTURAS DE APOIO
        # ------------------------------------------------------------------
        barra_eol_por_gerador = getattr(sistema, 'bus_wind', [0]*sistema.NGER_EOL)
        barras_bateria = sistema.BATTERIES if hasattr(sistema, 'BATTERIES') else []
        pmax_eol = getattr(sistema, 'PMAX_EOL', [1.0]*sistema.NGER_EOL)

        db_handler = OPF_DBHandler('DATA/output/resultados_OPF_RNA.db')
        db_handler.create_tables()
        cen_id = datetime.now().strftime('%Y%m%d%H%M%S')
        print(f"   ✓ Cenário ID: {cen_id}")

        # ------------------------------------------------------------------
        # 5. SIMULAÇÃO HORÁRIA
        # ------------------------------------------------------------------
        print("\n4. Iniciando simulação horária com RNAs especialistas...")
        soc_atual = soc_inicial_bateria.copy()  # para usar nas features
        resultados_por_snapshot = []

        # Cache de pipelines
        pipeline_cache = {}

        for dia in range(n_dias):
            data_atual = data_base + timedelta(days=dia)
            dia_semana = data_atual.weekday()  # 0 = segunda

            for hora in range(n_horas):
                print(f"\n   Processando Dia {dia+1} (semana {dia_semana}), Hora {hora:02d}:00 ...")

                # Carregar pipeline
                cache_key = (dia_semana, hora)
                if cache_key in pipeline_cache:
                    pipeline = pipeline_cache[cache_key]
                else:
                    possible_dirs = [f"dia_{dia_semana}", f"dia_{dia_semana+1}"]
                    pipeline = None
                    for d in possible_dirs:
                        pipeline_path = f'DATA/output/modelos_especialistas/{d}/hora_{hora:02d}/pipeline.joblib'
                        if os.path.exists(pipeline_path):
                            pipeline = joblib.load(pipeline_path)
                            print(f"   → Pipeline carregado de: {pipeline_path}")
                            break
                    if pipeline is None:
                        print(f"   ⚠ Aviso: modelo não encontrado. Usando fallback (tudo zero).")
                    pipeline_cache[cache_key] = pipeline

                f_v = fator_vento[dia, hora]
                f_c = fator_carga[dia, hora]

                # Disponibilidade eólica por gerador (MW)
                pgwind_disp_por_gerador = [pmax_eol[g] * f_v * sistema.SB for g in range(sistema.NGER_EOL)]

                # Disponibilidade por barra (para features)
                disp_eolica_barra_mw = np.zeros(sistema.NBAR)
                for g in range(sistema.NGER_EOL):
                    barra = barra_eol_por_gerador[g]
                    disp_eolica_barra_mw[barra] += pgwind_disp_por_gerador[g]

                # Montar DataFrame com as features esperadas pela RNA
                rows = []
                for barra in range(sistema.NBAR):
                    soc_val = soc_atual.get(barra, 0.5)
                    if isinstance(soc_val, np.ndarray):
                        soc_val = float(soc_val[0]) if soc_val.size > 0 else 0.5

                    disp_eolica_mw = disp_eolica_barra_mw[barra]
                    pger_barra_mw = 0.0  # placeholder

                    rows.append({
                        'BAR_id': barra + 1,
                        'BESS_init_cenario': soc_val,
                        'PGWIND_disponivel_cenario': disp_eolica_mw,
                        'PGER_CONV_total_result': pger_barra_mw
                    })

                df_measures = pd.DataFrame(rows).sort_values('BAR_id').reset_index(drop=True)
                #TODO: Avaliar SOC atual e SOC final do modelo
                # Construir modelo
                modelo = DC_OPF_RNA_Model()
                modelo.build(
                    sistema=sistema,
                    df_measures=df_measures,
                    pipeline=pipeline,
                    fator_carga=f_c,
                    pgwind_disponivel_por_gerador=pgwind_disp_por_gerador,
                    soc_atual_dict=soc_atual,
                    considerar_perdas=False
                )

                # Resolver
                if modelo.considerar_perdas:
                    resultado = modelo.solve_iterative(solver='glpk', tol=1e-4, max_iter=20)
                else:
                    from pyomo.opt import SolverFactory
                    solver = SolverFactory('glpk')
                    results = solver.solve(modelo.model, tee=False)
                    resultado = modelo.extract_results(results)

                resultado.timestamp = datetime.now()
                resultado.dia_semana = dia_semana
                resultados_por_snapshot.append(resultado)

                db_handler.save_hourly_result(
                    resultado=resultado,
                    sistema=sistema,
                    hora=hora,
                    solver_name='glpk',
                    dia=f"{dia+1}",
                    cen_id=cen_id
                )

                for b in barras_bateria:
                    if b < len(resultado.BESS_operation):
                        op = resultado.BESS_operation[b]
                        soc_atual[b] = soc_atual.get(b, 0.5) - op
                        # Limitar (opcional)
                        cap = sistema.BATTERY_CAPACITY[b] if isinstance(sistema.BATTERY_CAPACITY, (list, np.ndarray)) else sistema.BATTERY_CAPACITY
                        soc_atual[b] = np.clip(soc_atual[b], 0, cap)

                print(f"   ✓ Concluído. Status: {'sucesso' if resultado.sucesso else 'falha'}")
                # # Opcional: imprimir valores para depuração
                # print(f"      PGWIND_disponivel (MW): {resultado.PGWIND_disponivel}")
                # print(f"      PGWIND (MW): {resultado.PGWIND}")
                # print(f"      CURTAILMENT (MW): {resultado.CURTAILMENT}")

        # ------------------------------------------------------------------
        # 6. RESUMO
        # ------------------------------------------------------------------
        print("\n5. Resumo dos resultados:")
        for i, res in enumerate(resultados_por_snapshot):
            dia = i // n_horas
            hora = i % n_horas
            status = "✓" if res.sucesso else "✗"
            print(f"   Dia {dia+1}, Hora {hora:02d}:00 {status}  Custo: {sum(res.CUSTO):.4f}")

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