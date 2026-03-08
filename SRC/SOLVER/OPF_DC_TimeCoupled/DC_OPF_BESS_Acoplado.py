#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classe principal do modelo de otimização multi-período acoplado (DC OPF) com suporte a baterias e perdas iterativas.
Utiliza módulos separados para cada grupo de restrições.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pyomo.environ import *
from typing import List, Union

from SOLVER.OPF_DC_TimeCoupled.RES.BatteryConstraintsTime import BatteryConstraintsTime
from SOLVER.OPF_DC_TimeCoupled.RES.ThermalGeneratorConstraintsTime import ThermalGeneratorConstraints
from SOLVER.OPF_DC_TimeCoupled.RES.WindGeneratorConstraintsTime import WindGeneratorConstraints
from SOLVER.OPF_DC_TimeCoupled.RES.EletricConstraintsTime import ElectricConstraints

from DB.DBmodel_OPF import TimeCoupledOPFResult, TimeCoupledOPFSnapshotResult

class TimeCoupledOPFModel:
    """
    Modelo de otimização multi‑período com acoplamento temporal integrado.
    As variáveis são indexadas no tempo (períodos = n_dias * n_horas).
    Delega a construção de restrições para módulos específicos.
    """

    def __init__(self,
                 sistema,
                 n_horas=24,
                 n_dias=1,
                 db_handler=None,
                 considerar_perdas=True,
                 dia_inicial=0):
        self.sistema = sistema
        self.n_horas = n_horas
        self.n_dias = n_dias
        self.horizon_time = n_horas * n_dias
        self.db_handler = db_handler
        self.considerar_perdas = considerar_perdas
        self.dia_inicial = dia_inicial

        self.model = None
        self._solved = False
        self._raw_results = None
        self._battery_list = []
        self._battery_index = {}
        self.battery_data = None
        self._perdas_calculadas = None
        self._soc_inicial_list = []
        self._soc_final_list = []
        self._fator_carga = None
        self._fator_vento = None

    # -------------------------------------------------------------------------
    # Construção do modelo
    # -------------------------------------------------------------------------
    def build(self,
              fator_carga: np.ndarray = None,
              fator_vento: np.ndarray = None,
              soc_inicial: Union[float, List[float]] = 0.5,
              soc_final: Union[float, List[float]] = 0.5):
        """
        Constrói o modelo monolítico com variáveis temporais.
        soc_inicial e soc_final devem ser frações da capacidade (0 a 1).
        """
        self.model = ConcreteModel()
        m = self.model
        s = self.sistema
        T = self.horizon_time

        # Conjuntos base
        m.T = RangeSet(0, T-1)
        m.BUSES = Set(initialize=range(s.NBAR))
        m.LINES = Set(initialize=range(s.NLIN))

        # ==================== Processamento dos fatores de carga e vento ====================
        # --- Fator de carga ---
        if fator_carga is not None:
            if fator_carga.ndim == 3 and fator_carga.shape[0] == self.n_dias and fator_carga.shape[1] == self.n_horas:
                fator_carga = fator_carga.reshape((T, s.NBAR))
            elif fator_carga.ndim == 2 and fator_carga.shape[0] == self.n_dias and fator_carga.shape[1] == self.n_horas:
                fator_carga = np.repeat(fator_carga.reshape((T, 1)), s.NBAR, axis=1)
            elif fator_carga.ndim == 2 and fator_carga.shape[1] != s.NBAR:
                fator_carga = np.repeat(fator_carga, s.NBAR, axis=1)
            elif fator_carga.ndim == 1:
                fator_carga = np.tile(fator_carga.reshape(-1, 1), (1, s.NBAR))
        else:
            fator_carga = np.ones((T, s.NBAR))
        if fator_carga.shape != (T, s.NBAR):
            raise ValueError(f"fator_carga shape {fator_carga.shape} != {(T, s.NBAR)}")

        # --- Fator de vento ---
        if fator_vento is not None:
            if fator_vento.ndim == 3 and fator_vento.shape[0] == self.n_dias and fator_vento.shape[1] == self.n_horas:
                fator_vento = fator_vento.reshape((T, s.NGER_EOL))
            elif fator_vento.ndim == 2 and fator_vento.shape[0] == self.n_dias and fator_vento.shape[1] == self.n_horas:
                if s.NGER_EOL > 0:
                    fator_vento = np.repeat(fator_vento.reshape((T, 1)), s.NGER_EOL, axis=1)
                else:
                    fator_vento = np.ones((T, 0))
            elif fator_vento.ndim == 1:
                if s.NGER_EOL > 0:
                    fator_vento = np.tile(fator_vento.reshape(-1, 1), (1, s.NGER_EOL))
                else:
                    fator_vento = np.ones((T, 0))
            elif fator_vento.ndim == 2 and fator_vento.shape[1] != s.NGER_EOL:
                if s.NGER_EOL > 0:
                    fator_vento = np.repeat(fator_vento, s.NGER_EOL, axis=1)
                else:
                    fator_vento = np.ones((T, 0))
        else:
            fator_vento = np.ones((T, s.NGER_EOL)) if s.NGER_EOL > 0 else np.ones((T, 0))
        if fator_vento.shape != (T, max(s.NGER_EOL, 0)):
            raise ValueError(f"fator_vento shape {fator_vento.shape} != {(T, s.NGER_EOL)}")

        # ==================== Conjuntos de geradores ====================
        m.CONV_GENERATORS = Set(initialize=range(s.NGER_CONV))
        if s.NGER_EOL > 0:
            m.WIND_GENERATORS = Set(initialize=range(s.NGER_EOL))
        else:
            m.WIND_GENERATORS = Set(initialize=[])

        # ==================== Baterias ====================
        tem_bateria = hasattr(s, 'BARRAS_COM_BATERIA') and len(s.BARRAS_COM_BATERIA) > 0
        if tem_bateria:
            self._battery_list = list(s.BARRAS_COM_BATERIA)
            self._battery_index = {b: i for i, b in enumerate(self._battery_list)}
            m.BATTERIES = Set(initialize=self._battery_list)

            # Garantir que os arrays existam (já devem existir no SistemaLoader)
            if not hasattr(s, 'BATTERY_CAPACITY'):
                s.BATTERY_CAPACITY = np.zeros(s.NBAR)
            if not hasattr(s, 'BATTERY_MIN_SOC'):
                s.BATTERY_MIN_SOC = np.zeros(s.NBAR)
            if not hasattr(s, 'BATTERY_POWER_LIMIT'):
                s.BATTERY_POWER_LIMIT = np.zeros(s.NBAR)
            if not hasattr(s, 'BATTERY_POWER_OUT'):
                s.BATTERY_POWER_OUT = np.zeros(s.NBAR)
            if not hasattr(s, 'BATTERY_CHARGE_EFF'):
                s.BATTERY_CHARGE_EFF = 0.9
            if not hasattr(s, 'BATTERY_DISCHARGE_EFF'):
                s.BATTERY_DISCHARGE_EFF = 0.9

            # Calcular SOC inicial absoluto (pu) a partir da fração
            if isinstance(soc_inicial, (int, float)):
                soc_inicial_frac = [soc_inicial] * len(self._battery_list)
            else:
                soc_inicial_frac = soc_inicial

            if not hasattr(s, 'SOC_inicial'):
                s.SOC_inicial = np.zeros(s.NBAR)
            for i, b in enumerate(self._battery_list):
                s.SOC_inicial[b] = soc_inicial_frac[i] * s.BATTERY_CAPACITY[b]

            # SOC final (opcional)
            if soc_final is not None:
                if isinstance(soc_final, (int, float)):
                    soc_final_frac = [soc_final] * len(self._battery_list)
                else:
                    soc_final_frac = soc_final
                if not hasattr(s, 'SOC_final'):
                    s.SOC_final = np.zeros(s.NBAR)
                for i, b in enumerate(self._battery_list):
                    s.SOC_final[b] = soc_final_frac[i] * s.BATTERY_CAPACITY[b]
            else:
                if hasattr(s, 'SOC_final'):
                    delattr(s, 'SOC_final')

            # Guardar listas para extração posterior (valores absolutos)
            self._soc_inicial_list = [s.SOC_inicial[b] for b in self._battery_list]
            self._soc_final_list = [s.SOC_final[b] for b in self._battery_list] if hasattr(s, 'SOC_final') else []
        else:
            m.BATTERIES = Set(initialize=[])
            self._battery_list = []

        # ==================== Parâmetros de carga e geração eólica disponível ====================
        m.PLOAD = Param(m.T, m.BUSES, mutable=True, initialize=0.0, within=Reals)
        if s.NGER_EOL > 0:
            m.PGWIND_AVAIL = Param(m.T, m.WIND_GENERATORS, mutable=True, initialize=0.0, within=Reals)

        for t in m.T:
            for b in m.BUSES:
                m.PLOAD[t, b] = s.PLOAD[b] * fator_carga[t, b]
            if s.NGER_EOL > 0:
                for w in m.WIND_GENERATORS:
                    m.PGWIND_AVAIL[t, w] = s.PGWIND_disponivel[w] * fator_vento[t, w]

        # ==================== Variáveis básicas ====================
        m.PGER = Var(m.T, m.CONV_GENERATORS, within=NonNegativeReals)
        if s.NGER_EOL > 0:
            m.PGWIND = Var(m.T, m.WIND_GENERATORS, within=NonNegativeReals)
            m.CURTAILMENT = Var(m.T, m.WIND_GENERATORS, within=NonNegativeReals)
        m.DEFICIT = Var(m.T, m.BUSES, within=NonNegativeReals)
        m.V = Var(m.T, m.BUSES, within=NonNegativeReals, bounds=(0.95, 1.05))
        m.ANG = Var(m.T, m.BUSES, within=Reals, bounds=(-3.14, 3.14))
        m.FLUXO_LIN = Var(m.T, m.LINES, within=Reals)

        for t in m.T:
            for b in m.BUSES:
                m.V[t, b].fix(1.0)
            m.ANG[t, s.slack_idx].fix(0.0)

        if self.considerar_perdas:
            m.PERDAS_BARRA = Param(m.T, m.BUSES, mutable=True, initialize=0.0, within=Reals)
            m.PERDAS_LINHA = Param(m.T, m.LINES, mutable=True, initialize=0.0, within=Reals)

        # ==================== Adicionar restrições ====================
        self._add_all_constraints()
        self.add_FOB()

        self._fator_carga = fator_carga
        self._fator_vento = fator_vento
        self._solved = False

    def _map_battery_data(self, attr_name, default=None):
        """Método auxiliar para mapear dados de bateria (fallback, não usado atualmente)."""
        s = self.sistema
        if not hasattr(s, 'BARRAS_COM_BATERIA') or not hasattr(s, attr_name):
            return {}
        valores = getattr(s, attr_name)
        if len(valores) == s.NBAR:
            return {b: valores[b] for b in s.BARRAS_COM_BATERIA}
        else:
            return {b: valores[i] for i, b in enumerate(s.BARRAS_COM_BATERIA)}

    def _add_all_constraints(self):
        ThermalGeneratorConstraints.add_constraints(self.model, self.sistema)
        WindGeneratorConstraints.add_constraints(self.model, self.sistema)
        if hasattr(self.model, 'BATTERIES') and len(self.model.BATTERIES) > 0:
            BatteryConstraintsTime.add_constraints(self.model, self.sistema)
        ElectricConstraints.add_constraints(self.model, self.sistema, considerar_perdas=self.considerar_perdas)

    def add_FOB(self):
        from SOLVER.OPF_DC_TimeCoupled.FOB import EconomicDispatchTime
        EconomicDispatchTime.ObjectiveFunction.add_objective(self.model, self.sistema)

    # -------------------------------------------------------------------------
    # Métodos para perdas iterativas
    # -------------------------------------------------------------------------
    def calculate_losses(self) -> np.ndarray:
        s = self.sistema
        T = self.horizon_time
        perdas_barra = np.zeros((T, s.NBAR))
        m = self.model
        for t in range(T):
            for e in m.LINES:
                i = s.line_fr[e]
                j = s.line_to[e]
                fluxo_val = value(m.FLUXO_LIN[t, e]) if m.FLUXO_LIN[t, e].value is not None else 0.0
                r = s.r_line[e]
                perdas_linha = r * (fluxo_val ** 2)
                if self.considerar_perdas:
                    m.PERDAS_LINHA[t, e] = perdas_linha
                perdas_barra[t, i] += perdas_linha / 2
                perdas_barra[t, j] += perdas_linha / 2
        self._perdas_calculadas = perdas_barra
        return perdas_barra

    def update_losses(self, perdas_barra: np.ndarray):
        if not self.considerar_perdas:
            return
        m = self.model
        for t in m.T:
            for b in m.BUSES:
                if t < perdas_barra.shape[0] and b < perdas_barra.shape[1]:
                    m.PERDAS_BARRA[t, b] = perdas_barra[t, b]

    def solve_iterative(self, solver_name='glpk', tol=1e-10, max_iter=50, **solver_args):
        from pyomo.opt import TerminationCondition
        if not self.considerar_perdas:
            return self.solve(solver_name, **solver_args)
        m = self.model
        s = self.sistema
        T = self.horizon_time
        ang_prev = np.zeros((T, s.NBAR))
        for it in range(max_iter):
            raw = self.solve(solver_name, **solver_args)
            if raw.solver.termination_condition != TerminationCondition.optimal:
                print(f"Iteração {it+1}: solução não ótima ({raw.solver.termination_condition}).")
                break
            ang_curr = np.array([[value(m.ANG[t, b]) for b in m.BUSES] for t in range(T)])
            perdas_barra = self.calculate_losses()
            self.update_losses(perdas_barra)
            diff = np.max(np.abs(ang_curr - ang_prev))
            print(f"Iteração {it+1}: diff = {diff:.6f}")
            if diff < tol:
                print(f"Convergência alcançada na iteração {it+1}.")
                break
            ang_prev = ang_curr.copy()
            if it == max_iter - 1:
                print(f"Atenção: número máximo de iterações ({max_iter}) atingido. diff = {diff:.6f}")
        return raw

    def solve(self, solver_name='glpk', **solver_args):
        if self.model is None:
            raise RuntimeError("Modelo não construído. Chame build() primeiro.")
        solver = SolverFactory(solver_name)
        self._raw_results = solver.solve(self.model, tee=False, **solver_args)
        self._solved = True
        return self._raw_results

    # -------------------------------------------------------------------------
    # Extração de resultados
    # -------------------------------------------------------------------------
    def extract_results(self) -> TimeCoupledOPFResult:
        if not self._solved or self.model is None:
            raise RuntimeError("Modelo não resolvido. Execute solve() primeiro.")

        m = self.model
        s = self.sistema
        T = self.horizon_time
        snapshots = []
        dias_nomes = ["domingo", "segunda", "terça", "quarta", "quinta", "sexta", "sábado"]

        for t in range(T):
            dia = t // self.n_horas
            hora = t % self.n_horas
            dia_semana = ((self.dia_inicial + dia) % 7)+1
            dia_semana_nome = dias_nomes[dia_semana-1]

            try:
                # --- Extrair PLOAD por barra ---
                PLOAD_vals = [value(m.PLOAD[t, b]) for b in m.BUSES]

                PGER = [value(m.PGER[t, g]) for g in m.CONV_GENERATORS]

                if s.NGER_EOL > 0:
                    PGWIND_disponivel = [value(m.PGWIND_AVAIL[t, w]) for w in m.WIND_GENERATORS]
                    PGWIND = [value(m.PGWIND[t, w]) for w in m.WIND_GENERATORS]
                    CURTAILMENT = [value(m.CURTAILMENT[t, w]) for w in m.WIND_GENERATORS]
                else:
                    PGWIND_disponivel = PGWIND = CURTAILMENT = []

                DEFICIT = [value(m.DEFICIT[t, b]) for b in m.BUSES]

                # --- Baterias: inicializar listas com zeros para todas as barras ---
                SOC_init = [0.0] * s.NBAR
                SOC_atual = [0.0] * s.NBAR
                BESS_operation = [0.0] * s.NBAR

                if hasattr(m, 'BATTERIES') and len(m.BATTERIES) > 0:
                    for b in m.BATTERIES:
                        # b é o índice da barra (global)
                        if t == 0:
                            idx = self._battery_index.get(b, None)
                            if idx is not None and idx < len(self._soc_inicial_list):
                                soc_init_val = self._soc_inicial_list[idx]
                            else:
                                soc_init_val = 0.0
                        else:
                            soc_init_val = value(m.SOC[t-1, b]) if hasattr(m, 'SOC') and m.SOC[t-1, b].value is not None else 0.0
                        SOC_init[b] = soc_init_val

                        if hasattr(m, 'SOC'):
                            SOC_atual[b] = value(m.SOC[t, b])
                        else:
                            SOC_atual[b] = 0.0

                        charge = value(m.CHARGE[t, b]) if hasattr(m, 'CHARGE') and m.CHARGE[t, b].value is not None else 0.0
                        discharge = value(m.DISCHARGE[t, b]) if hasattr(m, 'DISCHARGE') and m.DISCHARGE[t, b].value is not None else 0.0
                        BESS_operation[b] = charge - discharge

                V = [value(m.V[t, b]) for b in m.BUSES]
                ANG = [value(m.ANG[t, b]) for b in m.BUSES]
                FLUXO_LIN = [value(m.FLUXO_LIN[t, e]) for e in m.LINES]

                CUSTO = [DEFICIT[b] * s.CPG_DEFICIT for b in m.BUSES] if hasattr(s, 'CPG_DEFICIT') else [0.0] * s.NBAR
                CMO = 0.0
                if hasattr(m, 'dual') and hasattr(m, 'PowerBalance'):
                    try:
                        CMO = m.dual[m.PowerBalance[t, s.slack_idx]]
                    except:
                        CMO = 0.0

                if self.considerar_perdas and self._perdas_calculadas is not None:
                    PERDAS_BARRA = self._perdas_calculadas[t, :].tolist()
                else:
                    PERDAS_BARRA = [0.0] * s.NBAR

                snapshots.append(TimeCoupledOPFSnapshotResult(
                    dia=dia,
                    dia_semana=dia_semana,
                    hora=hora,
                    sucesso=True,
                    PLOAD=PLOAD_vals,
                    PGER=PGER,
                    PGWIND_disponivel=PGWIND_disponivel,
                    PGWIND=PGWIND,
                    CURTAILMENT=CURTAILMENT,
                    SOC_init=SOC_init,
                    BESS_operation=BESS_operation,
                    SOC_atual=SOC_atual,
                    DEFICIT=DEFICIT,
                    V=V,
                    ANG=ANG,
                    FLUXO_LIN=FLUXO_LIN,
                    CUSTO=CUSTO,
                    CMO=[CMO],
                    PERDAS_BARRA=PERDAS_BARRA,
                    tempo_execucao=self._raw_results.solver.time if hasattr(self._raw_results.solver, 'time') else 0.0,
                    dia_semana_nome=dia_semana_nome
                ))

            except Exception as e:
                snapshots.append(TimeCoupledOPFSnapshotResult(
                    dia=dia,
                    hora=hora,
                    sucesso=False,
                    mensagem=str(e),
                    dia_semana=dia_semana,
                    dia_semana_nome=dia_semana_nome
                ))

        sucesso_global = all(s.sucesso for s in snapshots)
        return TimeCoupledOPFResult(snapshots=snapshots,
                                    sucesso_global=sucesso_global,
                                    mensagem_global="OK" if sucesso_global else "Falhas")
    # -------------------------------------------------------------------------
    # Método de conveniência para executar todo o processo
    # -------------------------------------------------------------------------
    def solve_multiday(self,
                       solver_name='glpk',
                       fator_carga=None,
                       fator_vento=None,
                       soc_inicial=0.5,
                       soc_final=0.5,
                       cen_id=None,
                       tol=1e-10,
                       max_iter=50):
        
        self.build(fator_carga, fator_vento, soc_inicial, soc_final)

        if self.considerar_perdas:
            raw = self.solve_iterative(solver_name, tol=tol, max_iter=max_iter)
        else:
            raw = self.solve(solver_name)

        if self.db_handler is not None and cen_id is not None:
            resultados = self.extract_results()
            for snap in resultados.snapshots:
                dia_str = f"{snap.dia+1}"

                self.db_handler.save_hourly_result(
                    resultado=snap,
                    sistema=self.sistema,
                    hora=snap.hora,
                    solver_name=solver_name,
                    dia=dia_str,
                    cen_id=cen_id
                )

        return raw


if __name__ == "__main__":
    import sys
    import os
    import numpy as np
    import pandas as pd
    from datetime import datetime

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from UTILS.SystemLoader import SistemaLoader
    from DB.DBhandler_OPF import OPF_DBHandler

    print("=" * 70)
    print("SIMULAÇÃO COM MODELO INTEGRADO NO TEMPO (DC OPF) - COM PERDAS ITERATIVAS")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Carregar sistema
    # -------------------------------------------------------------------------
    print("\n1. Carregando dados do sistema...")
    json_path = "DATA/input/3barras_BASE.json"
    if not os.path.exists(json_path):
        print(f"ERRO: Arquivo não encontrado: {json_path}")
        sys.exit(1)

    sistema = SistemaLoader(json_path)
    print(f"   ✓ Sistema carregado: {json_path}")
    print(f"   ✓ Potência base: {sistema.SB:.1f} MVA")
    print(f"   ✓ Barras: {sistema.NBAR}")
    print(f"   ✓ Linhas: {sistema.NLIN}")
    print(f"   ✓ Geradores convencionais: {sistema.NGER_CONV}")
    print(f"   ✓ Geradores eólicos: {sistema.NGER_EOL}")

    # -------------------------------------------------------------------------
    # 2. Parâmetros da simulação
    # -------------------------------------------------------------------------
    n_dias = 30
    n_horas = 24
    T = n_dias * n_horas
    print(f"\n2. Simulando {n_dias} dias x {n_horas} horas = {T} períodos.")

    # -------------------------------------------------------------------------
    # 3. Definir SOC inicial e final como FRAÇÕES da capacidade
    # -------------------------------------------------------------------------
    SOC_inicial = 0.5   # 50% da capacidade
    SOC_final = 0.5

    # -------------------------------------------------------------------------
    # 4. Configurar banco de dados
    # -------------------------------------------------------------------------
    print("\n5. Configurando banco de dados...")
    db_handler = OPF_DBHandler('DATA/output/resultados_PL_acoplado.db')
    db_handler.create_tables()
    cen_id = datetime.now().strftime('%Y%m%d%H%M%S')
    print(f"   ✓ Cenário ID: {cen_id}")

    # -------------------------------------------------------------------------
    # 5. Criar modelo
    # -------------------------------------------------------------------------
    modelo = TimeCoupledOPFModel(
        sistema=sistema,
        n_horas=n_horas,
        n_dias=n_dias,
        db_handler=db_handler,
        considerar_perdas=True,
        dia_inicial=0
    )

    # -------------------------------------------------------------------------
    # 6. Obter fatores de carga
    # -------------------------------------------------------------------------
    from UTILS.EvaluateFactors import EvaluateFactors

    avaliador = EvaluateFactors(
        sistema=sistema,
        n_dias=n_dias,
        n_horas=n_horas,
        carga_incerteza=0.2,
        vento_variacao=0.1,
        seed=42
    )

    fatores_carga, fatores_vento = avaliador.gerar_tudo()

    print("\n6. Resolvendo modelo integrado com perdas iterativas...")
    raw = modelo.solve_multiday(
        solver_name='glpk',
        fator_carga=fatores_carga,
        fator_vento=fatores_vento,
        soc_inicial=SOC_inicial,
        soc_final=SOC_final,
        cen_id=cen_id,
        tol=1e-4,
        max_iter=20
    )

    status = raw.solver.termination_condition
    print(f"\nStatus da solução: {status}")

    # -------------------------------------------------------------------------
    # 7. Extrair e exibir resumo
    # -------------------------------------------------------------------------
    resultados = modelo.extract_results()
    print(f"\nSucesso global: {resultados.sucesso_global}")
    print(f"Snapshots extraídos: {len(resultados.snapshots)}")

    custo_total = sum(sum(snap.CUSTO) for snap in resultados.snapshots if snap.sucesso)
    print(f"Custo total aproximado (apenas déficit): {custo_total:.2f} $")

    if hasattr(modelo.model, 'BATTERIES') and len(modelo.model.BATTERIES) > 0:
        prim_batt = list(modelo.model.BATTERIES)[0]
        soc_final_val = value(modelo.model.SOC[T-1, prim_batt]) if hasattr(modelo.model, 'SOC') else 0.0
        print(f"SOC final da bateria {prim_batt}: {soc_final_val:.3f} MWh")

    print("\n" + "=" * 70)
    print("EXECUÇÃO CONCLUÍDA")
    print("=" * 70)