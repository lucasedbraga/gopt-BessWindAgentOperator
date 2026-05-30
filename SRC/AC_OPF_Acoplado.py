#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo de otimização multi‑período acoplado (AC OPF) com Pyomo + IPOPT.
Todas as grandezas em pu, variáveis indexadas por (t, idx).
Utiliza as classes de restrição externas para Pyomo.
"""

import os
import sys
import uuid
import numpy as np
import pyomo.environ as pyo
from typing import List, Union, Optional, Tuple, Dict, Callable
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SOLVER.OPF_AC.RES.BatteryConstraints import BatteryConstraints
from SOLVER.OPF_AC.RES.ThermalGeneratorConstraints import ThermalGeneratorConstraints
from SOLVER.OPF_AC.RES.WindGeneratorConstraints import WindGeneratorConstraints
from SOLVER.OPF_AC.RES.EletricConstraints import ACElectricConstraints
from DB.DBmodel_OPF import TimeCoupledOPFResult, TimeCoupledOPFSnapshotResult


class TimeCoupledOPFModel:
    """
    Modelo de otimização multi‑período acoplado (AC OPF) usando Pyomo.
    """

    def __init__(self,
                 sistema,
                 n_horas: int = 24,
                 n_dias: int = 1,
                 db_handler=None,
                 dia_inicial: int = 0):
        self.sistema = sistema
        self.n_horas = n_horas
        self.n_dias = n_dias
        self.horizon_time = n_horas * n_dias
        self.db_handler = db_handler
        self.dia_inicial = dia_inicial

        self.model = None
        self._solved = False

        # Dicionários de variáveis Pyomo (chave (t, idx))
        self.PGER: Dict[Tuple[int, int], pyo.Var] = {}
        self.QGER: Dict[Tuple[int, int], pyo.Var] = {}
        self.PGWIND: Dict[Tuple[int, int], pyo.Var] = {}
        self.CURTAILMENT: Dict[Tuple[int, int], pyo.Var] = {}
        self.DEFICIT: Dict[Tuple[int, int], pyo.Var] = {}
        self.V: Dict[Tuple[int, int], pyo.Var] = {}          # magnitude
        self.ANG: Dict[Tuple[int, int], pyo.Var] = {}        # ângulo
        self.CHARGE: Dict[Tuple[int, int], pyo.Var] = {}
        self.DISCHARGE: Dict[Tuple[int, int], pyo.Var] = {}
        self.SOC: Dict[Tuple[int, int], pyo.Var] = {}
        self.BatteryOperation: Dict[Tuple[int, int], pyo.Var] = {}

        # Parâmetros (arrays em pu)
        self.PLOAD: Optional[np.ndarray] = None          # (T, n_bus)
        self.QLOAD: Optional[np.ndarray] = None          # (T, n_bus)
        self.PGWIND_AVAIL: Optional[np.ndarray] = None   # (T, n_wind)

        # Matrizes de admitância (calculadas uma vez)
        self.G: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None

        self._battery_list: List[int] = []
        self._battery_index: Dict[int, int] = {}
        self._soc_inicial_list: List[float] = []
        self._soc_final_list: List[float] = []

    # ----------------------------------------------------------------------
    # Construção do modelo
    # ----------------------------------------------------------------------

    def build(self,
              fator_carga: Optional[np.ndarray] = None,
              fator_vento: Optional[np.ndarray] = None,
              soc_inicial: Union[float, List[float]] = 0.5,
              soc_final: Optional[Union[float, List[float]]] = None,
              cost_function: Optional[Callable] = None) -> None:
        s = self.sistema
        T = self.horizon_time

        self._process_fatores(fator_carga, fator_vento)
        self._process_soc(soc_inicial, soc_final)
        self._build_admittance_matrix()

        self.model = pyo.ConcreteModel(name="ACOPF_MultiPeriod")

        # Criar variáveis
        self._create_voltage_vars()
        self._create_angle_vars()
        self._create_thermal_vars()
        self._create_wind_vars()
        self._create_deficit_vars()
        self._create_battery_vars()

        # Criar o objetivo ANTES das restrições
        self.build_objective(cost_function)

        # Adicionar restrições (usando classes externas)
        self._add_all_constraints()

        self._solved = False

    def _process_fatores(self, fator_carga, fator_vento):
        s = self.sistema
        T = self.horizon_time
        if fator_carga is None:
            fc = np.ones((T, s.NBAR))
        else:
            fc = np.asarray(fator_carga)
            if fc.ndim == 3:
                fc = fc.reshape((T, s.NBAR))
            elif fc.ndim == 2:
                if fc.shape[0] == self.n_dias and fc.shape[1] == self.n_horas:
                    fc = np.repeat(fc.reshape((T, 1)), s.NBAR, axis=1)
            elif fc.ndim == 1:
                if fc.size == T:
                    fc = fc[:, np.newaxis] * np.ones((1, s.NBAR))
        self.PLOAD = s.PLOAD[np.newaxis, :] * fc
        # Carga reativa: assume mesmo fator
        if hasattr(s, 'QLOAD'):
            self.QLOAD = s.QLOAD[np.newaxis, :] * fc
        else:
            self.QLOAD = np.zeros_like(self.PLOAD)

        if s.NGER_EOL == 0:
            self.PGWIND_AVAIL = np.zeros((T, 0))
        else:
            if fator_vento is None:
                fv = np.ones((T, s.NGER_EOL))
            else:
                fv = np.asarray(fator_vento)
                if fv.ndim == 3:
                    fv = fv.reshape((T, s.NGER_EOL))
                elif fv.ndim == 2:
                    if fv.shape[0] == self.n_dias and fv.shape[1] == self.n_horas:
                        fv = np.repeat(fv.reshape((T, 1)), s.NGER_EOL, axis=1)
                elif fv.ndim == 1:
                    if fv.size == T:
                        fv = fv[:, np.newaxis] * np.ones((1, s.NGER_EOL))
            self.PGWIND_AVAIL = s.PGMAX_EOL_ORIGINAL[np.newaxis, :] * fv

    def _process_soc(self, soc_inicial, soc_final):
        s = self.sistema
        self._battery_list = list(s.BARRAS_COM_BATERIA)
        self._battery_index = {b: i for i, b in enumerate(self._battery_list)}
        nb = len(self._battery_list)
        if nb == 0:
            return
        if isinstance(soc_inicial, (float, int)):
            soc_ini_frac = [soc_inicial] * nb
        else:
            soc_ini_frac = list(soc_inicial)
        self._soc_inicial_list = [
            soc_ini_frac[i] * s.BATTERY_CAPACITY[b] for i, b in enumerate(self._battery_list)
        ]
        if soc_final is not None:
            if isinstance(soc_final, (float, int)):
                soc_fin_frac = [soc_final] * nb
            else:
                soc_fin_frac = list(soc_final)
            self._soc_final_list = [
                soc_fin_frac[i] * s.BATTERY_CAPACITY[b] for i, b in enumerate(self._battery_list)
            ]
        else:
            self._soc_final_list = []

    def _build_admittance_matrix(self):
        s = self.sistema
        n_bus = s.NBAR
        self.G = np.zeros((n_bus, n_bus))
        self.B = np.zeros((n_bus, n_bus))
        for e in range(s.NLIN):
            i = s.line_fr[e]
            j = s.line_to[e]
            r = s.r_line[e]
            x = s.x_line[e]
            z2 = r*r + x*x
            if z2 == 0:
                continue
            g = r / z2
            b = -x / z2
            self.G[i, i] += g
            self.B[i, i] += b
            self.G[j, j] += g
            self.B[j, j] += b
            self.G[i, j] -= g
            self.B[i, j] -= b
            self.G[j, i] -= g
            self.B[j, i] -= b

    # -------------------- Criação de variáveis Pyomo --------------------
    def _create_voltage_vars(self):
        s = self.sistema
        for t in range(self.horizon_time):
            for b in range(s.NBAR):
                var = pyo.Var(bounds=(0.95, 1.05), initialize=1.0)
                setattr(self.model, f"V_{t}_{b}", var)
                self.V[t, b] = var

    def _create_angle_vars(self):
        s = self.sistema
        for t in range(self.horizon_time):
            for b in range(s.NBAR):
                var = pyo.Var(bounds=(-np.pi, np.pi), initialize=0.0)
                setattr(self.model, f"ANG_{t}_{b}", var)
                self.ANG[t, b] = var
            # Fixa ângulo da barra slack
            setattr(self.model, f"fix_ANG_slack_{t}",
                    pyo.Constraint(expr=self.ANG[t, s.slack_idx] == 0.0))

    def _create_thermal_vars(self):
        s = self.sistema
        for t in range(self.horizon_time):
            for g in range(s.NGER_CONV):
                p_var = pyo.Var(bounds=(s.PGMIN_CONV[g], s.PGMAX_CONV[g]), initialize=0.0)
                q_min = s.QGMIN_CONV[g] if hasattr(s, 'QGMIN_CONV') else -0.5 * s.PGMAX_CONV[g]
                q_max = s.QGMAX_CONV[g] if hasattr(s, 'QGMAX_CONV') else 0.5 * s.PGMAX_CONV[g]
                q_var = pyo.Var(bounds=(q_min, q_max), initialize=0.0)
                setattr(self.model, f"PGER_{t}_{g}", p_var)
                setattr(self.model, f"QGER_{t}_{g}", q_var)
                self.PGER[t, g] = p_var
                self.QGER[t, g] = q_var

    def _create_wind_vars(self):
        s = self.sistema
        if s.NGER_EOL == 0:
            return
        for t in range(self.horizon_time):
            for w in range(s.NGER_EOL):
                avail = self.PGWIND_AVAIL[t, w]
                p_var = pyo.Var(bounds=(0, avail), initialize=0.0)
                c_var = pyo.Var(bounds=(0, avail), initialize=0.0)
                setattr(self.model, f"PGWIND_{t}_{w}", p_var)
                setattr(self.model, f"CURTAIL_{t}_{w}", c_var)
                self.PGWIND[t, w] = p_var
                self.CURTAILMENT[t, w] = c_var

    def _create_deficit_vars(self):
        s = self.sistema
        for t in range(self.horizon_time):
            for b in range(s.NBAR):
                var = pyo.Var(bounds=(0, 1e6), initialize=0.0)
                setattr(self.model, f"DEFICIT_{t}_{b}", var)
                self.DEFICIT[t, b] = var

    def _create_battery_vars(self):
        if not self._battery_list:
            return
        s = self.sistema
        for t in range(self.horizon_time):
            for i, b in enumerate(self._battery_list):
                cap = float(s.BATTERY_CAPACITY[b])
                min_soc = float(s.BATTERY_MIN_SOC[b]) * cap
                p_lim = float(s.BATTERY_POWER_LIMIT[b])
                ch = pyo.Var(bounds=(0, p_lim), initialize=0.0)
                dch = pyo.Var(bounds=(0, p_lim), initialize=0.0)
                soc = pyo.Var(bounds=(min_soc, cap), initialize=min_soc)
                op = pyo.Var(bounds=(-p_lim, p_lim), initialize=0.0)
                setattr(self.model, f"CHARGE_{t}_{b}", ch)
                setattr(self.model, f"DISCHARGE_{t}_{b}", dch)
                setattr(self.model, f"SOC_{t}_{b}", soc)
                setattr(self.model, f"BattOp_{t}_{b}", op)
                self.CHARGE[t, b] = ch
                self.DISCHARGE[t, b] = dch
                self.SOC[t, b] = soc
                self.BatteryOperation[t, b] = op
                # Relação local
                setattr(self.model, f"battery_link_{t}_{b}",
                        pyo.Constraint(expr=op == dch - ch))

    # -------------------- Adição de restrições --------------------
    def _add_all_constraints(self):
        s = self.sistema
        T = self.horizon_time

        # 1. Térmicas
        if s.NGER_CONV > 0:
            ThermalGeneratorConstraints.add_constraints(
                model=self.model, T=T, NGER_CONV=s.NGER_CONV,
                PGER=self.PGER, QGER=self.QGER,
                pgmin_conv=s.PGMIN_CONV, pgmax_conv=s.PGMAX_CONV,
                qgmin_conv=s.QGMIN_CONV if hasattr(s, 'QGMIN_CONV') else None,
                qgmax_conv=s.QGMAX_CONV if hasattr(s, 'QGMAX_CONV') else None,
                pger_inicial_conv=s.PGER_INICIAL_CONV,
                ramp_up_mw=s.RAMP_UP, ramp_down_mw=s.RAMP_DOWN, SB=s.SB
            )

        # 2. Baterias
        if self._battery_list:
            BatteryConstraints.add_constraints(
                model=self.model, sistema=s, T=T,
                battery_list=self._battery_list,
                CHARGE=self.CHARGE, DISCHARGE=self.DISCHARGE,
                SOC=self.SOC, BatteryOperation=self.BatteryOperation,
                soc_inicial_list=self._soc_inicial_list,
                soc_final_list=self._soc_final_list if self._soc_final_list else None,
                daily_reset_to_initial=True
            )

        # 3. Eólicas
        if s.NGER_EOL > 0:
            WindGeneratorConstraints.add_constraints(
                model=self.model, T=T, NGER_EOL=s.NGER_EOL,
                PGWIND=self.PGWIND, CURTAILMENT=self.CURTAILMENT,
                PGWIND_AVAIL=self.PGWIND_AVAIL
            )

        # 4. Rede AC (coordenadas polares)
        wind_gen_to_bar = getattr(s, 'bus_wind', getattr(s, 'BARPG_EOL', [0]*s.NGER_EOL))
        ACElectricConstraints.add_constraints(
            model=self.model, sistema=s, T=T,
            G=self.G, B=self.B,
            V=self.V, ANG=self.ANG,
            PGER=self.PGER, QGER=self.QGER,
            PGWIND=self.PGWIND if s.NGER_EOL > 0 else None,
            conv_gen_to_bar=s.BARPG_CONV,
            wind_gen_to_bar=wind_gen_to_bar,
            PLOAD=self.PLOAD, QLOAD=self.QLOAD,
            DEFICIT=self.DEFICIT,
            CHARGE=self.CHARGE if self._battery_list else None,
            DISCHARGE=self.DISCHARGE if self._battery_list else None,
            battery_list=self._battery_list
        )

    # ----------------------------------------------------------------------
    def build_objective(self, cost_function: Optional[Callable] = None):
        """
        Adiciona a função objetivo ao modelo Pyomo.
        Se cost_function for fornecida, usa-a; caso contrário, constrói a função padrão.
        """
        # Remove qualquer objetivo anterior para evitar conflitos
        for obj in list(self.model.component_objects(pyo.Objective, active=True)):
            self.model.del_component(obj.name)

        if cost_function is not None:
            expr = cost_function(self)
            obj_name = f"_objective_custom_{uuid.uuid4().hex[:8]}"
            self.model.add_component(obj_name, pyo.Objective(expr=expr, sense=pyo.minimize))
            return obj_name

        # Função objetivo padrão
        s = self.sistema
        T = self.horizon_time
        expr = 0.0

        # 1. Custo dos geradores térmicos
        if hasattr(s, 'CPG_CONV'):
            for t in range(T):
                for g in range(s.NGER_CONV):
                    expr += float(s.CPG_CONV[g]) * self.PGER[t, g]

        # 2. Penalidade por corte eólico
        if (hasattr(s, 'CPG_CURTAILMENT') and
            hasattr(self, 'CURTAILMENT') and
            self.CURTAILMENT):
            for t in range(T):
                for w in range(s.NGER_EOL):
                    expr += float(s.CPG_CURTAILMENT[w]) * self.CURTAILMENT[t, w]

        # 3. Custo do déficit
        if hasattr(s, 'CPG_DEFICIT'):
            for t in range(T):
                for b in range(s.NBAR):
                    expr += float(s.CPG_DEFICIT) * self.DEFICIT[t, b]

        # 4. Custo de operação das baterias
        if (hasattr(self, 'CHARGE') and self.CHARGE and
            hasattr(s, 'BATTERY_COST')):
            for t in range(T):
                for b in self._battery_list:
                    if hasattr(s.BATTERY_COST, '__getitem__') and b < len(s.BATTERY_COST):
                        custo = float(s.BATTERY_COST[b])
                    else:
                        custo = float(s.BATTERY_COST)
                    expr += custo * (self.CHARGE[t, b] + self.DISCHARGE[t, b])

        obj_name = f"_objective_{uuid.uuid4().hex[:8]}"
        self.model.add_component(obj_name, pyo.Objective(expr=expr, sense=pyo.minimize))
        return obj_name

    # ----------------------------------------------------------------------
    def solve(self, solver_name='ipopt', tee=True, **kwargs):
        if self.model is None:
            raise RuntimeError("Modelo não construído.")
        opt = pyo.SolverFactory(solver_name)
        results = opt.solve(self.model, tee=tee, **kwargs)
        self._solved = (results.solver.status == pyo.SolverStatus.ok and
                        results.solver.termination_condition == pyo.TerminationCondition.optimal)
        return results

    # ----------------------------------------------------------------------
    def extract_results(self) -> TimeCoupledOPFResult:
        if not self._solved:
            raise RuntimeError("Modelo não resolvido.")
        s = self.sistema
        T = self.horizon_time
        snapshots = []
        dias_nomes = ["domingo", "segunda", "terça", "quarta", "quinta", "sexta", "sábado"]

        for t in range(T):
            dia = t // self.n_horas
            hora = t % self.n_horas
            dia_semana = ((self.dia_inicial + dia) % 7) + 1
            dia_semana_nome = dias_nomes[dia_semana-1]
            try:
                PLOAD_vals = (self.PLOAD[t, :]).tolist()
                PGER_vals = [pyo.value(self.PGER[t, g]) for g in range(s.NGER_CONV)]
                QGER_vals = [pyo.value(self.QGER[t, g]) for g in range(s.NGER_CONV)]

                if s.NGER_EOL > 0:
                    PGWIND_disponivel = (self.PGWIND_AVAIL[t, :]).tolist()
                    PGWIND_vals = [pyo.value(self.PGWIND[t, w]) for w in range(s.NGER_EOL)]
                    CURTAILMENT_vals = [pyo.value(self.CURTAILMENT[t, w]) for w in range(s.NGER_EOL)]
                else:
                    PGWIND_disponivel = PGWIND_vals = CURTAILMENT_vals = []

                DEFICIT_vals = [pyo.value(self.DEFICIT[t, b]) for b in range(s.NBAR)]

                SOC_init = [0.0] * s.NBAR
                SOC_atual = [0.0] * s.NBAR
                BESS_operation = [0.0] * s.NBAR
                if self._battery_list:
                    for i, b in enumerate(self._battery_list):
                        soc_init_val = self._soc_inicial_list[i] if t == 0 else pyo.value(self.SOC[t-1, b])
                        SOC_init[b] = soc_init_val
                        SOC_atual[b] = pyo.value(self.SOC[t, b])
                        BESS_operation[b] = pyo.value(self.DISCHARGE[t, b]) - pyo.value(self.CHARGE[t, b])

                V_vals = [pyo.value(self.V[t, b]) for b in range(s.NBAR)]
                ANG_vals = [pyo.value(self.ANG[t, b]) for b in range(s.NBAR)]

                # Fluxos calculados a partir das tensões (pós‑processamento)
                P_flow = np.zeros(s.NLIN)
                Q_flow = np.zeros(s.NLIN)
                for e in range(s.NLIN):
                    i = s.line_fr[e]
                    j = s.line_to[e]
                    vi, vj = V_vals[i], V_vals[j]
                    th_i, th_j = ANG_vals[i], ANG_vals[j]
                    r, x = s.r_line[e], s.x_line[e]
                    z2 = r*r + x*x
                    g = r / z2
                    b = -x / z2
                    P_flow[e] = vi**2 * g - vi*vj*(g*np.cos(th_i-th_j) + b*np.sin(th_i-th_j))
                    Q_flow[e] = -vi**2 * b - vi*vj*(g*np.sin(th_i-th_j) - b*np.cos(th_i-th_j))

                # Perdas totais por balanço (simplificado)
                total_gen = np.sum(PGER_vals) + (np.sum(PGWIND_vals) if s.NGER_EOL > 0 else 0) + np.sum(BESS_operation) + np.sum(DEFICIT_vals)
                total_load = np.sum(PLOAD_vals)
                perdas_total = total_gen - total_load
                PERDAS_BARRA = [perdas_total / s.NBAR] * s.NBAR

                DEFICIT_pu = [pyo.value(self.DEFICIT[t, b]) for b in range(s.NBAR)]
                custo_def = getattr(s, 'CPG_DEFICIT', 1000.0)
                CUSTO = [d * custo_def for d in DEFICIT_pu]
                CMO = [0.0]

                snapshots.append(TimeCoupledOPFSnapshotResult(
                    dia=dia, dia_semana=dia_semana, hora=hora, sucesso=True,
                    PLOAD=PLOAD_vals,
                    PGER=PGER_vals,
                    PGWIND_disponivel=PGWIND_disponivel,
                    PGWIND=PGWIND_vals,
                    CURTAILMENT=CURTAILMENT_vals,
                    SOC_init=SOC_init,
                    BESS_operation=BESS_operation,
                    SOC_atual=SOC_atual,
                    DEFICIT=DEFICIT_vals,
                    V=V_vals, ANG=ANG_vals,
                    FLUXO_LIN=P_flow.tolist(),
                    CUSTO=CUSTO, CMO=CMO,
                    PERDAS_BARRA=PERDAS_BARRA,
                    dia_semana_nome=dia_semana_nome
                ))
            except Exception as e:
                traceback.print_exc()
                snapshots.append(TimeCoupledOPFSnapshotResult(
                    dia=dia, hora=hora, sucesso=False, mensagem=str(e),
                    dia_semana=dia_semana, dia_semana_nome=dia_semana_nome
                ))

        sucesso_global = all(s.sucesso for s in snapshots)
        return TimeCoupledOPFResult(
            snapshots=snapshots,
            sucesso_global=sucesso_global,
            mensagem_global="OK" if sucesso_global else "Falhas na extração"
        )

    def solve_multiday(self,
                       solver_name='ipopt',
                       fator_carga=None, fator_vento=None,
                       soc_inicial=0.5, soc_final=None,
                       cost_function=None,
                       cen_id=None, tee=True):
        self.build(fator_carga, fator_vento, soc_inicial, soc_final, cost_function)
        results = self.solve(solver_name, tee=tee)
        if self.db_handler and cen_id:
            res = self.extract_results()
            for snap in res.snapshots:
                self.db_handler.save_hourly_result(
                    resultado=snap, sistema=self.sistema,
                    hora=snap.hora, solver_name=solver_name,
                    dia=str(snap.dia+1), cen_id=cen_id
                )
        return results


# =============================================================================
# Exemplo de uso
# =============================================================================
if __name__ == "__main__":
    import sys
    import os
    import numpy as np
    from datetime import datetime
    import secrets

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from UTILS.SystemLoader import SistemaLoader
    from DB.DBhandler_OPF import OPF_DBHandler
    from UTILS.EvaluateFactors import EvaluateFactors

    print("=" * 70)
    print("SIMULAÇÃO ACOPLADA - AC OPF (Pyomo + IPOPT)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Carregar sistema
    # -------------------------------------------------------------------------
    print("\n1. Carregando dados do sistema...")
    json_path = "DATA/input/ieee118_BASE.json"
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
    print(f"   ✓ Baterias: {len(getattr(sistema, 'BARRAS_COM_BATERIA', []))}")

    # -------------------------------------------------------------------------
    # 2. Parâmetros da simulação
    # -------------------------------------------------------------------------
    n_dias = 1
    n_horas = 24
    T = n_dias * n_horas
    print(f"\n2. Simulando {n_dias} dias x {n_horas} horas = {T} períodos.")

    SOC_inicial = 0.5
    SOC_final = 0.5

    # -------------------------------------------------------------------------
    # 3. Configurar banco de dados
    # -------------------------------------------------------------------------
    print("\n3. Configurando banco de dados...")
    db_handler = OPF_DBHandler('DATA/output/resultados_PL_acoplado_AC.db')
    db_handler.create_tables()
    cen_id = datetime.now().strftime('%Y%m%d%H%M%S')
    print(f"   ✓ Cenário ID: {cen_id}")

    # -------------------------------------------------------------------------
    # 4. Criar modelo
    # -------------------------------------------------------------------------
    modelo = TimeCoupledOPFModel(
        sistema=sistema,
        n_horas=n_horas,
        n_dias=n_dias,
        db_handler=db_handler,
        dia_inicial=0
    )

    # -------------------------------------------------------------------------
    # 5. Gerar fatores de carga e vento
    # -------------------------------------------------------------------------
    seed = secrets.randbits(32)
    avaliador = EvaluateFactors(
        sistema=sistema,
        n_dias=n_dias,
        n_horas=n_horas,
        carga_incerteza=0.05,
        vento_variacao=0.9,
        seed=seed
    )
    fatores_carga, fatores_vento = avaliador.gerar_tudo()

    # -------------------------------------------------------------------------
    # 6. Definir função objetivo (padrão ou customizada)
    # -------------------------------------------------------------------------
    def meu_objetivo(model):
        """Exemplo: minimizar apenas curtailment e déficit."""
        expr = 0.0
        for t in range(model.horizon_time):
            for w in range(model.sistema.NGER_EOL):
                expr += 1000 * model.CURTAILMENT[t, w]
            for b in range(model.sistema.NBAR):
                expr += 5000 * model.DEFICIT[t, b]
        return expr

    usar_objetivo_padrao = True
    cost_func = None if usar_objetivo_padrao else meu_objetivo

    # -------------------------------------------------------------------------
    # 7. Resolver modelo (AC não linear com IPOPT)
    # -------------------------------------------------------------------------
    print("\n4. Resolvendo modelo AC integrado com Pyomo + IPOPT...")
    results = modelo.solve_multiday(
        solver_name='ipopt',
        fator_carga=fatores_carga,
        fator_vento=fatores_vento,
        soc_inicial=SOC_inicial,
        soc_final=SOC_final,
        cost_function=cost_func,
        cen_id=cen_id,
        tee=True
    )

    # -------------------------------------------------------------------------
    # 8. Exibir resumo dos resultados
    # -------------------------------------------------------------------------
    resultados = modelo.extract_results()
    print(f"\nSucesso global: {resultados.sucesso_global}")
    print(f"Snapshots extraídos: {len(resultados.snapshots)}")

    t = 0
    if t < len(resultados.snapshots) and resultados.snapshots[t].sucesso:
        snap = resultados.snapshots[t]
        print(f"\n5. Resultados para Hora {snap.hora} (Dia {snap.dia+1}):")
        print(f"   Demanda ativa total:  {sum(snap.PLOAD):.3f} pu")
        print(f"   Geração térmica ativa: {sum(snap.PGER):.3f} pu")
        if sistema.NGER_EOL > 0:
            print(f"   Geração eólica total:   {sum(snap.PGWIND):.3f} pu")
            print(f"   Curtailment total:      {sum(snap.CURTAILMENT):.3f} pu")
        print(f"   Déficit total:          {sum(snap.DEFICIT):.3f} pu")
        if snap.V:
            print(f"   Tensão mínima: {min(snap.V):.3f} pu, máxima: {max(snap.V):.3f} pu")
        if snap.ANG:
            print(f"   Ângulo slack (ref): {snap.ANG[sistema.slack_idx]:.3f} rad")
        if snap.PERDAS_BARRA:
            print(f"   Perdas ativas totais:   {sum(snap.PERDAS_BARRA):.3f} pu")
        if sistema.BARRAS_COM_BATERIA:
            for b in sistema.BARRAS_COM_BATERIA:
                print(f"   Bateria barra {b}:")
                print(f"      operação = {snap.BESS_operation[b]:.3f} pu")
                print(f"      SOC inicial = {snap.SOC_init[b]:.3f} pu")
                print(f"      SOC final   = {snap.SOC_atual[b]:.3f} pu")
        print(f"   CMO (barra slack): {snap.CMO[0]:.2f} $/MWh")

    print("\n" + "=" * 70)
    print("EXECUÇÃO CONCLUÍDA")
    print("=" * 70)