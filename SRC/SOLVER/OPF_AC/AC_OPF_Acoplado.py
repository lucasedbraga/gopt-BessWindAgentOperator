#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo de otimização multi‑período acoplado (AC OPF) com Pyomo + IPOPT.
Todas as grandezas em pu, variáveis indexadas por (t, idx).
Estrutura padronizada conforme o modelo ACOPF_Snapshot.
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
from SOLVER.OPF_AC.RES.BalanceConstraints import AC_BalanceConstraints
from DB.DBmodel_OPF import TimeCoupled_OPF_Result, OPF_SnapshotResult


class ACOPF_TimeCoupled:
    """
    Modelo AC-OPF para múltiplos períodos acoplados usando Pyomo + IPOPT.
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

        # --- Arrays do sistema (padronizados) ---
        self._build_SistemaEletrico_arrays()

        # --- Matriz de admitância ---
        self._monta_matiz_admitancia()

        # --- Dicionários de variáveis (indexados por (t, idx)) ---
        self.var_lists: Dict[str, List] = {}
        self.var_indices: Dict[str, Dict] = {}
        self.PGER_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.QGER_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.PGWIND_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.CURTAILMENT_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.DEFICIT_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.V_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.ANG_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.CHARGE_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.DISCHARGE_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.SOC_dict: Dict[Tuple[int, int], pyo.Var] = {}
        self.BatteryOperation_dict: Dict[Tuple[int, int], pyo.Var] = {}

        # --- Parâmetros (preenchidos em _processa_multiplicadores) ---
        self.PLOAD: Optional[np.ndarray] = None          # (T, n_bus)
        self.QLOAD: Optional[np.ndarray] = None          # (T, n_bus)
        self.PGWIND_AVAIL: Optional[np.ndarray] = None   # (T, n_wind)
        self.soc_inicial_list: List[float] = []
        self.soc_final_list: List[float] = []

    # ----------------------------------------------------------------------
    # 1. Construção dos arrays do sistema (similar ao snapshot)
    # ----------------------------------------------------------------------
    def _build_SistemaEletrico_arrays(self):
        s = self.sistema
        self.NBAR = s.NBAR
        self.NUTE = s.NGER_UTE
        self.NGWD = s.NGER_GWD
        self.NLIN = s.NLIN
        self.NBESS = len(getattr(s, 'BARRAS_COM_BATERIA', []))

        self.thermal_bus = np.array(s.BARPG_CONV, dtype=int)
        self.wind_bus = np.array(getattr(s, 'bus_wind', getattr(s, 'BARPG_EOL', [])), dtype=int)

        self.line_from = np.array(s.line_fr, dtype=int)
        self.line_to = np.array(s.line_to, dtype=int)
        self.line_r = np.array(s.r_line, dtype=float)
        self.line_x = np.array(s.x_line, dtype=float)
        self.line_flow_max = np.array(s.FLIM, dtype=float)

        self.thermal_pmin = np.array(s.PGER_MIN_UTE, dtype=float)
        self.thermal_pmax = np.array(s.PGER_MAX_UTE, dtype=float)

        self.thermal_cost = np.array(getattr(s, 'CUSTO_GER', [50.0]*self.NUTE), dtype=float)

        if hasattr(s, 'QGER_MIN_UTE') and hasattr(s, 'QGER_MAX_UTE'):
            self.thermal_qmin = np.array(s.QGER_MIN_UTE, dtype=float)
            self.thermal_qmax = np.array(s.QGER_MAX_UTE, dtype=float)
        else:
            self.thermal_qmin = 0 * self.thermal_pmax
            self.thermal_qmax = 0 * self.thermal_pmax

        self.battery_buses = np.array(getattr(s, 'BARRAS_COM_BATERIA', []), dtype=int)
        if self.NBESS > 0:
            self.battery_capacity = np.array([s.BATTERY_CAPACITY[b] for b in self.battery_buses], dtype=float)
            self.battery_power_limit = np.array([s.BATTERY_POWER_LIMIT[b] for b in self.battery_buses], dtype=float)
            self.battery_min_soc_frac = np.array([s.BATTERY_MIN_SOC[b] for b in self.battery_buses], dtype=float)
            self.battery_charge_eff = getattr(s, 'BATTERY_CHARGE_EFF', 1.0)
            self.battery_discharge_eff = getattr(s, 'BATTERY_DISCHARGE_EFF', 1.0)
        else:
            self.battery_capacity = np.array([])
            self.battery_power_limit = np.array([])
            self.battery_min_soc_frac = np.array([])

        self.slack_bus = getattr(s, 'slack_idx', 0)
        self.base_Pload = np.array(s.PLOAD, dtype=float)
        self.base_Qload = np.array(s.QLOAD, dtype=float) if hasattr(s, 'QLOAD') else np.zeros(self.NBAR)

    # ----------------------------------------------------------------------
    # 2. Matriz de admitância (mesmo do snapshot)
    # ----------------------------------------------------------------------
    def _monta_matiz_admitancia(self):
        self.G = np.zeros((self.NBAR, self.NBAR))
        self.B = np.zeros((self.NBAR, self.NBAR))

        for LIN in range(self.NLIN):
            i = self.line_from[LIN]
            j = self.line_to[LIN]
            r = self.line_r[LIN]
            x = self.line_x[LIN]

            z2 = r * r + x * x
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

    # ----------------------------------------------------------------------
    # 3. Processamento de fatores (carga, vento, SOC inicial/final)
    # ----------------------------------------------------------------------
    def _processa_multiplicadores(self,
                                  fator_carga: Optional[Union[float, np.ndarray]] = None,
                                  fator_vento: Optional[Union[float, np.ndarray]] = None,
                                  soc_inicial: Union[float, List[float]] = 0.5,
                                  soc_final: Optional[Union[float, List[float]]] = None):
        """
        Processa os fatores de carga, vento e os SOCs iniciais/finais das baterias.
        """
        T = self.horizon_time

        # ----- Carga -----
        if fator_carga is None:
            fc = np.ones((T, self.NBAR))
        else:
            fc = np.asarray(fator_carga)
            if fc.ndim == 3:
                fc = fc.reshape((T, self.NBAR))
            elif fc.ndim == 2:
                if fc.shape[0] == self.n_dias and fc.shape[1] == self.n_horas:
                    fc = np.repeat(fc.reshape((T, 1)), self.NBAR, axis=1)
            elif fc.ndim == 1:
                if fc.size == T:
                    fc = fc[:, np.newaxis] * np.ones((1, self.NBAR))
        self.PLOAD = self.base_Pload[np.newaxis, :] * fc
        self.QLOAD = self.base_Qload[np.newaxis, :] * fc

        # ----- Vento -----
        if self.NGWD == 0:
            self.PGWIND_AVAIL = np.zeros((T, 0))
        else:
            if fator_vento is None:
                fv = np.ones((T, self.NGWD))
            else:
                fv = np.asarray(fator_vento)
                if fv.ndim == 3:
                    fv = fv.reshape((T, self.NGWD))
                elif fv.ndim == 2:
                    if fv.shape[0] == self.n_dias and fv.shape[1] == self.n_horas:
                        fv = np.repeat(fv.reshape((T, 1)), self.NGWD, axis=1)
                elif fv.ndim == 1:
                    if fv.size == T:
                        fv = fv[:, np.newaxis] * np.ones((1, self.NGWD))
            # Supondo que o sistema tenha PGWD_MAX_ORIGINAL ou PGWIND_disponivel
            if hasattr(self.sistema, 'PGWD_MAX_ORIGINAL'):
                base_wind = np.array(self.sistema.PGWD_MAX_ORIGINAL)
            else:
                base_wind = np.array(getattr(self.sistema, 'PGWIND_disponivel', [1.0] * self.NGWD))
            self.PGWIND_AVAIL = base_wind[np.newaxis, :] * fv

        # ----- SOC inicial e final -----
        self.soc_inicial_list = []
        self.soc_final_list = []
        if self.NBESS > 0:
            if isinstance(soc_inicial, (float, int)):
                soc_ini_frac = [soc_inicial] * self.NBESS
            else:
                soc_ini_frac = list(soc_inicial)
            self.soc_inicial_list = [
                soc_ini_frac[i] * self.battery_capacity[i] for i in range(self.NBESS)
            ]

            if soc_final is not None:
                if isinstance(soc_final, (float, int)):
                    soc_fin_frac = [soc_final] * self.NBESS
                else:
                    soc_fin_frac = list(soc_final)
                self.soc_final_list = [
                    soc_fin_frac[i] * self.battery_capacity[i] for i in range(self.NBESS)
                ]
            else:
                self.soc_final_list = []

    # ----------------------------------------------------------------------
    # 4. Criação das variáveis (padrão snapshot)
    # ----------------------------------------------------------------------
    def _add_VARS(self):
        T = self.horizon_time

        # --- Tensão ---
        self.var_lists['v_pu'] = []
        for t in range(T):
            for b in range(self.NBAR):
                var = pyo.Var(bounds=(0.95, 1.05), initialize=1.0)
                setattr(self.model, f"V_pu_T{t}_BAR{b+1}", var)
                self.var_lists['v_pu'].append(var)
                self.V_dict[(t, b)] = var
        self.var_indices['v_pu'] = {(t, b): v for (t, b), v in self.V_dict.items()}

        # --- Ângulo ---
        self.var_lists['ang_pu'] = []
        for t in range(T):
            for b in range(self.NBAR):
                var = pyo.Var(bounds=(-np.pi, np.pi), initialize=0.0)
                setattr(self.model, f"ANG_pu_T{t}_BAR{b+1}", var)
                self.var_lists['ang_pu'].append(var)
                self.ANG_dict[(t, b)] = var
            # Fixa ângulo da barra slack
            setattr(self.model, f"fix_slack_angle_T{t}",
                    pyo.Constraint(expr=self.ANG_dict[(t, self.slack_bus)] == 0.0))
        self.var_indices['ang_pu'] = {(t, b): v for (t, b), v in self.ANG_dict.items()}

        # --- Geração térmica ativa ---
        self.var_lists['PGER_UTE'] = []
        for t in range(T):
            for g in range(self.NUTE):
                var = pyo.Var(bounds=(self.thermal_pmin[g], self.thermal_pmax[g]), initialize=0.0)
                setattr(self.model, f"PGER_UTE_T{t}_{g+1}", var)
                self.var_lists['PGER_UTE'].append(var)
                self.PGER_dict[(t, g)] = var
        self.var_indices['PGER_UTE'] = {(t, g): v for (t, g), v in self.PGER_dict.items()}

        # --- Geração térmica reativa ---
        self.var_lists['QGER_UTE'] = []
        for t in range(T):
            for g in range(self.NUTE):
                var = pyo.Var(bounds=(self.thermal_qmin[g], self.thermal_qmax[g]), initialize=0.0)
                setattr(self.model, f"QGER_UTE_T{t}_{g+1}", var)
                self.var_lists['QGER_UTE'].append(var)
                self.QGER_dict[(t, g)] = var
        self.var_indices['QGER_UTE'] = {(t, g): v for (t, g), v in self.QGER_dict.items()}

        # --- Déficit ---
        self.var_lists['deficit'] = []
        for t in range(T):
            for b in range(self.NBAR):
                var = pyo.Var(bounds=(0, 1e6), initialize=0.0)
                setattr(self.model, f"DEFICT_T{t}_BAR{b+1}", var)
                self.var_lists['deficit'].append(var)
                self.DEFICIT_dict[(t, b)] = var
        self.var_indices['deficit'] = {(t, b): v for (t, b), v in self.DEFICIT_dict.items()}

        # --- Eólicas  ---
        if self.NGWD > 0:
            self.var_lists['p_wind'] = []
            self.var_lists['curtailment'] = []
            for t in range(T):
                for w in range(self.NGWD):
                    avail = self.PGWIND_AVAIL[t, w]
                    p_var = pyo.Var(bounds=(0, avail), initialize=0.0)
                    c_var = pyo.Var(bounds=(0, avail), initialize=0.0)
                    setattr(self.model, f"PGWD_T{t}_{w}", p_var)
                    setattr(self.model, f"CURTAILMENT_T{t}_{w}", c_var)
                    self.var_lists['p_wind'].append(p_var)
                    self.var_lists['curtailment'].append(c_var)
                    self.PGWIND_dict[(t, w)] = p_var
                    self.CURTAILMENT_dict[(t, w)] = c_var
            self.var_indices['p_wind'] = {(t, w): v for (t, w), v in self.PGWIND_dict.items()}
            self.var_indices['curtailment'] = {(t, w): v for (t, w), v in self.CURTAILMENT_dict.items()}

        # --- Baterias  ---
        if self.NBESS > 0:
            self.var_lists['charge'] = []
            self.var_lists['discharge'] = []
            self.var_lists['soc'] = []
            self.var_lists['battery_op'] = []
            for t in range(T):
                for i, bus in enumerate(self.battery_buses):
                    power_limit = self.battery_power_limit[i]
                    cap = self.battery_capacity[i]
                    min_soc = self.battery_min_soc_frac[i] * cap
                    ch = pyo.Var(bounds=(0, power_limit), initialize=0.0)
                    dch = pyo.Var(bounds=(0, power_limit), initialize=0.0)
                    soc = pyo.Var(bounds=(min_soc, cap), initialize=min_soc)
                    op = pyo.Var(bounds=(-power_limit, power_limit), initialize=0.0)
                    setattr(self.model, f"charge_T{t}_bus{bus+1}", ch)
                    setattr(self.model, f"discharge_T{t}_bus{bus+1}", dch)
                    setattr(self.model, f"soc_T{t}_bus{bus+1}", soc)
                    setattr(self.model, f"battery_op_T{t}_bus{bus+1}", op)
                    self.var_lists['charge'].append(ch)
                    self.var_lists['discharge'].append(dch)
                    self.var_lists['soc'].append(soc)
                    self.var_lists['battery_op'].append(op)
                    self.CHARGE_dict[(t, bus)] = ch
                    self.DISCHARGE_dict[(t, bus)] = dch
                    self.SOC_dict[(t, bus)] = soc
                    self.BatteryOperation_dict[(t, bus)] = op
                    setattr(self.model, f"battery_link_T{t}_bus{bus}",
                            pyo.Constraint(expr=op == dch - ch))
            self.var_indices['charge'] = {(t, b): v for (t, b), v in self.CHARGE_dict.items()}
            self.var_indices['discharge'] = {(t, b): v for (t, b), v in self.DISCHARGE_dict.items()}
            self.var_indices['soc'] = {(t, b): v for (t, b), v in self.SOC_dict.items()}
            self.var_indices['battery_op'] = {(t, b): v for (t, b), v in self.BatteryOperation_dict.items()}

    # ----------------------------------------------------------------------
    # 5. Adição das restrições (usa as mesmas classes externas do snapshot)
    # ----------------------------------------------------------------------
    def _add_CONS(self):
        T = self.horizon_time

        # 1. Geradores térmicos
        if self.NUTE > 0:
            ThermalGeneratorConstraints.add_constraints(
                model=self.model,
                T=T,
                NGER_UTE=self.NUTE,
                PGER=self.PGER_dict,
                QGER=self.QGER_dict,
                PGER_MIN_UTE=self.thermal_pmin,
                PGER_MAX_UTE=self.thermal_pmax,
                qgmin_conv=self.thermal_qmin,
                qgmax_conv=self.thermal_qmax,
                PGER_inicial_UTE=self.sistema.PGER_inicial_UTE,
                ramp_up_mw=self.sistema.RAMP_UP,
                ramp_down_mw=self.sistema.RAMP_DOWN,
                SB=self.sistema.SB
            )

        # 2. Eólicas
        if self.NGWD > 0:
            WindGeneratorConstraints.add_constraints(
                model=self.model,
                T=T,
                NGER_GWD=self.NGWD,
                PGWIND=self.PGWIND_dict,
                CURTAILMENT=self.CURTAILMENT_dict,
                PGWIND_AVAIL=self.PGWIND_AVAIL
            )

        # 3. Baterias
        if self.NBESS > 0:
            BatteryConstraints.add_constraints(
                model=self.model,
                sistema=self.sistema,
                T=T,
                battery_list=self.battery_buses.tolist(),
                CHARGE=self.CHARGE_dict,
                DISCHARGE=self.DISCHARGE_dict,
                SOC=self.SOC_dict,
                BatteryOperation=self.BatteryOperation_dict,
                soc_inicial_list=self.soc_inicial_list,
                soc_final_list=self.soc_final_list if self.soc_final_list else None,
                daily_reset_to_initial=True   # ou False, conforme desejo
            )

        # 4. Balanço de potência AC (coordenadas polares)
        wind_gen_to_bar = self.wind_bus.tolist() if self.NGWD > 0 else None
        battery_list = self.battery_buses.tolist() if self.NBESS > 0 else None

        AC_BalanceConstraints.add_constraints(
            model=self.model,
            sistema=self.sistema,
            HORA=T,  # aqui representa o número de períodos
            G=self.G,
            B=self.B,
            V=self.V_dict,
            ANG=self.ANG_dict,
            PGER=self.PGER_dict,
            QGER=self.QGER_dict,
            PLOAD=self.PLOAD,
            QLOAD=self.QLOAD,
            DEFICIT=self.DEFICIT_dict,
            PGWIND=self.PGWIND_dict if self.NGWD > 0 else None,
            BESS_SOC_op=self.BatteryOperation_dict if self.NBESS > 0 else None,
            conv_gen_to_bar=self.thermal_bus.tolist(),
            wind_gen_to_bar=wind_gen_to_bar,
            battery_list=battery_list
        )

    # ----------------------------------------------------------------------
    # 6. Função objetivo (padrão: minimizar custos + penalidades)
    # ----------------------------------------------------------------------
    def _add_FOB(self):
        """
        Adiciona a função objetivo ao modelo.
        Se  for fornecida, usa-a; caso contrário, usa a função padrão.
        """       

        s = self.sistema
        T = self.horizon_time
        expr = 0.0

        # Custo dos geradores térmicos
        if hasattr(s, 'custo_GER'):
            for t in range(T):
                for g in range(self.NUTE):
                    expr += float(s.custo_GER[g]) * self.PGER_dict[(t, g)]

        # Penalidade por corte eólico
        if hasattr(s, 'custo_CURTAILMENT') and self.NGWD > 0:
            for t in range(T):
                for w in range(self.NGWD):
                    expr += float(s.custo_CURTAILMENT[w]) * self.CURTAILMENT_dict[(t, w)]

        # Penalidade por déficit
        if hasattr(s, 'custo_DEFICIT'):
            for t in range(T):
                for b in range(self.NBAR):
                    expr += float(s.custo_DEFICIT) * self.DEFICIT_dict[(t, b)]

        # Custo de operação das baterias (carga e descarga)
        if hasattr(s, 'BATTERY_COST_CHARGE') and self.NBESS > 0:
            for t in range(T):
                for i, b in enumerate(self.battery_buses):
                    # Obtém o custo de carga
                    custo_charge = s.BATTERY_COST_CHARGE
                    # Se for um array, pega o valor para a bateria i (ou b)
                    if hasattr(custo_charge, '__getitem__') and len(custo_charge) > 1:
                        custo_charge = float(custo_charge[i])
                    else:
                        custo_charge = float(custo_charge)
                    expr += custo_charge * self.CHARGE_dict[(t, b)]

        if hasattr(s, 'BATTERY_COST_DISCHARGE') and self.NBESS > 0:
            for t in range(T):
                for i, b in enumerate(self.battery_buses):
                    custo_discharge = s.BATTERY_COST_DISCHARGE
                    if hasattr(custo_discharge, '__getitem__') and len(custo_discharge) > 1:
                        custo_discharge = float(custo_discharge[i])
                    else:
                        custo_discharge = float(custo_discharge)
                    expr += custo_discharge * self.DISCHARGE_dict[(t, b)]

        self.model.FOB = pyo.Objective(expr=expr, sense=pyo.minimize)

    # ----------------------------------------------------------------------
    # 7. Construção completa do modelo
    # ----------------------------------------------------------------------
    def _build_AC_OPF_TIME(self,
                           fator_carga: Optional[Union[float, np.ndarray]] = None,
                           fator_vento: Optional[Union[float, np.ndarray]] = None,
                           soc_inicial: Union[float, List[float]] = 0.5,
                           soc_final: Optional[Union[float, List[float]]] = None) -> None:
        """
        Monta o modelo Pyomo com todos os componentes.
        """
        self.model = pyo.ConcreteModel(name="ACOPF_TimeCoupled")

        # Processa parâmetros
        self._processa_multiplicadores(fator_carga, fator_vento, soc_inicial, soc_final)

        # Cria variáveis
        self._add_VARS()

        # Adiciona restrições
        self._add_CONS()

        # Adiciona função objetivo
        self._add_FOB()

        self._solved = False

    def _build_Cenario(self,
                       fator_carga=None,
                       fator_vento=None,
                       soc_inicial=0.5,
                       soc_final=None):
        """
        Wrapper para _build_AC_OPF_TIME (mesmo nome do snapshot).
        """
        self._build_AC_OPF_TIME(fator_carga, fator_vento, soc_inicial, soc_final)

    # ----------------------------------------------------------------------
    # 8. Solução
    # ----------------------------------------------------------------------
    def solve(self, solver_name='ipopt', tee=True, **kwargs):
        if self.model is None:
            raise RuntimeError("Modelo não construído.")
        opt = pyo.SolverFactory(solver_name)
        try:
            opt.set_executable('/home/lucasedbraga/anaconda3/envs/otm_venv/bin/ipopt')
        except:
            pass
        results = opt.solve(self.model, tee=tee, **kwargs)
        self._solved = (results.solver.status == pyo.SolverStatus.ok and
                        results.solver.termination_condition == pyo.TerminationCondition.optimal)
        return results

    def solve_timecoupled(self,
                          solver_name='ipopt',
                          fator_carga=None,
                          fator_vento=None,
                          soc_inicial=0.5,
                          soc_final=None,
                          cen_id=None,
                          tee=True):
        """
        Método principal: constrói, resolve, extrai e salva (se db_handler).
        """
        self._build_Cenario(fator_carga, fator_vento, soc_inicial, soc_final)

        # Opcional: imprimir modelo
        # self.model.pprint()

        results = self.solve(solver_name, tee=tee)

        if self.db_handler is not None and cen_id is not None:
            resultado_global = self.extract_results()
            for snap in resultado_global.snapshots:
                self.db_handler.save_hourly_result(
                    resultado=snap,
                    sistema=self.sistema,
                    hora=snap.hora,
                    solver_name=solver_name,
                    dia=str(snap.dia + 1),
                    cen_id=cen_id
                )

        return results

    # ----------------------------------------------------------------------
    # 9. Extração de resultados 
    # ----------------------------------------------------------------------
    def extract_results(self) -> TimeCoupled_OPF_Result:
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
            dia_semana_nome = dias_nomes[dia_semana - 1]

            try:
                PLOAD_vals = self.PLOAD[t, :].tolist()
                QLOAD_vals = self.QLOAD[t, :].tolist() if self.QLOAD is not None else []
                PGER_vals = [pyo.value(self.PGER_dict[(t, g)]) for g in range(self.NUTE)]
                QGER_vals = [pyo.value(self.QGER_dict[(t, g)]) for g in range(self.NUTE)]

                if self.NGWD > 0:
                    PGWIND_disponivel = self.PGWIND_AVAIL[t, :].tolist()
                    PGWIND_vals = [pyo.value(self.PGWIND_dict[(t, w)]) for w in range(self.NGWD)]
                    CURTAILMENT_vals = [pyo.value(self.CURTAILMENT_dict[(t, w)]) for w in range(self.NGWD)]
                else:
                    PGWIND_disponivel = PGWIND_vals = CURTAILMENT_vals = []

                DEFICIT_vals = [pyo.value(self.DEFICIT_dict[(t, b)]) for b in range(self.NBAR)]

                SOC_init = [0.0] * self.NBAR
                SOC_atual = [0.0] * self.NBAR
                BESS_operation = [0.0] * self.NBAR
                if self.NBESS > 0:
                    for i, bus in enumerate(self.battery_buses):
                        if t == 0:
                            soc_init_val = self.soc_inicial_list[i]
                        else:
                            soc_init_val = pyo.value(self.SOC_dict[(t - 1, bus)])
                        SOC_init[bus] = soc_init_val
                        SOC_atual[bus] = pyo.value(self.SOC_dict[(t, bus)])
                        charge = pyo.value(self.CHARGE_dict[(t, bus)])
                        discharge = pyo.value(self.DISCHARGE_dict[(t, bus)])
                        BESS_operation[bus] = discharge - charge

                V_vals = [pyo.value(self.V_dict[(t, b)]) for b in range(self.NBAR)]
                ANG_vals = [pyo.value(self.ANG_dict[(t, b)]) for b in range(self.NBAR)]

                # Fluxos de potência (pós‑processamento)
                P_flow = np.zeros(self.NLIN)
                Q_flow = np.zeros(self.NLIN)
                for e in range(self.NLIN):
                    i = self.line_from[e]
                    j = self.line_to[e]
                    vi, vj = V_vals[i], V_vals[j]
                    th_i, th_j = ANG_vals[i], ANG_vals[j]
                    r, x = self.line_r[e], self.line_x[e]
                    z2 = r * r + x * x
                    if z2 == 0:
                        continue
                    g = r / z2
                    b = -x / z2
                    P_flow[e] = vi ** 2 * g - vi * vj * (g * np.cos(th_i - th_j) + b * np.sin(th_i - th_j))
                    Q_flow[e] = -vi ** 2 * b - vi * vj * (g * np.sin(th_i - th_j) - b * np.cos(th_i - th_j))

                # Perdas totais
                total_gen = (np.sum(PGER_vals) + np.sum(PGWIND_vals) +
                             np.sum(BESS_operation) + np.sum(DEFICIT_vals))
                total_load = np.sum(PLOAD_vals)
                PERDAS_TOTAIS = total_gen - total_load
                PERDAS_BARRA = [PERDAS_TOTAIS / self.NBAR] * self.NBAR

                custo_deficit_pu = getattr(s, 'Custo_DEFICT', 1000.0)
                CUSTO = [d * custo_deficit_pu for d in DEFICIT_vals]
                CMO = [0.0]

                snapshots.append(OPF_SnapshotResult(
                    dia=dia,
                    dia_semana=dia_semana,
                    hora=hora,
                    sucesso=True,
                    PLOAD=PLOAD_vals,
                    QLOAD=QLOAD_vals,
                    PGER=PGER_vals,
                    QGER=QGER_vals,
                    PGWIND_disponivel=PGWIND_disponivel,
                    PGWIND=PGWIND_vals,
                    CURTAILMENT=CURTAILMENT_vals,
                    SOC_init=SOC_init,
                    BESS_operation=BESS_operation,
                    SOC_atual=SOC_atual,
                    DEFICIT=DEFICIT_vals,
                    V=V_vals,
                    ANG=ANG_vals,
                    FLUXO_LIN=P_flow.tolist(),
                    REATIVO_LIN=Q_flow.tolist(),
                    CUSTO=CUSTO,
                    CMO=CMO,
                    PERDAS_TOTAIS=PERDAS_TOTAIS,
                    dia_semana_nome=dia_semana_nome
                ))
            except Exception as e:
                traceback.print_exc()
                snapshots.append(OPF_SnapshotResult(
                    dia=dia,
                    hora=hora,
                    sucesso=False,
                    mensagem=str(e),
                    dia_semana=dia_semana,
                    dia_semana_nome=dia_semana_nome
                ))

        sucesso_global = all(s.sucesso for s in snapshots)
        return TimeCoupled_OPF_Result(
            snapshots=snapshots,
            sucesso_global=sucesso_global,
            mensagem_global="OK" if sucesso_global else "Falhas na extração"
        )


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
    print("SIMULAÇÃO ACOPLADA - AC OPF (Pyomo + IPOPT) - PADRÃO SNAPSHOT")
    print("=" * 70)

    # 1. Carregar sistema
    print("\n1. Carregando dados do sistema...")
    json_path = "DATA/input/ieee118_BESS.json"
    if not os.path.exists(json_path):
        print(f"ERRO: Arquivo não encontrado: {json_path}")
        sys.exit(1)

    sistema = SistemaLoader(json_path)
    print(f"   ✓ Sistema carregado: {json_path}")
    print(f"   ✓ Potência base: {sistema.SB:.1f} MVA")
    print(f"   ✓ Barras: {sistema.NBAR}")
    print(f"   ✓ Linhas: {sistema.NLIN}")
    print(f"   ✓ Geradores convencionais: {sistema.NGER_UTE}")
    print(f"   ✓ Geradores eólicos: {sistema.NGER_GWD}")
    print(f"   ✓ Baterias: {len(getattr(sistema, 'BARRAS_COM_BATERIA', []))}")

    # 2. Parâmetros
    n_dias = 1
    n_horas = 24
    T = n_dias * n_horas
    print(f"\n2. Simulando {n_dias} dias x {n_horas} horas = {T} períodos.")
    SOC_inicial = 0.5
    SOC_final = 0.5

    # 3. Banco de dados
    print("\n3. Configurando banco de dados...")
    db_handler = OPF_DBHandler('DATA/output/resultados_PL_acoplado_AC.db')
    db_handler.create_tables()
    cen_id = datetime.now().strftime('%Y%m%d%H%M%S')
    print(f"   ✓ Cenário ID: {cen_id}")

    # 4. Criar modelo (agora com a nova classe padronizada)
    modelo = ACOPF_TimeCoupled(
        sistema=sistema,
        n_horas=n_horas,
        n_dias=n_dias,
        db_handler=db_handler,
        dia_inicial=0
    )

    # 5. Gerar fatores
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

    # 6. (Opcional) função objetivo customizada
    def meu_objetivo(model):
        expr = 0.0
        for t in range(model.horizon_time):
            for w in range(model.NGWD):
                expr += 1000 * model.CURTAILMENT_dict[(t, w)]
            for b in range(model.NBAR):
                expr += 5000 * model.DEFICIT_dict[(t, b)]
        return expr

    usar_custom = False
    cost_func = meu_objetivo if usar_custom else None

    # 7. Resolver
    print("\n4. Resolvendo modelo AC integrado com Pyomo + IPOPT...")
    results = modelo.solve_timecoupled(
        solver_name='ipopt',
        fator_carga=fatores_carga,
        fator_vento=fatores_vento,
        soc_inicial=SOC_inicial,
        soc_final=SOC_final,
        cen_id=cen_id,
        tee=True
    )

    # 8. Resultados
    resultados = modelo.extract_results()
    print(f"\nSucesso global: {resultados.sucesso_global}")
    print(f"Snapshots extraídos: {len(resultados.snapshots)}")

    for t in range(T):
        snap = resultados.snapshots[t]
        print(f"\n5. Resultados para Hora {snap.hora} (Dia {snap.dia+1}):")
        print(f"   Demanda Ativa :  {sum(snap.PLOAD):.3f} pu")
        print(f"   Geração Ativa (UTE): {sum(snap.PGER):.3f} pu")
        print(f"   Demanda Reativa :  {sum(snap.QLOAD):.3f} pu")
        print(f"   Geração Reativa (UTE): {sum(snap.QGER):.3f} pu")
        if sistema.NGER_GWD > 0:
            print(f"   Geração WIND total:   {sum(snap.PGWIND):.3f} pu")
            print(f"   WIND Curtailment total:      {sum(snap.CURTAILMENT):.3f} pu")
        print(f"   Déficit total:          {sum(snap.DEFICIT):.3f} pu")
        print(f"   Perdas ativas totais:   {snap.PERDAS_TOTAIS:.3f} pu")
        print(f"   Tensão mínima: {min(snap.V):.3f} pu, máxima: {max(snap.V):.3f} pu")
        print(f"   Ângulo slack (ref): {snap.ANG[sistema.slack_idx]:.3f} rad")
        if sistema.BARRAS_COM_BATERIA:
            for b in sistema.BARRAS_COM_BATERIA:
                print(f"   Bateria barra {b}:")
                print(f"      operação = {snap.BESS_operation[b]:.3f} pu")
                print(f"      SOC inicial = {snap.SOC_init[b]:.3f} pu")
                print(f"      SOC final   = {snap.SOC_atual[b]:.3f} pu")
        #print(f"   CMO (barra slack): {snap.CMO[0]:.2f} $/MWh")


    print("\n" + "=" * 70)
    print("EXECUÇÃO CONCLUÍDA")
    print("=" * 70)