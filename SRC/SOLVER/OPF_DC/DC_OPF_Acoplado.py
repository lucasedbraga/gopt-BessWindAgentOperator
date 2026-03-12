#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo de otimização multi‑período acoplado (DC OPF) com suporte a baterias e perdas iterativas.
Versão refatorada com todas as grandezas em pu, seguindo o padrão do snapshot.
Utiliza as classes de restrição externas (BatteryConstraints, ThermalGeneratorConstraints,
WindGeneratorConstraints, ElectricConstraints).
"""

import os
import sys
import numpy as np
import pyoptinterface as poi
from pyoptinterface import highs
from typing import List, Union, Optional, Tuple, Dict, Callable
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SOLVER.OPF_DC.RES.BatteryConstraints import BatteryConstraints
from SOLVER.OPF_DC.RES.ThermalGeneratorConstraints import ThermalGeneratorConstraints
from SOLVER.OPF_DC.RES.WindGeneratorConstraints import WindGeneratorConstraints
from SOLVER.OPF_DC.RES.EletricConstraints import ElectricConstraints
from DB.DBmodel_OPF import TimeCoupledOPFResult, TimeCoupledOPFSnapshotResult


class TimeCoupledOPFModel:
    """
    Modelo de otimização multi‑período com acoplamento temporal (PyOptInterface).
    Todas as grandezas de potência e energia são tratadas em pu (por unidade).
    As variáveis são indexadas por (t, idx) e as restrições são delegadas a classes externas.
    """

    def __init__(self,
                 sistema,
                 n_horas: int = 24,
                 n_dias: int = 1,
                 db_handler=None,
                 considerar_perdas: bool = True,
                 dia_inicial: int = 0):
        self.sistema = sistema
        self.n_horas = n_horas
        self.n_dias = n_dias
        self.horizon_time = n_horas * n_dias
        self.db_handler = db_handler
        self.considerar_perdas = considerar_perdas
        self.dia_inicial = dia_inicial

        # Modelo PyOptInterface
        self.model = None
        self._solved = False

        # Dicionários para variáveis (chave (t, idx))
        self.PGER: Dict[Tuple[int, int], poi.Variable] = {}
        self.PGWIND: Dict[Tuple[int, int], poi.Variable] = {}
        self.CURTAILMENT: Dict[Tuple[int, int], poi.Variable] = {}
        self.DEFICIT: Dict[Tuple[int, int], poi.Variable] = {}
        self.V: Dict[Tuple[int, int], poi.Variable] = {}
        self.ANG: Dict[Tuple[int, int], poi.Variable] = {}
        self.FLUXO_LIN: Dict[Tuple[int, int], poi.Variable] = {}
        self.CHARGE: Dict[Tuple[int, int], poi.Variable] = {}
        self.DISCHARGE: Dict[Tuple[int, int], poi.Variable] = {}
        self.SOC: Dict[Tuple[int, int], poi.Variable] = {}
        self.BatteryOperation: Dict[Tuple[int, int], poi.Variable] = {}

        # Parâmetros (arrays em pu)
        self.PLOAD: Optional[np.ndarray] = None          # (T, n_bus)
        self.PGWIND_AVAIL: Optional[np.ndarray] = None   # (T, n_wind)

        # Lista de restrições de balanço (para iterações de perdas)
        self.balance_constraints: List[Tuple[int, int, poi.Constraint]] = []

        # Listas de SOC inicial/final (pu)
        self._battery_list: List[int] = []
        self._battery_index: Dict[int, int] = {}
        self._soc_inicial_list: List[float] = []
        self._soc_final_list: List[float] = []

        # Perdas (atualizadas iterativamente) em pu
        self._perdas_calculadas: Optional[np.ndarray] = None

    # ----------------------------------------------------------------------
    # Construção do modelo
    # ----------------------------------------------------------------------
    def build(self,
              fator_carga: Optional[np.ndarray] = None,
              fator_vento: Optional[np.ndarray] = None,
              soc_inicial: Union[float, List[float]] = 0.5,
              soc_final: Optional[Union[float, List[float]]] = None) -> None:
        """
        Constrói o modelo com variáveis temporais.

        Parâmetros:
        -----------
        fator_carga : np.ndarray, opcional
            Fatores de carga, shape (n_dias, n_horas, n_bus) ou (n_dias, n_horas).
        fator_vento : np.ndarray, opcional
            Fatores de vento, shape (n_dias, n_horas, n_wind) ou (n_dias, n_horas).
        soc_inicial : float ou list
            SOC inicial (fração da capacidade) para cada bateria.
        soc_final : float ou list, opcional
            SOC final desejado (fração da capacidade) para cada bateria.
        """
        s = self.sistema
        T = self.horizon_time

        # ------------------------------------------------------------------
        # Processar fatores de carga e vento -> arrays 2D (T, n_bus) e (T, n_wind) em pu
        # ------------------------------------------------------------------
        self._process_fatores(fator_carga, fator_vento)

        # ------------------------------------------------------------------
        # Processar SOC inicial/final (converter fração -> pu)
        # ------------------------------------------------------------------
        self._process_soc(soc_inicial, soc_final)

        # ------------------------------------------------------------------
        # Inicializar modelo
        # ------------------------------------------------------------------
        self.model = highs.Model()

        # ------------------------------------------------------------------
        # Criar variáveis (todas com limites em pu)
        # ------------------------------------------------------------------
        self._create_thermal_vars()
        self._create_wind_vars()
        self._create_deficit_vars()
        self._create_voltage_vars()
        self._create_angle_vars()
        self._create_flow_vars()
        self._create_battery_vars()


        # ------------------------------------------------------------------
        # Adicionar restrições usando as classes externas
        # ------------------------------------------------------------------
        self._add_all_constraints()

        self._solved = False

    def _process_fatores(self, fator_carga, fator_vento):
        """Converte fatores em arrays 2D (T, n_bus) e (T, n_wind) em pu."""
        s = self.sistema
        T = self.horizon_time

        # -------------------- Carga --------------------
        if fator_carga is None:
            fc = np.ones((T, s.NBAR))
        else:
            fc = np.asarray(fator_carga)
            # Redimensionar para (T, n_bus)
            if fc.ndim == 3:  # (n_dias, n_horas, n_bus)
                fc = fc.reshape((T, s.NBAR))
            elif fc.ndim == 2:
                if fc.shape[0] == self.n_dias and fc.shape[1] == self.n_horas:
                    # Expandir para todas as barras
                    fc = np.repeat(fc.reshape((T, 1)), s.NBAR, axis=1)
                elif fc.shape[1] == s.NBAR:
                    # já está (T, n_bus)
                    pass
                else:
                    raise ValueError("fator_carga com shape incompatível")
            elif fc.ndim == 1:
                if fc.size == T:
                    fc = fc[:, np.newaxis] * np.ones((1, s.NBAR))
                else:
                    raise ValueError("fator_carga com shape incompatível")
        # PLOAD = base_load (pu) * fator_carga
        self.PLOAD = s.PLOAD[np.newaxis, :] * fc  # (T, n_bus)

        # -------------------- Vento --------------------
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
                    elif fv.shape[1] == s.NGER_EOL:
                        pass
                    else:
                        raise ValueError("fator_vento com shape incompatível")
                elif fv.ndim == 1:
                    if fv.size == T:
                        fv = fv[:, np.newaxis] * np.ones((1, s.NGER_EOL))
                    else:
                        raise ValueError("fator_vento com shape incompatível")
            # PGWIND_AVAIL = capacidade instalada (pu) * fator
            self.PGWIND_AVAIL = s.PGMAX_EOL_ORIGINAL[np.newaxis, :] * fv  # (T, n_wind)

    def _process_soc(self, soc_inicial, soc_final):
        """Converte SOC inicial/final (frações) para listas em pu."""
        s = self.sistema
        self._battery_list = list(s.BARRAS_COM_BATERIA)
        self._battery_index = {b: i for i, b in enumerate(self._battery_list)}
        nb = len(self._battery_list)
        if nb == 0:
            return

        # SOC inicial
        if isinstance(soc_inicial, (float, int)):
            soc_ini_frac = [soc_inicial] * nb
        else:
            soc_ini_frac = list(soc_inicial)
            if len(soc_ini_frac) != nb:
                raise ValueError(f"soc_inicial deve ter {nb} elementos")
        self._soc_inicial_list = [
            soc_ini_frac[i] * s.BATTERY_CAPACITY[b] for i, b in enumerate(self._battery_list)
        ]

        # SOC final (opcional)
        if soc_final is not None:
            if isinstance(soc_final, (float, int)):
                soc_fin_frac = [soc_final] * nb
            else:
                soc_fin_frac = list(soc_final)
                if len(soc_fin_frac) != nb:
                    raise ValueError(f"soc_final deve ter {nb} elementos")
            self._soc_final_list = [
                soc_fin_frac[i] * s.BATTERY_CAPACITY[b] for i, b in enumerate(self._battery_list)
            ]
        else:
            self._soc_final_list = []

    # -------------------- Criação de variáveis (bounds em pu) --------------------
    def _create_thermal_vars(self):
        s = self.sistema
        for t in range(self.horizon_time):
            for g in range(s.NGER_CONV):
                self.PGER[t, g] = self.model.add_variable(
                    lb=s.PGMIN_CONV[g],          # pu
                    ub=s.PGMAX_CONV[g],          # pu
                    name=f"PGER_{t}_{g}"
                )

    def _create_wind_vars(self):
        if self.sistema.NGER_EOL == 0:
            return
        s = self.sistema
        for t in range(self.horizon_time):
            for w in range(s.NGER_EOL):
                avail = self.PGWIND_AVAIL[t, w]
                self.PGWIND[t, w] = self.model.add_variable(
                    lb=0,
                    ub=avail,
                    name=f"PGWIND_{t}_{w}"
                )
                self.CURTAILMENT[t, w] = self.model.add_variable(
                    lb=0,
                    ub=avail,
                    name=f"CURTAILMENT_{t}_{w}"
                )

    def _create_deficit_vars(self):
        s = self.sistema
        for t in range(self.horizon_time):
            for b in range(s.NBAR):
                self.DEFICIT[t, b] = self.model.add_variable(
                    lb=0,
                    ub=1e6,          # sem limite superior efetivo
                    name=f"DEFICIT_{t}_{b}"
                )

    def _create_voltage_vars(self):
        s = self.sistema
        for t in range(self.horizon_time):
            for b in range(s.NBAR):
                self.V[t, b] = self.model.add_variable(
                    lb=0.95,
                    ub=1.05,
                    name=f"V_{t}_{b}"
                )
                self.model.add_linear_constraint(self.V[t, b] == 1.0, name=f"fix_V_{t}_{b}")

    def _create_angle_vars(self):
        s = self.sistema
        for t in range(self.horizon_time):
            for b in range(s.NBAR):
                self.ANG[t, b] = self.model.add_variable(
                    lb=-np.pi,
                    ub=np.pi,
                    name=f"ANG_{t}_{b}"
                )
            # Fixar barra slack
            self.model.add_linear_constraint(self.ANG[t, s.slack_idx] == 0.0, name=f"fix_ANG_slack_{t}")

    def _create_flow_vars(self):
        s = self.sistema
        for t in range(self.horizon_time):
            for e in range(s.NLIN):
                self.FLUXO_LIN[t, e] = self.model.add_variable(
                    lb=-s.FLIM[e],    # pu
                    ub=s.FLIM[e],     # pu
                    name=f"FLUXO_LIN_{t}_{e}"
                )
    def _create_battery_vars(self):
        """Cria variáveis de bateria para todos os períodos e preenche os dicionários."""
        if not self._battery_list:
            return
        s = self.sistema
        for t in range(self.horizon_time):
            for i, b in enumerate(self._battery_list):
                # Capacidade e limites (todos em pu) - garantir que são escalares
                cap = float(s.BATTERY_CAPACITY[b])                     # energia total
                min_soc = float(s.BATTERY_MIN_SOC[b]) * cap            # mínimo
                power_limit = float(s.BATTERY_POWER_LIMIT[b])          # potência de carga

                # Potência de descarga (pode ser diferente)
                power_out_attr = getattr(s, 'BATTERY_POWER_OUT', None)
                if power_out_attr is None:
                    power_out = power_limit
                else:
                    if isinstance(power_out_attr, (list, tuple, np.ndarray)):
                        power_out = float(power_out_attr[b])
                    elif isinstance(power_out_attr, dict):
                        power_out = float(power_out_attr.get(b, power_limit))
                    else:
                        power_out = float(power_out_attr)  # assume escalar

                # Variável de carga
                self.CHARGE[t, b] = self.model.add_variable(
                    lb=0.0,
                    ub=power_limit,
                    name=f"CHARGE_{t}_{b}"
                )
                # Variável de descarga
                self.DISCHARGE[t, b] = self.model.add_variable(
                    lb=0.0,
                    ub=power_out,
                    name=f"DISCHARGE_{t}_{b}"
                )
                # Variável de estado de carga (SOC)
                self.SOC[t, b] = self.model.add_variable(
                    lb=min_soc,
                    ub=cap,
                    name=f"SOC_{t}_{b}"
                )
                # Variável de operação líquida (descarga - carga)
                self.BatteryOperation[t, b] = self.model.add_variable(
                    lb=-power_out,
                    ub=power_out,
                    name=f"BatteryOperation_{t}_{b}"
                )

    # -------------------- Adição de restrições via classes externas --------------------
    def _add_all_constraints(self):
        s = self.sistema
        T = self.horizon_time

        # 1. Geradores térmicos
        ThermalGeneratorConstraints.add_constraints(
            model=self.model,
            T=T,
            NGER_CONV=s.NGER_CONV,
            PGER=self.PGER,
            pgmin_conv=s.PGMIN_CONV,
            pgmax_conv=s.PGMAX_CONV,
            pger_inicial_conv=s.PGER_INICIAL_CONV,
            ramp_up_mw=s.RAMP_UP,
            ramp_down_mw=s.RAMP_DOWN,
            SB=s.SB
        )

        # 2. Baterias
        if self._battery_list:
            BatteryConstraints.add_constraints(
                model=self.model,
                sistema=s,
                T=T,
                battery_list=self._battery_list,
                battery_index=self._battery_index,
                CHARGE=self.CHARGE,
                DISCHARGE=self.DISCHARGE,
                SOC=self.SOC,
                BatteryOperation=self.BatteryOperation,
                soc_inicial_list=self._soc_inicial_list,
                soc_final_list=self._soc_final_list if self._soc_final_list else None
            )

        # 3. Geradores eólicos
        if s.NGER_EOL > 0:
            WindGeneratorConstraints.add_constraints(
                model=self.model,
                T=T,
                NGER_EOL=s.NGER_EOL,
                PGWIND=self.PGWIND,
                CURTAILMENT=self.CURTAILMENT,
                PGWIND_AVAIL=self.PGWIND_AVAIL
            )

        # 4. Restrições elétricas
        # Mapeamento barra dos geradores eólicos
        if hasattr(s, 'bus_wind'):
            wind_gen_to_bar = s.bus_wind
        elif hasattr(s, 'BARPG_EOL'):
            wind_gen_to_bar = s.BARPG_EOL
        else:
            wind_gen_to_bar = [0] * s.NGER_EOL

        self.balance_constraints = ElectricConstraints.add_constraints(
            model=self.model,
            sistema=s,
            T=T,
            ANG=self.ANG,
            FLUXO_LIN=self.FLUXO_LIN,
            DEFICIT=self.DEFICIT,
            PLOAD=self.PLOAD,
            PGER=self.PGER,
            conv_gen_to_bar=s.BARPG_CONV,
            PGWIND=self.PGWIND if s.NGER_EOL > 0 else None,
            wind_gen_to_bar=wind_gen_to_bar,
            CHARGE=self.CHARGE if self._battery_list else None,
            DISCHARGE=self.DISCHARGE if self._battery_list else None,
            battery_list=self._battery_list,
            PERDAS_BARRA=self._perdas_calculadas if self.considerar_perdas else None,
            considerar_perdas=self.considerar_perdas
        )

    # ----------------------------------------------------------------------
    # Função objetivo
    # ----------------------------------------------------------------------
    def build_objective(self, cost_function: Optional[Callable] = None):
        """
        Define a função objetivo. Se não for fornecida, usa a soma dos custos horários:
            custo térmico + penalidade de déficit + penalidade de curtailment.
        A função customizada deve receber o modelo (self) e retornar uma expressão.
        """
        if cost_function is None:
            expr = 0.0
            # Custo térmico (USD/pu)
            for t in range(self.horizon_time):
                for g in range(self.sistema.NGER_CONV):
                    custo = float(self.sistema.CPG_CONV[g])          # garantir escalar
                    expr += custo * self.PGER[t, g]

            # Déficit
            custo_deficit = float(getattr(self.sistema, 'CPG_DEFICIT', 1000.0))
            for t in range(self.horizon_time):
                for b in range(self.sistema.NBAR):
                    expr += custo_deficit * self.DEFICIT[t, b]

            # Curtailment
            if self.sistema.NGER_EOL > 0:
                for t in range(self.horizon_time):
                    for w in range(self.sistema.NGER_EOL):
                        # Pega o custo para o gerador w, garantindo escalar
                        if hasattr(self.sistema, 'CPG_CURTAILMENT'):
                            custo_curtail = float(self.sistema.CPG_CURTAILMENT[w])
                        else:
                            custo_curtail = 100.0
                        expr += custo_curtail * self.CURTAILMENT[t, w]

            self.model.set_objective(expr, poi.ObjectiveSense.Minimize)
        else:
            obj_expr = cost_function(self)
            self.model.set_objective(obj_expr, poi.ObjectiveSense.Minimize)

    # ----------------------------------------------------------------------
    # Métodos para perdas iterativas
    # ----------------------------------------------------------------------
    def calculate_losses(self) -> np.ndarray:
        """Calcula perdas nas linhas baseado na solução atual (resultado em pu)."""
        s = self.sistema
        T = self.horizon_time
        perdas_barra = np.zeros((T, s.NBAR))
        for t in range(T):
            for e in range(s.NLIN):
                i = s.line_fr[e]
                j = s.line_to[e]
                fluxo_val = self.model.get_value(self.FLUXO_LIN[t, e])  # pu
                r = s.r_line[e]                                         # pu
                perdas_linha = r * (fluxo_val ** 2)                     # pu
                perdas_barra[t, i] += perdas_linha / 2
                perdas_barra[t, j] += perdas_linha / 2
        self._perdas_calculadas = perdas_barra
        return perdas_barra

    def update_losses(self, perdas_barra: np.ndarray) -> None:
        """Atualiza o vetor de perdas (pu)."""
        self._perdas_calculadas = perdas_barra

    def solve_iterative(self, solver_name: str = 'highs', tol: float = 1e-4,
                        max_iter: int = 50, write_lp: bool = True, **solver_args):
        """Resolve o modelo com iterações de perdas (ponto fixo)."""
        if not self.considerar_perdas:
            return self.solve(solver_name, write_lp=write_lp, **solver_args)

        # Primeira solução sem perdas
        self._perdas_calculadas = np.zeros((self.horizon_time, self.sistema.NBAR))
        raw = self.solve(solver_name, write_lp=write_lp, **solver_args)
        if not self._solved or self.model.get_model_attribute(
                poi.ModelAttribute.TerminationStatus) != poi.TerminationStatusCode.OPTIMAL:
            print("Primeira iteração: solução não ótima.")
            return raw

        ang_prev = np.array([[self.model.get_value(self.ANG[t, b])
                               for b in range(self.sistema.NBAR)]
                              for t in range(self.horizon_time)])

        for it in range(1, max_iter):
            perdas = self.calculate_losses()
            self.update_losses(perdas)

            # Remover restrições de balanço antigas
            for _, _, constr in self.balance_constraints:
                self.model.delete_constraint(constr)

            # Recriar restrições de balanço com novas perdas
            s = self.sistema
            T = self.horizon_time
            wind_gen_to_bar = getattr(s, 'bus_wind', getattr(s, 'BARPG_EOL', [0]*s.NGER_EOL))
            self.balance_constraints = ElectricConstraints.add_constraints(
                model=self.model,
                sistema=s,
                T=T,
                ANG=self.ANG,
                FLUXO_LIN=self.FLUXO_LIN,
                DEFICIT=self.DEFICIT,
                PLOAD=self.PLOAD,
                PGER=self.PGER,
                conv_gen_to_bar=s.BARPG_CONV,
                PGWIND=self.PGWIND if s.NGER_EOL > 0 else None,
                wind_gen_to_bar=wind_gen_to_bar,
                CHARGE=self.CHARGE if self._battery_list else None,
                DISCHARGE=self.DISCHARGE if self._battery_list else None,
                battery_list=self._battery_list,
                PERDAS_BARRA=perdas,
                considerar_perdas=self.considerar_perdas
            )

            raw = self.solve(solver_name, write_lp=False, **solver_args)
            if not self._solved or self.model.get_model_attribute(
                    poi.ModelAttribute.TerminationStatus) != poi.TerminationStatusCode.OPTIMAL:
                print(f"Iteração {it+1}: solução não ótima.")
                break

            ang_curr = np.array([[self.model.get_value(self.ANG[t, b])
                                   for b in range(self.sistema.NBAR)]
                                  for t in range(self.horizon_time)])
            diff = np.max(np.abs(ang_curr - ang_prev))
            print(f"Iteração {it+1}: diff = {diff:.6f}")
            if diff < tol:
                print(f"Convergência alcançada na iteração {it+1}.")
                break
            ang_prev = ang_curr.copy()

        return raw

    def solve(self, solver_name: str = 'highs', write_lp: bool = False, **solver_args):
        """Resolve o modelo uma única vez."""
        if self.model is None:
            raise RuntimeError("Modelo não construído. Chame build() primeiro.")
        if write_lp:
            import inspect
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            caller_filename = caller_frame.f_code.co_filename
            base = os.path.splitext(os.path.basename(caller_filename))[0]
            lp_filename = f"DATA/output/{base}_timecoupled.lp"
            os.makedirs(os.path.dirname(lp_filename), exist_ok=True)
            self.model.write(lp_filename)
            print(f"Modelo escrito em {lp_filename}")
        self.model.optimize()
        self._solved = True
        return self.model

    # ----------------------------------------------------------------------
    # Extração de resultados (converte pu → MW)
    # ----------------------------------------------------------------------
    def extract_results(self) -> TimeCoupledOPFResult:
        """Extrai os resultados da solução e retorna um objeto TimeCoupledOPFResult."""
        if not self._solved or self.model is None:
            raise RuntimeError("Modelo não resolvido. Execute solve() primeiro.")

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
                # Demanda (pu → MW)
                PLOAD_vals = (self.PLOAD[t, :] ).tolist()
                # Geração térmica
                PGER_vals = [self.model.get_value(self.PGER[t, g])  for g in range(s.NGER_CONV)]

                # Eólica
                if s.NGER_EOL > 0:
                    PGWIND_disponivel = (self.PGWIND_AVAIL[t, :] ).tolist()
                    PGWIND_vals = [self.model.get_value(self.PGWIND[t, w])  for w in range(s.NGER_EOL)]
                    CURTAILMENT_vals = [self.model.get_value(self.CURTAILMENT[t, w])  for w in range(s.NGER_EOL)]
                else:
                    PGWIND_disponivel = PGWIND_vals = CURTAILMENT_vals = []

                # Déficit
                DEFICIT_vals = [self.model.get_value(self.DEFICIT[t, b])  for b in range(s.NBAR)]

                # Baterias
                SOC_init = [0.0] * s.NBAR
                SOC_atual = [0.0] * s.NBAR
                BESS_operation = [0.0] * s.NBAR
                if self._battery_list:
                    for i, b in enumerate(self._battery_list):
                        # SOC inicial (antes do período)
                        if t == 0:
                            soc_init_val = self._soc_inicial_list[i]
                        else:
                            soc_init_val = self.model.get_value(self.SOC[t-1, b])
                        SOC_init[b] = soc_init_val   # MWh

                        # SOC atual (após o período)
                        soc_atual_val = self.model.get_value(self.SOC[t, b])
                        SOC_atual[b] = soc_atual_val 

                        # Operação líquida (descarga - carga)
                        charge = self.model.get_value(self.CHARGE[t, b]) 
                        discharge = self.model.get_value(self.DISCHARGE[t, b]) 
                        BESS_operation[b] = discharge - charge

                # Tensão (fixa em 1.0 pu)
                V = [1.0] * s.NBAR
                ANG = [self.model.get_value(self.ANG[t, b]) for b in range(s.NBAR)]  # rad
                FLUXO_LIN = [self.model.get_value(self.FLUXO_LIN[t, e])  for e in range(s.NLIN)]

                # Custos (déficit em MW, custo em $/MW = custo_pu / SB)
                custo_deficit_pu = getattr(s, 'CPG_DEFICIT', 1000.0)
                # Para obter $, usamos DEFICIT_pu * custo_deficit_pu
                DEFICIT_pu = [self.model.get_value(self.DEFICIT[t, b]) for b in range(s.NBAR)]
                CUSTO = [d_pu * custo_deficit_pu for d_pu in DEFICIT_pu]

                # CMO (não disponível diretamente)
                CMO = [0.0]

                # Perdas
                if self.considerar_perdas and self._perdas_calculadas is not None:
                    PERDAS_BARRA = (self._perdas_calculadas[t, :] ).tolist()
                else:
                    PERDAS_BARRA = [0.0] * s.NBAR

                snapshots.append(TimeCoupledOPFSnapshotResult(
                    dia=dia,
                    dia_semana=dia_semana,
                    hora=hora,
                    sucesso=True,
                    PLOAD=PLOAD_vals,
                    PGER=PGER_vals,
                    PGWIND_disponivel=PGWIND_disponivel,
                    PGWIND=PGWIND_vals,
                    CURTAILMENT=CURTAILMENT_vals,
                    SOC_init=SOC_init,
                    BESS_operation=BESS_operation,
                    SOC_atual=SOC_atual,
                    DEFICIT=DEFICIT_vals,
                    V=V,
                    ANG=ANG,
                    FLUXO_LIN=FLUXO_LIN,
                    CUSTO=CUSTO,
                    CMO=CMO,
                    PERDAS_BARRA=PERDAS_BARRA,
                    dia_semana_nome=dia_semana_nome
                ))

            except Exception as e:
                print(f"Erro ao extrair snapshot t={t} (dia={dia}, hora={hora}): {e}")
                traceback.print_exc()
                snapshots.append(TimeCoupledOPFSnapshotResult(
                    dia=dia,
                    hora=hora,
                    sucesso=False,
                    mensagem=str(e),
                    dia_semana=dia_semana,
                    dia_semana_nome=dia_semana_nome
                ))

        sucesso_global = all(s.sucesso for s in snapshots)
        return TimeCoupledOPFResult(
            snapshots=snapshots,
            sucesso_global=sucesso_global,
            mensagem_global="OK" if sucesso_global else "Falhas na extração de alguns snapshots"
        )

    # ----------------------------------------------------------------------
    # Método de conveniência
    # ----------------------------------------------------------------------
    def solve_multiday(self,
                       solver_name: str = 'highs',
                       fator_carga: Optional[np.ndarray] = None,
                       fator_vento: Optional[np.ndarray] = None,
                       soc_inicial: Union[float, List[float]] = 0.5,
                       soc_final: Optional[Union[float, List[float]]] = None,
                       cost_function: Optional[Callable] = None,
                       cen_id: Optional[str] = None,
                       tol: float = 1e-4,
                       max_iter: int = 50,
                       write_lp: bool = True):
        """
        Executa todo o processo: construir, resolver (com perdas) e opcionalmente salvar no banco.
        """
        self.build(fator_carga, fator_vento, soc_inicial, soc_final)
        self.build_objective(cost_function)

        if self.considerar_perdas:
            raw = self.solve_iterative(solver_name, tol=tol, max_iter=max_iter, write_lp=write_lp)
        else:
            raw = self.solve(solver_name, write_lp=write_lp)

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


# =============================================================================
# Exemplo de uso (main)
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
    print("SIMULAÇÃO COM MODELO INTEGRADO NO TEMPO (DC OPF)")
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
    print(f"   ✓ Baterias: {len(getattr(sistema, 'BARRAS_COM_BATERIA', []))}")

    # -------------------------------------------------------------------------
    # 2. Parâmetros da simulação
    # -------------------------------------------------------------------------
    n_dias = 1
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
    print("\n3. Configurando banco de dados...")
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
    # 6. Gerar fatores de carga e vento
    # -------------------------------------------------------------------------
    seed = secrets.randbits(32)
    avaliador = EvaluateFactors(
        sistema=sistema,
        n_dias=n_dias,
        n_horas=n_horas,
        carga_incerteza=0.2,
        vento_variacao=0.1,
        seed=seed
    )
    fatores_carga, fatores_vento = avaliador.gerar_tudo()

    # -------------------------------------------------------------------------
    # 7. Exemplo de função objetivo customizada (opcional)
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

    print("\n4. Resolvendo modelo integrado com perdas iterativas...")
    raw = modelo.solve_multiday(
        solver_name='highs',
        fator_carga=fatores_carga,
        fator_vento=fatores_vento,
        soc_inicial=SOC_inicial,
        soc_final=SOC_final,
        cost_function=cost_func,
        cen_id=cen_id,
        tol=1e-4,
        max_iter=10,
        write_lp=True
    )

    status = modelo.model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
    print(f"\nStatus da solução: {status}")

    # -------------------------------------------------------------------------
    # 8. Extrair e exibir resumo
    # -------------------------------------------------------------------------
    resultados = modelo.extract_results()
    print(f"\nSucesso global: {resultados.sucesso_global}")
    print(f"Snapshots extraídos: {len(resultados.snapshots)}")

    custo_total = sum(sum(snap.CUSTO) for snap in resultados.snapshots if snap.sucesso)
    print(f"Custo total aproximado (apenas déficit): {custo_total:.2f} $")

    if modelo._battery_list:
        prim_batt = modelo._battery_list[0]
        soc_final_val = modelo.model.get_value(modelo.SOC[T-1, prim_batt]) * sistema.SB
        print(f"SOC final da bateria {prim_batt}: {soc_final_val:.3f} MWh")

    print("\n" + "=" * 70)
    print("EXECUÇÃO CONCLUÍDA")
    print("=" * 70)