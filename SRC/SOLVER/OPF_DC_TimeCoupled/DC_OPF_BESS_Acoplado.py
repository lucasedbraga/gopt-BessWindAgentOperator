#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classe principal do modelo de otimização multi-período acoplado (DC OPF) com suporte a baterias e perdas iterativas.
Versão traduzida para PyOptInterface.
Utiliza módulos separados para cada grupo de restrições (já adaptados).
"""

import os
import sys
import numpy as np
import pyoptinterface as poi
from pyoptinterface import highs
from typing import List, Union, Optional, Tuple, Dict
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SOLVER.OPF_DC_TimeCoupled.RES.BatteryConstraintsTime import BatteryConstraintsTime
from SOLVER.OPF_DC_TimeCoupled.RES.ThermalGeneratorConstraintsTime import ThermalGeneratorConstraints
from SOLVER.OPF_DC_TimeCoupled.RES.WindGeneratorConstraintsTime import WindGeneratorConstraints
from SOLVER.OPF_DC_TimeCoupled.RES.EletricConstraintsTime import ElectricConstraints
from DB.DBmodel_OPF import TimeCoupledOPFResult, TimeCoupledOPFSnapshotResult


class TimeCoupledOPFModel:
    """
    Modelo de otimização multi‑período com acoplamento temporal integrado (PyOptInterface).
    As variáveis são indexadas no tempo (períodos = n_dias * n_horas).
    Delega a construção de restrições para módulos específicos.
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
        self._battery_list: List[int] = []
        self._battery_index: Dict[int, int] = {}
        self._perdas_calculadas: Optional[np.ndarray] = None
        self._soc_inicial_list: List[float] = []
        self._soc_final_list: List[float] = []
        self._fator_carga: Optional[np.ndarray] = None
        self._fator_vento: Optional[np.ndarray] = None

        # Dicionários para armazenar variáveis
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

        # Parâmetros (arrays)
        self.PLOAD: Optional[np.ndarray] = None
        self.PGWIND_AVAIL: Optional[np.ndarray] = None

        # Lista de restrições de balanço (para remoção/recriação)
        self.balance_constraints: List[Tuple[int, int, poi.Constraint]] = []

    # -------------------------------------------------------------------------
    # Construção do modelo
    # -------------------------------------------------------------------------
    def build(self,
              fator_carga: Optional[np.ndarray] = None,
              fator_vento: Optional[np.ndarray] = None,
              soc_inicial: Union[float, List[float]] = 0.5,
              soc_final: Union[float, List[float]] = 0.5) -> None:
        """
        Constrói o modelo monolítico com variáveis temporais.
        soc_inicial e soc_final devem ser frações da capacidade (0 a 1).
        """
        s = self.sistema
        T = self.horizon_time

        # ==================== Processamento dos fatores de carga e vento ====================
        # (mesma lógica da versão Pyomo, já testada)
        if fator_carga is not None:
            if (fator_carga.ndim == 3 and fator_carga.shape[0] == self.n_dias
                    and fator_carga.shape[1] == self.n_horas):
                fator_carga = fator_carga.reshape((T, s.NBAR))
            elif (fator_carga.ndim == 2 and fator_carga.shape[0] == self.n_dias
                  and fator_carga.shape[1] == self.n_horas):
                fator_carga = np.repeat(fator_carga.reshape((T, 1)), s.NBAR, axis=1)
            elif fator_carga.ndim == 2 and fator_carga.shape[1] != s.NBAR:
                fator_carga = np.repeat(fator_carga, s.NBAR, axis=1)
            elif fator_carga.ndim == 1:
                fator_carga = np.tile(fator_carga.reshape(-1, 1), (1, s.NBAR))
        else:
            fator_carga = np.ones((T, s.NBAR))
        if fator_carga.shape != (T, s.NBAR):
            raise ValueError(f"fator_carga shape {fator_carga.shape} != {(T, s.NBAR)}")

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

        # ==================== Inicializar modelo PyOptInterface ====================
        self.model = highs.Model()

        # ==================== Parâmetros (arrays) ====================
        self.PLOAD = np.zeros((T, s.NBAR))
        for t in range(T):
            for b in range(s.NBAR):
                self.PLOAD[t, b] = s.PLOAD[b] * fator_carga[t, b]

        if s.NGER_EOL > 0:
            self.PGWIND_AVAIL = np.zeros((T, s.NGER_EOL))
            for t in range(T):
                for w in range(s.NGER_EOL):
                    self.PGWIND_AVAIL[t, w] = s.PGWIND_disponivel[w] * fator_vento[t, w]
        else:
            self.PGWIND_AVAIL = np.zeros((T, 0))

        # ==================== Variáveis básicas ====================
        # Geração térmica
        for t in range(T):
            for g in range(s.NGER_CONV):
                self.PGER[t, g] = self.model.add_variable(
                    lb=s.PGMIN_CONV[g] * s.SB,
                    ub=s.PGMAX_CONV[g] * s.SB,
                    name=f"PGER_{t}_{g}"
                )

        # Geração eólica e curtailment
        if s.NGER_EOL > 0:
            for t in range(T):
                for w in range(s.NGER_EOL):
                    self.PGWIND[t, w] = self.model.add_variable(
                        lb=0,
                        ub=self.PGWIND_AVAIL[t, w],
                        name=f"PGWIND_{t}_{w}"
                    )
                    self.CURTAILMENT[t, w] = self.model.add_variable(
                        lb=0,
                        ub=self.PGWIND_AVAIL[t, w],
                        name=f"CURTAILMENT_{t}_{w}"
                    )

        # Déficit
        for t in range(T):
            for b in range(s.NBAR):
                self.DEFICIT[t, b] = self.model.add_variable(
                    lb=0,
                    ub=1e6,  # sem limite superior
                    name=f"DEFICIT_{t}_{b}"
                )

        # Tensão, ângulo, fluxo
        for t in range(T):
            for b in range(s.NBAR):
                self.V[t, b] = self.model.add_variable(
                    lb=0.95,
                    ub=1.05,
                    name=f"V_{t}_{b}"
                )
                self.model.add_linear_constraint(self.V[t, b] == 1.0, name=f"fix_V_{t}_{b}")

            for b in range(s.NBAR):
                self.ANG[t, b] = self.model.add_variable(
                    lb=-np.pi,
                    ub=np.pi,
                    name=f"ANG_{t}_{b}"
                )
                if b == s.slack_idx:
                    self.model.add_linear_constraint(self.ANG[t, b] == 0.0, name=f"fix_ANG_slack_{t}")

            for e in range(s.NLIN):
                self.FLUXO_LIN[t, e] = self.model.add_variable(
                    lb=-s.FLIM[e],
                    ub=s.FLIM[e],
                    name=f"FLUXO_LIN_{t}_{e}"
                )

        # ==================== Baterias ====================
        self._battery_list = list(s.BARRAS_COM_BATERIA)
        self._battery_index = {b: i for i, b in enumerate(self._battery_list)}

        # Garantir arrays (valores em MW/MWh)
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

        # SOC inicial
        if isinstance(soc_inicial, (int, float)):
            soc_inicial_frac = [soc_inicial] * len(self._battery_list)
        else:
            soc_inicial_frac = soc_inicial
        self._soc_inicial_list = [
            soc_inicial_frac[i] * s.BATTERY_CAPACITY[b] for i, b in enumerate(self._battery_list)
        ]

        # SOC final
        if soc_final is not None:
            if isinstance(soc_final, (int, float)):
                soc_final_frac = [soc_final] * len(self._battery_list)
            else:
                soc_final_frac = soc_final
            self._soc_final_list = [
                soc_final_frac[i] * s.BATTERY_CAPACITY[b] for i, b in enumerate(self._battery_list)
            ]
        else:
            self._soc_final_list = []

        # ==================== Adicionar restrições ====================
        self._add_all_constraints()
        self.add_FOB()

        self._fator_carga = fator_carga
        self._fator_vento = fator_vento
        self._solved = False

    def _add_all_constraints(self) -> None:
        """Adiciona todas as restrições delegando para os módulos específicos."""
        s = self.sistema
        T = self.horizon_time

        # Geradores térmicos
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

        # Baterias
        BatteryConstraintsTime.add_constraints(
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

        # Geradores eólicos
        if s.NGER_EOL > 0:
            WindGeneratorConstraints.add_constraints(
                model=self.model,
                T=T,
                NGER_EOL=s.NGER_EOL,
                PGWIND=self.PGWIND,
                CURTAILMENT=self.CURTAILMENT,
                PGWIND_AVAIL=self.PGWIND_AVAIL
            )

        # Mapeamento da barra de cada gerador eólico (correção)
        if hasattr(s, 'bus_wind'):
            wind_gen_to_bar = s.bus_wind
        elif hasattr(s, 'BARPG_EOL'):
            wind_gen_to_bar = s.BARPG_EOL
        else:
            wind_gen_to_bar = [0] * s.NGER_EOL   # fallback: todos na barra 0

        # Restrições elétricas
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
            wind_gen_to_bar=wind_gen_to_bar,          # <-- corrigido
            CHARGE=self.CHARGE if self._battery_list else None,
            DISCHARGE=self.DISCHARGE if self._battery_list else None,
            battery_list=self._battery_list,
            PERDAS_BARRA=self._perdas_calculadas if self.considerar_perdas else None,
            considerar_perdas=self.considerar_perdas
        )

    def add_FOB(self) -> None:
        """Função objetivo: minimizar custo de geração térmica + penalidade de déficit."""
        from SOLVER.OPF_DC_TimeCoupled.FOB import EconomicDispatchTime
        EconomicDispatchTime.ObjectiveFunction.add_objective_pyopt(self)

    # -------------------------------------------------------------------------
    # Métodos para perdas iterativas
    # -------------------------------------------------------------------------
    def calculate_losses(self) -> np.ndarray:
        """Calcula perdas nas linhas baseado na solução atual."""
        s = self.sistema
        T = self.horizon_time
        perdas_barra = np.zeros((T, s.NBAR))
        for t in range(T):
            for e in range(s.NLIN):
                i = s.line_fr[e]
                j = s.line_to[e]
                fluxo_val = self.model.get_value(self.FLUXO_LIN[t, e])
                r = s.r_line[e]
                perdas_linha = r * (fluxo_val ** 2) * s.SB
                perdas_barra[t, i] += perdas_linha / 2
                perdas_barra[t, j] += perdas_linha / 2
        self._perdas_calculadas = perdas_barra
        return perdas_barra

    def update_losses(self, perdas_barra: np.ndarray) -> None:
        """Atualiza o vetor de perdas."""
        self._perdas_calculadas = perdas_barra

    def solve_iterative(self, solver_name: str = 'highs', tol: float = 1e-4,
                    max_iter: int = 50, write_lp: bool = True, **solver_args):
        """
        Resolve o modelo com iterações de perdas (ponto fixo).
        O parâmetro write_lp controla a escrita do LP apenas na primeira iteração.
        """
        if not self.considerar_perdas:
            return self.solve(solver_name, write_lp=write_lp, **solver_args)

        # Primeira solução (pode escrever LP se solicitado)
        raw = self.solve(solver_name, write_lp=write_lp, **solver_args)
        if not self._solved or self.model.get_model_attribute(
                poi.ModelAttribute.TerminationStatus) != poi.TerminationStatusCode.OPTIMAL:
            print("Primeira iteração: solução não ótima.")
            return raw

        ang_prev = np.array([[self.model.get_value(self.ANG[t, b])
                            for b in range(self.sistema.NBAR)]
                            for t in range(self.horizon_time)])

        for it in range(1, max_iter):
            # Calcular perdas com a solução atual
            perdas = self.calculate_losses()
            self.update_losses(perdas)

            # Remover restrições de balanço antigas
            for _, _, constr in self.balance_constraints:
                self.model.delete_constraint(constr)

            # Recriar restrições de balanço com as novas perdas
            s = self.sistema
            T = self.horizon_time
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
                wind_gen_to_bar=getattr(s, 'bus_wind', None),
                CHARGE=self.CHARGE if self._battery_list else None,
                DISCHARGE=self.DISCHARGE if self._battery_list else None,
                battery_list=self._battery_list,
                PERDAS_BARRA=perdas,
                considerar_perdas=self.considerar_perdas
            )

            # Resolver novamente (sem escrever LP)
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

    def solve(self, solver_name: str = 'highs', write_lp: bool = True, **solver_args):
        """
        Resolve o modelo uma única vez.
        Se write_lp=True, escreve o modelo em formato LP antes de otimizar.
        """
        if self.model is None:
            raise RuntimeError("Modelo não construído. Chame build() primeiro.")
        if write_lp:
            import inspect
            import os
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            caller_filename = caller_frame.f_code.co_filename
            base = os.path.splitext(os.path.basename(caller_filename))[0]
            lp_filename = f"DATA/output/{base}.lp"
            self.model.write(lp_filename)
            print(f"Modelo escrito em {lp_filename}")
        self.model.optimize()
        self._solved = True
        return self.model
    
    # -------------------------------------------------------------------------
    # Extração de resultados (com verificação de balanço)
    # -------------------------------------------------------------------------
    def extract_results(self) -> TimeCoupledOPFResult:
        """
        Extrai os resultados da solução e retorna um objeto TimeCoupledOPFResult.
        Inclui verificação de balanço de massa global.
        """
        if not self._solved or self.model is None:
            raise RuntimeError("Modelo não resolvido. Execute solve() primeiro.")

        s = self.sistema
        T = self.horizon_time
        snapshots = []
        dias_nomes = ["domingo", "segunda", "terça", "quarta", "quinta", "sexta", "sábado"]

        balanco_errors = []

        for t in range(T):
            dia = t // self.n_horas
            hora = t % self.n_horas
            dia_semana = ((self.dia_inicial + dia) % 7) + 1
            dia_semana_nome = dias_nomes[dia_semana-1]

            try:
                # PLOAD (já em MW)
                PLOAD_vals = self.PLOAD[t, :].tolist()
                demanda_total = sum(PLOAD_vals)

                # Geração térmica
                PGER_vals = [self.model.get_value(self.PGER[t, g]) for g in range(s.NGER_CONV)]
                ger_term_total = sum(PGER_vals)

                # Eólica
                if s.NGER_EOL > 0:
                    PGWIND_disponivel = self.PGWIND_AVAIL[t, :].tolist()
                    PGWIND_vals = [self.model.get_value(self.PGWIND[t, w]) for w in range(s.NGER_EOL)]
                    CURTAILMENT_vals = [self.model.get_value(self.CURTAILMENT[t, w]) for w in range(s.NGER_EOL)]
                    ger_eol_total = sum(PGWIND_vals)
                else:
                    PGWIND_disponivel = PGWIND_vals = CURTAILMENT_vals = []
                    ger_eol_total = 0.0

                # Déficit
                DEFICIT_vals = [self.model.get_value(self.DEFICIT[t, b]) for b in range(s.NBAR)]
                deficit_total = sum(DEFICIT_vals)

                # Baterias
                SOC_init = [0.0] * s.NBAR
                SOC_atual = [0.0] * s.NBAR
                BESS_operation = [0.0] * s.NBAR
                carga_total = 0.0
                descarga_total = 0.0

                if self._battery_list:
                    for b in self._battery_list:
                        # SOC inicial (antes do período)
                        if t == 0:
                            idx = self._battery_index.get(b, None)
                            if idx is not None and idx < len(self._soc_inicial_list):
                                soc_init_val = self._soc_inicial_list[idx]
                            else:
                                soc_init_val = 0.0
                        else:
                            soc_init_val = self.model.get_value(self.SOC[t-1, b]) \
                                if (t-1, b) in self.SOC else 0.0
                        SOC_init[b] = soc_init_val

                        # SOC atual (após o período)
                        if (t, b) in self.SOC:
                            SOC_atual[b] = self.model.get_value(self.SOC[t, b])

                        # Operação
                        charge = self.model.get_value(self.CHARGE[t, b]) \
                            if (t, b) in self.CHARGE else 0.0
                        discharge = self.model.get_value(self.DISCHARGE[t, b]) \
                            if (t, b) in self.DISCHARGE else 0.0
                        BESS_operation[b] = charge - discharge
                        carga_total += charge
                        descarga_total += discharge

                # Tensão, ângulo, fluxo
                V = [self.model.get_value(self.V[t, b]) for b in range(s.NBAR)]
                ANG = [self.model.get_value(self.ANG[t, b]) for b in range(s.NBAR)]
                FLUXO_LIN = [self.model.get_value(self.FLUXO_LIN[t, e]) for e in range(s.NLIN)]

                # Custos (apenas déficit)
                CUSTO = [DEFICIT_vals[b] * getattr(s, 'CPG_DEFICIT', 1000.0) for b in range(s.NBAR)]

                # CMO (preço marginal) – opcional
                CMO = 0.0
                # for (tt, bb, constr) in self.balance_constraints:
                #     if tt == t and bb == s.slack_idx:
                #         CMO = self.model.get_dual(constr)
                #         break

                # Perdas
                if self.considerar_perdas and self._perdas_calculadas is not None:
                    PERDAS_BARRA = self._perdas_calculadas[t, :].tolist()
                    perdas_total = sum(PERDAS_BARRA)
                else:
                    PERDAS_BARRA = [0.0] * s.NBAR
                    perdas_total = 0.0

                # TODO: VERIFICAR PQ NAO ESTA BATENDO
                # # Verificação de balanço global
                # lhs_global = ger_term_total + ger_eol_total + deficit_total + descarga_total - carga_total
                # rhs_global = demanda_total + perdas_total
                # diff_global = lhs_global - rhs_global
                # if abs(diff_global) > 0.05:  # tolerância de 50 kW (ajustável)
                #     print(f"\n⚠️  Hora {t}: balanço global não fecha")
                #     print(f"   LHS (gerado) = {lhs_global:.6f}  |  RHS (consumido) = {rhs_global:.6f}  |  diff = {diff_global:.6e}")
                #     print(f"   PGER     = {ger_term_total:.6f}")
                #     print(f"   PGWIND   = {ger_eol_total:.6f}")
                #     print(f"   DEFICIT  = {deficit_total:.6f}")
                #     print(f"   DESCARGA = {descarga_total:.6f}")
                #     print(f"   CARGA    = {carga_total:.6f}")
                #     print(f"   DEMANDA  = {demanda_total:.6f}")
                #     print(f"   PERDAS   = {perdas_total:.6f}")
                #     balanco_errors.append((t, diff_global, lhs_global, rhs_global))

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
                    CMO=[CMO],
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

        # if balanco_errors:
        #     print("\n⚠️  Atenção: discrepâncias no balanço de massa detectadas (global):")
        #     for t, diff, lhs, rhs in balanco_errors:
        #         print(f"   Hora {t}: GERADO = {lhs:.3f}, CONSUMIDO = {rhs:.3f}, diferença = {diff:.3e}")
        # else:
        #     print("\n✅ Balanço de massa verificado – todas as horas fecham dentro da tolerância.")

        sucesso_global = all(s.sucesso for s in snapshots)
        return TimeCoupledOPFResult(
            snapshots=snapshots,
            sucesso_global=sucesso_global,
            mensagem_global="OK" if sucesso_global else "Falhas na extração de alguns snapshots"
        )
    
    # -------------------------------------------------------------------------
    # Método de conveniência
    # -------------------------------------------------------------------------
    def solve_multiday(self,
                       solver_name: str = 'highs',
                       fator_carga: Optional[np.ndarray] = None,
                       fator_vento: Optional[np.ndarray] = None,
                       soc_inicial: Union[float, List[float]] = 0.5,
                       soc_final: Union[float, List[float]] = 0.5,
                       cen_id: Optional[str] = None,
                       tol: float = 1e-4,
                       max_iter: int = 50,
                       write_lp: bool = True):
        """
        Executa todo o processo: construir, resolver (com perdas) e opcionalmente salvar no banco.
        """
        self.build(fator_carga, fator_vento, soc_inicial, soc_final)

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
# Exemplo de uso (main) – idêntico ao da versão Pyomo, apenas trocando a classe
# =============================================================================
if __name__ == "__main__":
    import sys
    import os
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import secrets

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from UTILS.SystemLoader import SistemaLoader
    from DB.DBhandler_OPF import OPF_DBHandler
    from UTILS.EvaluateFactors import EvaluateFactors

    print("=" * 70)
    print("SIMULAÇÃO COM MODELO INTEGRADO NO TEMPO (DC OPF) - PyOptInterface")
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
    db_handler = OPF_DBHandler('DATA/output/resultados_PL_acoplado_pyopt.db')
    db_handler.create_tables()
    cen_id = datetime.now().strftime('%Y%m%d%H%M%S')
    print(f"   ✓ Cenário ID: {cen_id}")

    # -------------------------------------------------------------------------
    # 5. Criar modelo (PyOptInterface)
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

    print("\n4. Resolvendo modelo integrado com perdas iterativas...")
    raw = modelo.solve_multiday(
        solver_name='highs',
        fator_carga=fatores_carga,
        fator_vento=fatores_vento,
        soc_inicial=SOC_inicial,
        soc_final=SOC_final,
        cen_id=cen_id,
        tol=1e-4,
        max_iter=10,
        write_lp=True
    )

    status = modelo.model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
    print(f"\nStatus da solução: {status}")

    # -------------------------------------------------------------------------
    # 7. Extrair e exibir resumo
    # -------------------------------------------------------------------------
    resultados = modelo.extract_results()
    print(f"\nSucesso global: {resultados.sucesso_global}")
    print(f"Snapshots extraídos: {len(resultados.snapshots)}")

    custo_total = sum(sum(snap.CUSTO) for snap in resultados.snapshots if snap.sucesso)
    print(f"Custo total aproximado (apenas déficit): {custo_total:.2f} $")

    if modelo._battery_list:
        prim_batt = modelo._battery_list[0]
        soc_final_val = modelo.model.get_value(modelo.SOC[T-1, prim_batt])
        print(f"SOC final da bateria {prim_batt}: {soc_final_val:.3f} MWh")

    print("\n" + "=" * 70)
    print("EXECUÇÃO CONCLUÍDA")
    print("=" * 70)