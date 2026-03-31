import os
import sys
import numpy as np
import pyoptinterface as poi
from pyoptinterface import highs
from typing import List, Union, Optional, Dict, Callable
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SOLVER.OPF_DC.RES.ThermalGeneratorConstraints import ThermalGeneratorConstraints
from SOLVER.OPF_DC.RES.WindGeneratorConstraints import WindGeneratorConstraints
from SOLVER.OPF_DC.RES.EletricConstraints import ElectricConstraints
from SOLVER.OPF_DC.RES.BatteryConstraints import BatteryConstraints
from DB.DBmodel_OPF import TimeCoupledOPFSnapshotResult


class DCOPFSnapshot:
    """
    Modelo DC-OPF para um único instante (snapshot) usando pyoptinterface.
    Utiliza as classes de restrição externas para térmicas, eólicas, elétricas e baterias.
    """

    def __init__(self, sistema, db_handler=None, considerar_perdas: bool = True):
        self.sistema = sistema
        self.db_handler = db_handler
        self.considerar_perdas = considerar_perdas

        self.model = None
        self._solved = False

        self._init_system_arrays()

        # Estruturas para variáveis
        self.var_lists = {}          # nome -> lista (ordem original)
        self.var_indices = {}         # nome -> dict {índice original: variável}

        # Dicionários com chave (t, idx) para uso nas classes de restrição (t=0)
        self.PGER_dict = {}
        self.PGWIND_dict = {}
        self.CURTAILMENT_dict = {}
        self.DEFICIT_dict = {}
        self.ANG_dict = {}
        self.FLUXO_LIN_dict = {}
        self.CHARGE_dict = {}
        self.DISCHARGE_dict = {}
        self.BatteryOperation_dict = {}
        self.SOC_dict = {}            # SOC ao final do período

        # Parâmetros atualizáveis (pu)
        self.PLOAD = None             # demanda por barra [pu]
        self.PGWIND_AVAIL = None      # disponibilidade eólica por gerador [pu]
        self._losses = None            # perdas por barra [pu]

        # Lista de SOC inicial para cada bateria (parâmetro)
        self.soc_inicial_list = []

        # Restrições de balanço (para iterações de perdas)
        self.balance_constraints = []

    def _init_system_arrays(self):
        s = self.sistema
        self.n_bus = s.NBAR
        self.n_thermal = s.NGER_CONV
        self.n_wind = s.NGER_EOL
        self.n_line = s.NLIN
        self.n_battery = len(getattr(s, 'BARRAS_COM_BATERIA', []))

        self.thermal_bus = np.array(s.BARPG_CONV, dtype=int)
        self.wind_bus = np.array(getattr(s, 'bus_wind', getattr(s, 'BARPG_EOL', [])), dtype=int)

        self.line_from = np.array(s.line_fr, dtype=int)
        self.line_to = np.array(s.line_to, dtype=int)
        self.line_x = np.array(s.x_line, dtype=float)
        self.line_r = np.array(s.r_line, dtype=float) if hasattr(s, 'r_line') else np.zeros(self.n_line)
        self.line_flow_max = np.array(s.FLIM, dtype=float)

        self.thermal_pmin = np.array(s.PGMIN_CONV, dtype=float)
        self.thermal_pmax = np.array(s.PGMAX_CONV, dtype=float)
        self.thermal_cost = np.array(getattr(s, 'CUSTO_GER', [50.0] * self.n_thermal), dtype=float)

        self.battery_buses = np.array(getattr(s, 'BARRAS_COM_BATERIA', []), dtype=int)
        if self.n_battery > 0:
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
        self.base_load = np.array(s.PLOAD, dtype=float)

    # ----------------------------------------------------------------------
    # Construção da física
    # ----------------------------------------------------------------------
    def build_physics(self,
                      fator_carga: Optional[Union[float, np.ndarray]] = None,
                      fator_vento: Optional[Union[float, np.ndarray]] = None,
                      soc_baterias: Optional[Dict[int, float]] = None) -> None:
        """
        Constrói as variáveis e restrições físicas do problema.

        Args:
            fator_carga: Fator de carga (escalar ou array por barra) – adimensional.
            fator_vento: Fator de vento (escalar ou array por gerador eólico) – adimensional.
            soc_baterias: Dicionário com SOC inicial (fração da capacidade) para cada barra com bateria.
        """
        self.model = highs.Model()

        self._process_factors(fator_carga, fator_vento)
        self._process_battery_initial_soc(soc_baterias)

        # Criar variáveis e preencher os dicionários (t=0)
        self._create_v_vars()
        self._create_thermal_vars()
        self._create_wind_vars()
        self._create_deficit_vars()
        self._create_angle_vars()
        self._create_flow_vars()
        self._create_battery_vars()  

        # Adicionar restrições usando as classes externas
        self._add_external_constraints()

        self._solved = False

    def _process_factors(self, fator_carga, fator_vento):
        s = self.sistema
        if fator_carga is None:
            fator_carga = 1.0
        if np.isscalar(fator_carga):
            fator_carga_array = np.ones(self.n_bus) * fator_carga
        else:
            fator_carga_array = np.asarray(fator_carga)
            assert fator_carga_array.shape == (self.n_bus,)
        self.PLOAD = self.base_load * fator_carga_array

        if self.n_wind > 0:
            if fator_vento is None:
                fator_vento = 1.0
            if np.isscalar(fator_vento):
                fator_vento_array = np.ones(self.n_wind) * fator_vento
            else:
                fator_vento_array = np.asarray(fator_vento)
                assert fator_vento_array.shape == (self.n_wind,)
            self.PGWIND_AVAIL = np.array(s.PGWIND_disponivel) * fator_vento_array
        else:
            self.PGWIND_AVAIL = np.array([])

    def _process_battery_initial_soc(self, soc_baterias):
        """Processa SOC inicial das baterias (fração → energia em pu)."""
        if self.n_battery == 0:
            return
        self.soc_inicial_list = []
        for i, bus in enumerate(self.battery_buses):
            if soc_baterias and bus in soc_baterias:
                frac = soc_baterias[bus]
            else:
                frac = 0.5
            self.soc_inicial_list.append(frac * self.battery_capacity[i])  # energia [pu]

    # -------------------- Criação de variáveis --------------------
    def _create_thermal_vars(self):
        """Geração térmica (pu)."""
        self.var_lists['p_thermal'] = []
        for g in range(self.n_thermal):
            var = self.model.add_variable(
                lb=self.thermal_pmin[g],
                ub=self.thermal_pmax[g],
                name=f"p_thermal_{g}"
            )
            self.var_lists['p_thermal'].append(var)
            self.PGER_dict[(0, g)] = var
        self.var_indices['p_thermal'] = {g: var for g, var in enumerate(self.var_lists['p_thermal'])}

    def _create_wind_vars(self):
        if self.n_wind == 0:
            return
        self.var_lists['p_wind'] = []
        self.var_lists['curtailment'] = []
        for w in range(self.n_wind):
            p_wind = self.model.add_variable(
                lb=0,
                ub=self.PGWIND_AVAIL[w],
                name=f"p_wind_{w}"
            )
            curtail = self.model.add_variable(
                lb=0,
                ub=self.PGWIND_AVAIL[w],
                name=f"curtailment_{w}"
            )
            self.var_lists['p_wind'].append(p_wind)
            self.var_lists['curtailment'].append(curtail)
            self.PGWIND_dict[(0, w)] = p_wind
            self.CURTAILMENT_dict[(0, w)] = curtail
        self.var_indices['p_wind'] = {w: self.var_lists['p_wind'][w] for w in range(self.n_wind)}
        self.var_indices['curtailment'] = {w: self.var_lists['curtailment'][w] for w in range(self.n_wind)}

    def _create_deficit_vars(self):
        self.var_lists['deficit'] = []
        for b in range(self.n_bus):
            var = self.model.add_variable(
                lb=0,
                ub=1e6,
                name=f"deficit_{b}"
            )
            self.var_lists['deficit'].append(var)
            self.DEFICIT_dict[(0, b)] = var
        self.var_indices['deficit'] = {b: self.var_lists['deficit'][b] for b in range(self.n_bus)}

    def _create_v_vars(self):
        self.var_lists['v_pu'] = []
        for b in range(self.n_bus):
            var = self.model.add_variable(
                lb=0.95,
                ub=1.05,
                name=f"v_pu_{b}"
            )
            self.var_lists['v_pu'].append(var)
        self.var_indices['v_pu'] = {b: self.var_lists['v_pu'][b] for b in range(self.n_bus)}

    def _create_angle_vars(self):
        self.var_lists['ang_pu'] = []
        for b in range(self.n_bus):
            var = self.model.add_variable(
                lb=-np.pi,
                ub=np.pi,
                name=f"ang_pu_{b}"
            )
            self.var_lists['ang_pu'].append(var)
            self.ANG_dict[(0, b)] = var
        self.model.add_linear_constraint(
            self.var_lists['ang_pu'][self.slack_bus] == 0.0,
            name="fix_slack_angle"
        )
        self.var_indices['ang_pu'] = {b: self.var_lists['ang_pu'][b] for b in range(self.n_bus)}

    def _create_flow_vars(self):
        self.var_lists['flow'] = []
        for e in range(self.n_line):
            var = self.model.add_variable(
                lb=-self.line_flow_max[e],
                ub=self.line_flow_max[e],
                name=f"flow_{e}"
            )
            self.var_lists['flow'].append(var)
            self.FLUXO_LIN_dict[(0, e)] = var
        self.var_indices['flow'] = {e: self.var_lists['flow'][e] for e in range(self.n_line)}

    def _create_battery_vars(self):
        """Cria variáveis de bateria (CHARGE, DISCHARGE, SOC, BatteryOperation) para t=0."""
        if self.n_battery == 0:
            return
        self.var_lists['charge'] = []
        self.var_lists['discharge'] = []
        self.var_lists['soc'] = []
        self.var_lists['battery_op'] = []
        for i, bus in enumerate(self.battery_buses):
            power_limit = self.battery_power_limit[i]
            cap = self.battery_capacity[i]
            min_soc = self.battery_min_soc_frac[i] * cap

            ch = self.model.add_variable(lb=0, ub=power_limit, name=f"charge_{bus}")
            dch = self.model.add_variable(lb=0, ub=power_limit, name=f"discharge_{bus}")
            soc = self.model.add_variable(lb=min_soc, ub=cap, name=f"soc_{bus}")
            op = self.model.add_variable(lb=-power_limit, ub=power_limit, name=f"battery_op_{bus}")

            self.var_lists['charge'].append(ch)
            self.var_lists['discharge'].append(dch)
            self.var_lists['soc'].append(soc)
            self.var_lists['battery_op'].append(op)

            # Preencher dicionários com chave (0, bus)
            self.CHARGE_dict[(0, bus)] = ch
            self.DISCHARGE_dict[(0, bus)] = dch
            self.SOC_dict[(0, bus)] = soc
            self.BatteryOperation_dict[(0, bus)] = op

            # Relação entre operação líquida e potências (pode ser feita depois nas restrições, mas faremos aqui)
            self.model.add_linear_constraint(op == dch - ch, name=f"battery_link_{bus}")

        self.var_indices['charge'] = {bus: ch for bus, ch in zip(self.battery_buses, self.var_lists['charge'])}
        self.var_indices['discharge'] = {bus: dch for bus, dch in zip(self.battery_buses, self.var_lists['discharge'])}
        self.var_indices['soc'] = {bus: soc for bus, soc in zip(self.battery_buses, self.var_lists['soc'])}
        self.var_indices['battery_op'] = {bus: op for bus, op in zip(self.battery_buses, self.var_lists['battery_op'])}

    # -------------------- Restrições físicas (usando classes externas) --------------------
    def _add_external_constraints(self):
        # 1. Restrições térmicas
        if self.n_thermal > 0:
            ThermalGeneratorConstraints.add_constraints(
                model=self.model,
                T=1,
                NGER_CONV=self.n_thermal,
                PGER=self.PGER_dict,
                pgmin_conv=self.thermal_pmin,
                pgmax_conv=self.thermal_pmax,
                pger_inicial_conv=self.sistema.PGER_INICIAL_CONV,  # array em pu
                ramp_up_mw=self.sistema.RAMP_UP,
                ramp_down_mw=self.sistema.RAMP_DOWN,
                SB=self.sistema.SB
            )

        # 2. Restrições eólicas
        if self.n_wind > 0:
            WindGeneratorConstraints.add_constraints(
                model=self.model,
                T=1,
                NGER_EOL=self.n_wind,
                PGWIND=self.PGWIND_dict,
                CURTAILMENT=self.CURTAILMENT_dict,
                PGWIND_AVAIL=self.PGWIND_AVAIL.reshape(1, -1)  # array (1, n_wind)
            )

        # 3. Restrições elétricas (fluxo, balanço)
        PLOAD_2d = self.PLOAD.reshape(1, -1)
        losses_2d = None
        if self.considerar_perdas and self._losses is not None:
            losses_2d = self._losses.reshape(1, -1)

        wind_gen_to_bar = self.wind_bus.tolist() if self.n_wind > 0 else None

        self.balance_constraints = ElectricConstraints.add_constraints(
            model=self.model,
            sistema=self.sistema,
            T=1,
            ANG=self.ANG_dict,
            FLUXO_LIN=self.FLUXO_LIN_dict,
            DEFICIT=self.DEFICIT_dict,
            PLOAD=PLOAD_2d,
            PGER=self.PGER_dict,
            conv_gen_to_bar=self.thermal_bus.tolist(),
            PGWIND=self.PGWIND_dict if self.n_wind > 0 else None,
            wind_gen_to_bar=wind_gen_to_bar,
            CHARGE=self.CHARGE_dict if self.n_battery > 0 else None,
            DISCHARGE=self.DISCHARGE_dict if self.n_battery > 0 else None,
            battery_list=self.battery_buses.tolist() if self.n_battery > 0 else None,
            PERDAS_BARRA=losses_2d,
            considerar_perdas=self.considerar_perdas
        )

        # 4. Restrições de bateria (usando a classe BatteryConstraints)
        if self.n_battery > 0:
            BatteryConstraints.add_constraints(
                model=self.model,
                sistema=self.sistema,
                T=1,
                battery_list=self.battery_buses.tolist(),
                battery_index=None,  # não utilizado
                CHARGE=self.CHARGE_dict,
                DISCHARGE=self.DISCHARGE_dict,
                SOC=self.SOC_dict,
                BatteryOperation=self.BatteryOperation_dict,
                soc_inicial_list=self.soc_inicial_list,
                soc_final_list=None  # não impomos SOC final no snapshot
            )

    # ----------------------------------------------------------------------
    # Função objetivo
    # ----------------------------------------------------------------------
    def build_objective(self, cost_function: Optional[Callable] = None):
        if cost_function is None:
            expr = 0.0
            for g in range(self.n_thermal):
                expr += self.thermal_cost[g] * self.var_lists['p_thermal'][g]
            custo_deficit = getattr(self.sistema, 'CPG_DEFICIT', 1000.0)
            for b in range(self.n_bus):
                expr += custo_deficit * self.var_lists['deficit'][b]
            self.model.set_objective(expr, poi.ObjectiveSense.Minimize)
        else:
            obj_expr = cost_function(self)
            self.model.set_objective(obj_expr, poi.ObjectiveSense.Minimize)

    # ----------------------------------------------------------------------
    # Métodos públicos para compatibilidade
    # ----------------------------------------------------------------------
    def build(self, fator_carga=None, fator_vento=None, soc_baterias=None):
        self.build_physics(fator_carga, fator_vento, soc_baterias)
        self.build_objective()

    def add_FOB(self):
        self.build_objective()

    # ----------------------------------------------------------------------
    # Resolução
    # ----------------------------------------------------------------------
    def solve(self, solver_name: str = 'highs', write_lp: bool = False, **solver_args):
        if self.model is None:
            raise RuntimeError("Modelo não construído.")
        if write_lp:
            import inspect
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            caller_filename = caller_frame.f_code.co_filename
            base = os.path.splitext(os.path.basename(caller_filename))[0]
            lp_filename = f"DATA/output_CUR_Oficial/{base}_snapshot.lp"
            os.makedirs(os.path.dirname(lp_filename), exist_ok=True)
            self.model.write(lp_filename)
            print(f"Modelo escrito em {lp_filename}")
        self.model.optimize()
        self._solved = True
        return self.model

    def solve_iterative(self, solver_name: str = 'highs', tol: float = 1e-4,
                        max_iter: int = 50, write_lp: bool = False, **solver_args):
        if not self.considerar_perdas:
            return self.solve(solver_name, write_lp=write_lp, **solver_args)

        self._losses = np.zeros(self.n_bus)
        raw = self.solve(solver_name, write_lp=write_lp, **solver_args)
        if not self._solved or self.model.get_model_attribute(
                poi.ModelAttribute.TerminationStatus) != poi.TerminationStatusCode.OPTIMAL:
            print("Primeira iteração: solução não ótima.")
            return raw

        ang_prev = np.array([self.model.get_value(self.var_lists['ang_pu'][b]) for b in range(self.n_bus)])

        for it in range(1, max_iter):
            perdas = self.calculate_losses_from_flow()
            self.update_losses(perdas)

            # Remover restrições de balanço antigas
            for _, _, constr in self.balance_constraints:
                self.model.delete_constraint(constr)

            # Recriar restrições de balanço com novas perdas
            losses_2d = perdas.reshape(1, -1)
            wind_gen_to_bar = self.wind_bus.tolist() if self.n_wind > 0 else None
            self.balance_constraints = ElectricConstraints.add_constraints(
                model=self.model,
                sistema=self.sistema,
                T=1,
                ANG=self.ANG_dict,
                FLUXO_LIN=self.FLUXO_LIN_dict,
                DEFICIT=self.DEFICIT_dict,
                PLOAD=self.PLOAD.reshape(1, -1),
                PGER=self.PGER_dict,
                conv_gen_to_bar=self.thermal_bus.tolist(),
                PGWIND=self.PGWIND_dict if self.n_wind > 0 else None,
                wind_gen_to_bar=wind_gen_to_bar,
                CHARGE=self.CHARGE_dict if self.n_battery > 0 else None,
                DISCHARGE=self.DISCHARGE_dict if self.n_battery > 0 else None,
                battery_list=self.battery_buses.tolist() if self.n_battery > 0 else None,
                PERDAS_BARRA=losses_2d,
                considerar_perdas=self.considerar_perdas
            )

            raw = self.solve(solver_name, write_lp=False, **solver_args)
            if not self._solved or self.model.get_model_attribute(
                    poi.ModelAttribute.TerminationStatus) != poi.TerminationStatusCode.OPTIMAL:
                print(f"Iteração {it+1}: solução não ótima.")
                break

            ang_curr = np.array([self.model.get_value(self.var_lists['ang_pu'][b]) for b in range(self.n_bus)])
            diff = np.max(np.abs(ang_curr - ang_prev))
            print(f"Iteração {it+1}: diff = {diff:.6f}")
            if diff < tol:
                print(f"Convergência alcançada na iteração {it+1}.")
                break
            ang_prev = ang_curr.copy()

        return raw

    def calculate_losses_from_flow(self):
        perdas_barra = np.zeros(self.n_bus)
        for e in range(self.n_line):
            flow = self.model.get_value(self.var_lists['flow'][e])
            r = self.line_r[e]
            loss = r * (flow ** 2)
            i = self.line_from[e]
            j = self.line_to[e]
            perdas_barra[i] += loss / 2
            perdas_barra[j] += loss / 2
        return perdas_barra

    def update_losses(self, perdas_barra):
        self._losses = perdas_barra

    # ----------------------------------------------------------------------
    # Extração de resultados
    # ----------------------------------------------------------------------
    def extract_results(self, hora: int = 0, dia: int = 0,
                        cen_id: Optional[str] = None) -> TimeCoupledOPFSnapshotResult:
        if not self._solved:
            raise RuntimeError("Modelo não resolvido.")

        s = self.sistema
        dias_nomes = ["domingo", "segunda", "terça", "quarta", "quinta", "sexta", "sábado"]
        dia_semana = ((dia) % 7) + 1
        dia_semana_nome = dias_nomes[dia_semana-1]

        try:
            PLOAD_vals = (self.PLOAD).tolist() if self.PLOAD is not None else []
            PGER_vals = [self.model.get_value(self.var_lists['p_thermal'][g]) for g in range(self.n_thermal)]

            if self.n_wind > 0:
                PGWIND_disponivel = (self.PGWIND_AVAIL).tolist()
                PGWIND_vals = [self.model.get_value(self.var_lists['p_wind'][w]) for w in range(self.n_wind)]
                CURTAILMENT_vals = [self.model.get_value(self.var_lists['curtailment'][w])for w in range(self.n_wind)]
            else:
                PGWIND_disponivel = PGWIND_vals = CURTAILMENT_vals = []

            DEFICIT_vals = [self.model.get_value(self.var_lists['deficit'][b]) for b in range(self.n_bus)]

            # Baterias
            SOC_init = [0.0] * self.n_bus
            SOC_atual = [0.0] * self.n_bus
            BESS_operation = [0.0] * self.n_bus
            if self.n_battery > 0:
                for i, bus in enumerate(self.battery_buses):
                    # SOC inicial (parâmetro)
                    SOC_init[bus] = self.soc_inicial_list[i]
                    # SOC final (variável)
                    soc_val = self.model.get_value(self.SOC_dict[(0, bus)])
                    SOC_atual[bus] = soc_val 
                    charge = self.model.get_value(self.CHARGE_dict[(0, bus)]) 
                    discharge = self.model.get_value(self.DISCHARGE_dict[(0, bus)])
                    BESS_operation[bus] = discharge - charge

            V = [self.model.get_value(self.var_lists['v_pu'][b]) for b in range(self.n_bus)]
            ANG = [self.model.get_value(self.var_lists['ang_pu'][b]) for b in range(self.n_bus)]
            FLUXO_LIN = [self.model.get_value(self.var_lists['flow'][e]) for e in range(self.n_line)]

            DEFICIT_pu = [self.model.get_value(self.var_lists['deficit'][b]) for b in range(self.n_bus)]
            custo_deficit_pu = getattr(s, 'CPG_DEFICIT', 1000.0)
            CUSTO = [d_pu * custo_deficit_pu for d_pu in DEFICIT_pu]

            CMO = [0.0]

            if self.considerar_perdas and self._losses is not None:
                PERDAS_BARRA = (self._losses).tolist()
            else:
                PERDAS_BARRA = [0.0] * self.n_bus

            return TimeCoupledOPFSnapshotResult(
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
            )

        except Exception as e:
            print(f"Erro ao extrair snapshot: {e}")
            traceback.print_exc()
            return TimeCoupledOPFSnapshotResult(
                dia=dia,
                hora=hora,
                sucesso=False,
                mensagem=str(e),
                dia_semana=dia_semana,
                dia_semana_nome=dia_semana_nome
            )

    # ----------------------------------------------------------------------
    # Método de conveniência
    # ----------------------------------------------------------------------
    def solve_snapshot(self,
                       solver_name: str = 'highs',
                       fator_carga: Optional[Union[float, np.ndarray]] = None,
                       fator_vento: Optional[Union[float, np.ndarray]] = None,
                       soc_baterias: Optional[Dict[int, float]] = None,
                       cost_function: Optional[Callable] = None,
                       hora: int = 0,
                       dia: int = 0,
                       cen_id: Optional[str] = None,
                       tol: float = 1e-4,
                       max_iter: int = 50,
                       write_lp: bool = False,
                       verify: bool = False):
        self.build_physics(fator_carga, fator_vento, soc_baterias)
        self.build_objective(cost_function)

        if self.considerar_perdas:
            raw = self.solve_iterative(solver_name, tol=tol, max_iter=max_iter, write_lp=write_lp)
        else:
            raw = self.solve(solver_name, write_lp=write_lp)

        if self.db_handler is not None and cen_id is not None:
            resultado = self.extract_results(hora=hora, dia=dia, cen_id=cen_id)
            dia_str = f"{dia+1}"
            self.db_handler.save_hourly_result(
                resultado=resultado,
                sistema=self.sistema,
                hora=hora,
                solver_name=solver_name,
                dia=dia_str,
                cen_id=cen_id
            )

        if verify:
            self.print_verification_report(tol=tol)

        return raw


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

    print("=" * 70)
    print("SIMULAÇÃO SNAPSHOT - DC OPF (COM BATTERYCONSTRAINTS)")
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
    print(f"   ✓ Geradores convencionais: {sistema.NGER_CONV}")
    print(f"   ✓ Geradores eólicos: {sistema.NGER_EOL}")
    print(f"   ✓ Baterias: {len(getattr(sistema, 'BARRAS_COM_BATERIA', []))}")

    # -------------------------------------------------------------------------
    # 2. Configurar banco de dados
    # -------------------------------------------------------------------------
    print("\n2. Configurando banco de dados...")
    
    db_handler = OPF_DBHandler('DATA/output_CUR_Oficial/resultados_snapshot.db')
    db_handler.create_tables()
    cen_id = datetime.now().strftime('%Y%m%d%H%M%S') + "_snapshot"
    print(f"   ✓ Cenário ID: {cen_id}")

    # -------------------------------------------------------------------------
    # 3. Criar modelo snapshot
    # -------------------------------------------------------------------------
    modelo = DCOPFSnapshot(
        sistema=sistema,
        db_handler=db_handler,
        considerar_perdas=True
    )

    # -------------------------------------------------------------------------
    # 4. Definir parâmetros para uma hora específica
    # -------------------------------------------------------------------------
    hora_desejada = 0

    seed = secrets.randbits(32)
    from UTILS.EvaluateFactors import EvaluateFactors

    avaliador = EvaluateFactors(
        sistema=sistema,
        n_dias=1,
        n_horas=24,
        carga_incerteza=0.2,
        vento_variacao=0.1,
        seed=seed
    )
    fatores_carga_completo, fatores_vento_completo = avaliador.gerar_tudo()

    fator_carga_hora = fatores_carga_completo[0, hora_desejada, :]
    fator_vento_hora = fatores_vento_completo[0, hora_desejada, :] if sistema.NGER_EOL > 0 else 1.0

    print(f"\n3. Parâmetros para Hora {hora_desejada}:")
    print(f"   Fator de carga médio: {np.mean(fator_carga_hora):.3f}")
    if sistema.NGER_EOL > 0:
        print(f"   Fator de vento médio: {np.mean(fator_vento_hora):.3f}")

    # SOC inicial das baterias (fração da capacidade)
    soc_baterias = {b: 0.5 for b in sistema.BARRAS_COM_BATERIA}
    print(f"   SOC inicial das baterias: {soc_baterias}")

    # -------------------------------------------------------------------------
    # 5. Resolver
    # -------------------------------------------------------------------------
    print("\n4. Resolvendo snapshot...")
    raw = modelo.solve_snapshot(
        solver_name='highs',
        fator_carga=fator_carga_hora,
        fator_vento=fator_vento_hora,
        soc_baterias=soc_baterias,
        hora=hora_desejada,
        dia=0,
        cen_id=cen_id,
        tol=1e-4,
        max_iter=10,
        write_lp=True,
        verify=False
    )

    status = modelo.model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
    print(f"\nStatus da solução: {status}")

    # -------------------------------------------------------------------------
    # 6. Extrair e exibir resultados
    # -------------------------------------------------------------------------
    if status == poi.TerminationStatusCode.OPTIMAL:
        resultado = modelo.extract_results(hora=hora_desejada, cen_id=cen_id)

        print(f"\n5. Resultados para Hora {hora_desejada}:")
        print(f"   Demanda total: {sum(resultado.PLOAD):.3f} pu")
        print(f"   Perdas: {sum(resultado.PERDAS_BARRA):.3f} pu")
        print(f"   Geração térmica total: {sum(resultado.PGER):.3f} pu")
        if sistema.NGER_EOL > 0:
            print(f"   Geração eólica total: {sum(resultado.PGWIND):.3f} pu")
            print(f"   Curtailment total: {sum(resultado.CURTAILMENT):.3f} pu")
        print(f"   Déficit total: {sum(resultado.DEFICIT):.3f} pu")

        if sistema.BARRAS_COM_BATERIA:
            for b in sistema.BARRAS_COM_BATERIA:
                print(f"   Bateria barra {b}:")
                print(f"      operação = {resultado.BESS_operation[b]:.3f} pu")
                print(f"      SOC inicial = {resultado.SOC_init[b]:.3f} pu")
                print(f"      SOC final   = {resultado.SOC_atual[b]:.3f} pu")

        print(f"   CMO (barra slack): {resultado.CMO[0]:.2f} $/MWh")

    print("\n" + "=" * 70)
    print("EXECUÇÃO CONCLUÍDA")
    print("=" * 70)