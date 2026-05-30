import os
import sys
import numpy as np
import pyomo.environ as pyo
from typing import List, Union, Optional, Dict, Callable
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Classes de restrição (agora Pyomo)
from SOLVER.OPF_AC.RES.ThermalGeneratorConstraints import ThermalGeneratorConstraints
from SOLVER.OPF_AC.RES.WindGeneratorConstraints import WindGeneratorConstraints
from SOLVER.OPF_AC.RES.BatteryConstraints import BatteryConstraints
from DB.DBmodel_OPF import TimeCoupledOPFSnapshotResult


class ACOPFSnapshot:
    """
    Modelo AC-OPF (não linear) para um único instante usando Pyomo + IPOPT.
    """

    def __init__(self, sistema, db_handler=None):
        self.sistema = sistema
        self.db_handler = db_handler
        self.model = None
        self._solved = False
        self._init_system_arrays()
        self._build_admittance_matrix()

        self.var_lists = {}
        self.var_indices = {}
        self.PGER_dict = {}
        self.QGER_dict = {}
        self.PGWIND_dict = {}
        self.CURTAILMENT_dict = {}
        self.DEFICIT_dict = {}
        self.ANG_dict = {}
        self.CHARGE_dict = {}
        self.DISCHARGE_dict = {}
        self.BatteryOperation_dict = {}
        self.SOC_dict = {}

        self.PLOAD = None
        self.QLOAD = None
        self.PGWIND_AVAIL = None
        self.soc_inicial_list = []

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
        self.line_r = np.array(s.r_line, dtype=float)
        self.line_x = np.array(s.x_line, dtype=float)
        self.line_flow_max = np.array(s.FLIM, dtype=float)

        self.thermal_pmin = np.array(s.PGMIN_CONV, dtype=float)
        self.thermal_pmax = np.array(s.PGMAX_CONV, dtype=float)
        self.thermal_cost = np.array(getattr(s, 'CUSTO_GER', [50.0]*self.n_thermal), dtype=float)

        if hasattr(s, 'QGMIN_CONV') and hasattr(s, 'QGMAX_CONV'):
            self.thermal_qmin = np.array(s.QGMIN_CONV, dtype=float)
            self.thermal_qmax = np.array(s.QGMAX_CONV, dtype=float)
        else:
            self.thermal_qmin = -0.5 * self.thermal_pmax
            self.thermal_qmax =  0.5 * self.thermal_pmax

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
        self.base_Pload = np.array(s.PLOAD, dtype=float)
        self.base_Qload = np.array(s.QLOAD, dtype=float) if hasattr(s, 'QLOAD') else np.zeros(self.n_bus)

    def _build_admittance_matrix(self):
        self.G = np.zeros((self.n_bus, self.n_bus))
        self.B = np.zeros((self.n_bus, self.n_bus))
        for e in range(self.n_line):
            i = self.line_from[e]
            j = self.line_to[e]
            r = self.line_r[e]
            x = self.line_x[e]
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

    def build_physics(self,
                      fator_carga: Optional[Union[float, np.ndarray]] = None,
                      fator_vento: Optional[Union[float, np.ndarray]] = None,
                      soc_baterias: Optional[Dict[int, float]] = None) -> None:
        self.model = pyo.ConcreteModel(name="ACOPFSnapshot")
        self._process_factors(fator_carga, fator_vento)
        self._process_battery_initial_soc(soc_baterias)

        self._create_v_vars()
        self._create_angle_vars()
        self._create_thermal_vars()
        self._create_wind_vars()
        self._create_deficit_vars()
        self._create_battery_vars()

        self._add_balance_constraints()
        self._add_external_constraints()
        self._solved = False

    def _process_factors(self, fator_carga, fator_vento):
        if fator_carga is None:
            fator_carga = 1.0
        if np.isscalar(fator_carga):
            fator_carga_array = np.ones(self.n_bus) * fator_carga
        else:
            fator_carga_array = np.asarray(fator_carga)
            assert fator_carga_array.shape == (self.n_bus,)
        self.PLOAD = self.base_Pload * fator_carga_array
        self.QLOAD = self.base_Qload * fator_carga_array

        if self.n_wind > 0:
            if fator_vento is None:
                fator_vento = 1.0
            if np.isscalar(fator_vento):
                fator_vento_array = np.ones(self.n_wind) * fator_vento
            else:
                fator_vento_array = np.asarray(fator_vento)
                assert fator_vento_array.shape == (self.n_wind,)
            self.PGWIND_AVAIL = np.array(self.sistema.PGWIND_disponivel) * fator_vento_array
        else:
            self.PGWIND_AVAIL = np.array([])

    def _process_battery_initial_soc(self, soc_baterias):
        if self.n_battery == 0:
            return
        self.soc_inicial_list = []
        for i, bus in enumerate(self.battery_buses):
            frac = soc_baterias.get(bus, 0.5) if soc_baterias else 0.5
            self.soc_inicial_list.append(frac * self.battery_capacity[i])

    def _create_v_vars(self):
        self.var_lists['v_pu'] = []
        for b in range(self.n_bus):
            var = pyo.Var(bounds=(0.95, 1.05), initialize=1.0)
            setattr(self.model, f"v_pu_{b}", var)
            self.var_lists['v_pu'].append(var)
        self.var_indices['v_pu'] = {b: v for b, v in enumerate(self.var_lists['v_pu'])}

    def _create_angle_vars(self):
        self.var_lists['ang_pu'] = []
        for b in range(self.n_bus):
            var = pyo.Var(bounds=(-np.pi, np.pi), initialize=0.0)
            setattr(self.model, f"ang_pu_{b}", var)
            self.var_lists['ang_pu'].append(var)
            self.ANG_dict[(0, b)] = var
        self.model.fix_slack_angle = pyo.Constraint(
            expr=self.var_lists['ang_pu'][self.slack_bus] == 0.0
        )
        self.var_indices['ang_pu'] = {b: v for b, v in enumerate(self.var_lists['ang_pu'])}

    def _create_thermal_vars(self):
        self.var_lists['p_thermal'] = []
        self.var_lists['q_thermal'] = []
        for g in range(self.n_thermal):
            p_var = pyo.Var(bounds=(self.thermal_pmin[g], self.thermal_pmax[g]), initialize=0.0)
            q_var = pyo.Var(bounds=(self.thermal_qmin[g], self.thermal_qmax[g]), initialize=0.0)
            setattr(self.model, f"p_thermal_{g}", p_var)
            setattr(self.model, f"q_thermal_{g}", q_var)
            self.var_lists['p_thermal'].append(p_var)
            self.var_lists['q_thermal'].append(q_var)
            self.PGER_dict[(0, g)] = p_var
            self.QGER_dict[(0, g)] = q_var
        self.var_indices['p_thermal'] = {g: v for g, v in enumerate(self.var_lists['p_thermal'])}
        self.var_indices['q_thermal'] = {g: v for g, v in enumerate(self.var_lists['q_thermal'])}

    def _create_wind_vars(self):
        if self.n_wind == 0:
            return
        self.var_lists['p_wind'] = []
        self.var_lists['curtailment'] = []
        for w in range(self.n_wind):
            avail = self.PGWIND_AVAIL[w]
            p_wind = pyo.Var(bounds=(0, avail), initialize=0.0)
            curtail = pyo.Var(bounds=(0, avail), initialize=0.0)
            setattr(self.model, f"p_wind_{w}", p_wind)
            setattr(self.model, f"curtailment_{w}", curtail)
            self.var_lists['p_wind'].append(p_wind)
            self.var_lists['curtailment'].append(curtail)
            self.PGWIND_dict[(0, w)] = p_wind
            self.CURTAILMENT_dict[(0, w)] = curtail
        self.var_indices['p_wind'] = {w: v for w, v in enumerate(self.var_lists['p_wind'])}
        self.var_indices['curtailment'] = {w: v for w, v in enumerate(self.var_lists['curtailment'])}

    def _create_deficit_vars(self):
        self.var_lists['deficit'] = []
        for b in range(self.n_bus):
            var = pyo.Var(bounds=(0, 1e6), initialize=0.0)
            setattr(self.model, f"deficit_{b}", var)
            self.var_lists['deficit'].append(var)
            self.DEFICIT_dict[(0, b)] = var
        self.var_indices['deficit'] = {b: v for b, v in enumerate(self.var_lists['deficit'])}

    def _create_battery_vars(self):
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
            ch = pyo.Var(bounds=(0, power_limit), initialize=0.0)
            dch = pyo.Var(bounds=(0, power_limit), initialize=0.0)
            soc = pyo.Var(bounds=(min_soc, cap), initialize=min_soc)
            op = pyo.Var(bounds=(-power_limit, power_limit), initialize=0.0)
            setattr(self.model, f"charge_{bus}", ch)
            setattr(self.model, f"discharge_{bus}", dch)
            setattr(self.model, f"soc_{bus}", soc)
            setattr(self.model, f"battery_op_{bus}", op)
            self.var_lists['charge'].append(ch)
            self.var_lists['discharge'].append(dch)
            self.var_lists['soc'].append(soc)
            self.var_lists['battery_op'].append(op)
            self.CHARGE_dict[(0, bus)] = ch
            self.DISCHARGE_dict[(0, bus)] = dch
            self.SOC_dict[(0, bus)] = soc
            self.BatteryOperation_dict[(0, bus)] = op
            setattr(self.model, f"battery_link_{bus}", pyo.Constraint(expr=op == dch - ch))
        self.var_indices['charge'] = {b: v for b, v in zip(self.battery_buses, self.var_lists['charge'])}
        self.var_indices['discharge'] = {b: v for b, v in zip(self.battery_buses, self.var_lists['discharge'])}
        self.var_indices['soc'] = {b: v for b, v in zip(self.battery_buses, self.var_lists['soc'])}
        self.var_indices['battery_op'] = {b: v for b, v in zip(self.battery_buses, self.var_lists['battery_op'])}

    def _add_balance_constraints(self):
        m = self.model
        V = self.var_lists['v_pu']
        ANG = self.ANG_dict
        G, B = self.G, self.B
        PLOAD, QLOAD = self.PLOAD, self.QLOAD

        thermal_at_bus = [[] for _ in range(self.n_bus)]
        for g, bus in enumerate(self.thermal_bus):
            thermal_at_bus[bus].append(g)

        wind_at_bus = [[] for _ in range(self.n_bus)]
        if self.n_wind > 0:
            for w, bus in enumerate(self.wind_bus):
                wind_at_bus[bus].append(w)

        battery_set = set(self.battery_buses) if self.n_battery > 0 else set()

        def p_balance_rule(m, i):
            Vi = V[i]
            theta_i = ANG[(0, i)]
            expr = 0.0
            for j in range(self.n_bus):
                if G[i, j] == 0 and B[i, j] == 0:
                    continue
                delta = theta_i - ANG[(0, j)]
                expr += V[j] * (G[i, j] * pyo.cos(delta) + B[i, j] * pyo.sin(delta))
            P_inj = Vi * expr
            gen = sum(self.PGER_dict[(0, g)] for g in thermal_at_bus[i])
            if self.n_wind > 0:
                gen += sum(self.PGWIND_dict[(0, w)] for w in wind_at_bus[i])
            bat = 0.0
            if self.n_battery > 0 and i in battery_set:
                bat = self.DISCHARGE_dict[(0, i)] - self.CHARGE_dict[(0, i)]
            deficit = self.DEFICIT_dict[(0, i)]
            load = PLOAD[i]
            return P_inj - (gen + bat + deficit - load) == 0

        m.P_balance = pyo.Constraint(range(self.n_bus), rule=p_balance_rule)

        def q_balance_rule(m, i):
            Vi = V[i]
            theta_i = ANG[(0, i)]
            expr = 0.0
            for j in range(self.n_bus):
                if G[i, j] == 0 and B[i, j] == 0:
                    continue
                delta = theta_i - ANG[(0, j)]
                expr += V[j] * (G[i, j] * pyo.sin(delta) - B[i, j] * pyo.cos(delta))
            Q_inj = Vi * expr
            gen_q = sum(self.QGER_dict[(0, g)] for g in thermal_at_bus[i])
            load_q = QLOAD[i]
            return Q_inj - (gen_q - load_q) == 0

        m.Q_balance = pyo.Constraint(range(self.n_bus), rule=q_balance_rule)

    def _add_external_constraints(self):
        if self.n_thermal > 0:
            ThermalGeneratorConstraints.add_constraints(
                model=self.model, T=1, NGER_CONV=self.n_thermal,
                PGER=self.PGER_dict, QGER=self.QGER_dict,
                pgmin_conv=self.thermal_pmin, pgmax_conv=self.thermal_pmax,
                qgmin_conv=self.thermal_qmin, qgmax_conv=self.thermal_qmax,
                pger_inicial_conv=self.sistema.PGER_INICIAL_CONV,
                ramp_up_mw=self.sistema.RAMP_UP, ramp_down_mw=self.sistema.RAMP_DOWN,
                SB=self.sistema.SB
            )

        if self.n_wind > 0:
            WindGeneratorConstraints.add_constraints(
                model=self.model, T=1, NGER_EOL=self.n_wind,
                PGWIND=self.PGWIND_dict, CURTAILMENT=self.CURTAILMENT_dict,
                PGWIND_AVAIL=self.PGWIND_AVAIL.reshape(1, -1)
            )

        if self.n_battery > 0:
            BatteryConstraints.add_constraints(
                model=self.model, sistema=self.sistema, T=1,
                battery_list=self.battery_buses.tolist(),
                CHARGE=self.CHARGE_dict, DISCHARGE=self.DISCHARGE_dict,
                SOC=self.SOC_dict, BatteryOperation=self.BatteryOperation_dict,
                soc_inicial_list=self.soc_inicial_list,
                soc_final_list=None, daily_reset_to_initial=False
            )

    def build_objective(self, cost_function=None):
        if cost_function is None:
            expr = sum(self.thermal_cost[g] * self.var_lists['p_thermal'][g] for g in range(self.n_thermal))
            custo_deficit = getattr(self.sistema, 'CPG_DEFICIT', 1000.0)
            expr += sum(custo_deficit * self.var_lists['deficit'][b] for b in range(self.n_bus))
            self.model.obj = pyo.Objective(expr=expr, sense=pyo.minimize)
        else:
            self.model.obj = pyo.Objective(expr=cost_function(self), sense=pyo.minimize)

    def build(self, fator_carga=None, fator_vento=None, soc_baterias=None):
        self.build_physics(fator_carga, fator_vento, soc_baterias)
        self.build_objective()

    def add_FOB(self):
        self.build_objective()

    def solve(self, solver_name='ipopt', write_lp=False, **solver_args):
        if self.model is None:
            raise RuntimeError("Modelo não construído.")
        opt = pyo.SolverFactory(solver_name)
        results = opt.solve(self.model, tee=True)
        self._solved = (results.solver.status == pyo.SolverStatus.ok and
                        results.solver.termination_condition == pyo.TerminationCondition.optimal)
        return results

    def extract_results(self, hora=0, dia=0, cen_id=None):
        if not self._solved:
            raise RuntimeError("Modelo não resolvido.")

        s = self.sistema
        dias_nomes = ["domingo", "segunda", "terça", "quarta", "quinta", "sexta", "sábado"]
        dia_semana = ((dia) % 7) + 1
        dia_semana_nome = dias_nomes[dia_semana-1]

        try:
            PLOAD_vals = self.PLOAD.tolist()
            PGER_vals = [pyo.value(self.var_lists['p_thermal'][g]) for g in range(self.n_thermal)]

            if self.n_wind > 0:
                PGWIND_disponivel = self.PGWIND_AVAIL.tolist()
                PGWIND_vals = [pyo.value(self.var_lists['p_wind'][w]) for w in range(self.n_wind)]
                CURTAILMENT_vals = [pyo.value(self.var_lists['curtailment'][w]) for w in range(self.n_wind)]
            else:
                PGWIND_disponivel = PGWIND_vals = CURTAILMENT_vals = []

            DEFICIT_vals = [pyo.value(self.var_lists['deficit'][b]) for b in range(self.n_bus)]

            SOC_init = [0.0] * self.n_bus
            SOC_atual = [0.0] * self.n_bus
            BESS_operation = [0.0] * self.n_bus
            if self.n_battery > 0:
                for i, bus in enumerate(self.battery_buses):
                    SOC_init[bus] = self.soc_inicial_list[i]
                    SOC_atual[bus] = pyo.value(self.SOC_dict[(0, bus)])
                    charge = pyo.value(self.CHARGE_dict[(0, bus)])
                    discharge = pyo.value(self.DISCHARGE_dict[(0, bus)])
                    BESS_operation[bus] = discharge - charge

            V = [pyo.value(self.var_lists['v_pu'][b]) for b in range(self.n_bus)]
            ANG = [pyo.value(self.var_lists['ang_pu'][b]) for b in range(self.n_bus)]

            P_flow = np.zeros(self.n_line)
            Q_flow = np.zeros(self.n_line)
            for e in range(self.n_line):
                i = self.line_from[e]
                j = self.line_to[e]
                vi, vj = V[i], V[j]
                theta_i, theta_j = ANG[i], ANG[j]
                r, x = self.line_r[e], self.line_x[e]
                z2 = r*r + x*x
                g = r / z2
                b = -x / z2
                P_flow[e] = vi**2 * g - vi*vj*(g*np.cos(theta_i-theta_j) + b*np.sin(theta_i-theta_j))
                Q_flow[e] = -vi**2 * b - vi*vj*(g*np.sin(theta_i-theta_j) - b*np.cos(theta_i-theta_j))

            perdas_total = np.sum(P_flow) + np.sum(self.PLOAD) - np.sum(PGER_vals) - (
                np.sum(PGWIND_vals) if self.n_wind > 0 else 0
            ) + np.sum(DEFICIT_vals) + np.sum(BESS_operation)
            PERDAS_BARRA = [perdas_total / self.n_bus] * self.n_bus

            custo_deficit_pu = getattr(s, 'CPG_DEFICIT', 1000.0)
            CUSTO = [d * custo_deficit_pu for d in DEFICIT_vals]

            CMO = [0.0]

            return TimeCoupledOPFSnapshotResult(
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
                V=V, ANG=ANG,
                FLUXO_LIN=P_flow.tolist(),
                CUSTO=CUSTO, CMO=CMO,
                PERDAS_BARRA=PERDAS_BARRA,
                dia_semana_nome=dia_semana_nome
            )

        except Exception as e:
            print(f"Erro ao extrair snapshot: {e}")
            traceback.print_exc()
            return TimeCoupledOPFSnapshotResult(
                dia=dia, hora=hora, sucesso=False, mensagem=str(e),
                dia_semana=dia_semana, dia_semana_nome=dia_semana_nome
            )

    def solve_snapshot(self, solver_name='ipopt', fator_carga=None, fator_vento=None,
                       soc_baterias=None, cost_function=None, hora=0, dia=0,
                       cen_id=None, write_lp=False, verify=False):
        self.build_physics(fator_carga, fator_vento, soc_baterias)
        self.build_objective(cost_function)
        self.model.pprint()
        results = self.solve(solver_name, write_lp=write_lp)

        if self.db_handler is not None and cen_id is not None:
            resultado = self.extract_results(hora=hora, dia=dia, cen_id=cen_id)
            dia_str = f"{dia+1}"
            self.db_handler.save_hourly_result(
                resultado=resultado, sistema=self.sistema,
                hora=hora, solver_name=solver_name,
                dia=dia_str, cen_id=cen_id
            )

        if verify:
            self.print_verification_report()
        return results


# =============================================================================
# Exemplo de uso
# =============================================================================
if __name__ == "__main__":
    import sys, os, secrets
    from datetime import datetime

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from UTILS.SystemLoader import SistemaLoader
    from DB.DBhandler_OPF import OPF_DBHandler

    print("=" * 70)
    print("SIMULAÇÃO SNAPSHOT - AC OPF (Pyomo + IPOPT)")
    print("=" * 70)

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

    print("\n2. Configurando banco de dados...")
    db_handler = OPF_DBHandler('DATA/output/resultados_snapshot_AC.db')
    db_handler.create_tables()
    cen_id = datetime.now().strftime('%Y%m%d%H%M%S') + "_ac_snapshot"
    print(f"   ✓ Cenário ID: {cen_id}")

    modelo = ACOPFSnapshot(sistema=sistema, db_handler=db_handler)

    hora_desejada = 0
    seed = secrets.randbits(32)
    from UTILS.EvaluateFactors import EvaluateFactors

    avaliador = EvaluateFactors(sistema=sistema, n_dias=1, n_horas=24,
                                carga_incerteza=0.2, vento_variacao=0.1, seed=seed)
    fatores_carga_completo, fatores_vento_completo = avaliador.gerar_tudo()
    fator_carga_hora = fatores_carga_completo[0, hora_desejada, :]
    fator_vento_hora = fatores_vento_completo[0, hora_desejada, :] if sistema.NGER_EOL > 0 else 1.0

    print(f"\n3. Parâmetros para Hora {hora_desejada}:")
    print(f"   Fator de carga médio: {np.mean(fator_carga_hora):.3f}")
    if sistema.NGER_EOL > 0:
        print(f"   Fator de vento médio: {np.mean(fator_vento_hora):.3f}")

    soc_baterias = {b: 0.5 for b in sistema.BARRAS_COM_BATERIA}
    print(f"   SOC inicial das baterias: {soc_baterias}")

    print("\n4. Resolvendo snapshot AC...")
    raw = modelo.solve_snapshot(
        solver_name='ipopt',
        fator_carga=fator_carga_hora,
        fator_vento=fator_vento_hora,
        soc_baterias=soc_baterias,
        hora=hora_desejada, dia=0, cen_id=cen_id
    )

    if modelo._solved:
        resultado = modelo.extract_results(hora=hora_desejada, cen_id=cen_id)
        print(f"\n5. Resultados para Hora {hora_desejada}:")
        print(f"   Demanda ativa total:  {sum(resultado.PLOAD):.3f} pu")
        print(f"   Geração térmica ativa: {sum(resultado.PGER):.3f} pu")
        if sistema.NGER_EOL > 0:
            print(f"   Geração eólica total:   {sum(resultado.PGWIND):.3f} pu")
            print(f"   Curtailment total:      {sum(resultado.CURTAILMENT):.3f} pu")
        print(f"   Déficit total:          {sum(resultado.DEFICIT):.3f} pu")
        print(f"   Perdas ativas totais:   {sum(resultado.PERDAS_BARRA):.3f} pu")
        print(f"   Tensão mínima: {min(resultado.V):.3f} pu, máxima: {max(resultado.V):.3f} pu")
        print(f"   Ângulo slack (ref): {resultado.ANG[sistema.slack_idx]:.3f} rad")
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