import os
import sys
import numpy as np
import pyomo.environ as pyo
from typing import List, Union, Optional, Dict, Callable
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UTILS.EvaluateFactors import EvaluateFactors

from SOLVER.OPF_AC.RES.BalanceConstraints import AC_BalanceConstraints
from SOLVER.OPF_AC.RES.ThermalGeneratorConstraints import ThermalGeneratorConstraints
from SOLVER.OPF_AC.RES.WindGeneratorConstraints import WindGeneratorConstraints
from SOLVER.OPF_AC.RES.BatteryConstraints import BatteryConstraints

from DB.DBmodel_OPF import OPF_SnapshotResult



class ACOPF_Snapshot:
    """
    Modelo AC-OPF para um único instante usando Pyomo + IPOPT.
    """

    def __init__(self, sistema, db_handler=None):
        
        self.sistema = sistema
        self.db_handler = db_handler
        self.model = None
        self._solved = False
        
        self._build_SistemaEletrico_arrays()

        self._monta_matiz_admitancia()

        self.var_lists = {}
        self.var_indices = {}
        self.PGER_dict = {}
        self.QGER_dict = {}
        self.PGWIND_dict = {}
        self.CURTAILMENT_dict = {}
        self.DEFICIT_dict = {}
        self.V_dict = {}
        self.ANG_dict = {}
        self.CHARGE_dict = {}
        self.DISCHARGE_dict = {}
        self.BatteryOperation_dict = {}
        self.SOC_dict = {}

        self.PLOAD = None
        self.QLOAD = None
        self.PGWIND_AVAIL = None
        self.soc_inicial_list = []
        

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

    def _monta_matiz_admitancia(self):
        self.G = np.zeros((self.NBAR, self.NBAR))
        self.B = np.zeros((self.NBAR, self.NBAR))

        for LIN in range(self.NLIN):
            i = self.line_from[LIN]
            j = self.line_to[LIN]
            r = self.line_r[LIN]
            x = self.line_x[LIN]

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

    def _processa_multiplicadores(self, fator_carga, fator_vento,fator_soc_init):
        
        def processa_fator_carga(fator_carga):
            if fator_carga is None:
                fator_carga = 1.0
            if np.isscalar(fator_carga):
                fator_carga_array = np.ones(self.NBAR) * fator_carga
            else:
                fator_carga_array = np.asarray(fator_carga)
                assert fator_carga_array.shape == (self.NBAR,)

            self.PLOAD = self.base_Pload * fator_carga_array
            self.QLOAD = self.base_Qload * fator_carga_array

        def processa_fator_vento(fator_vento):
            if self.NGWD > 0:
                if fator_vento is None:
                    fator_vento = 1.0
                if np.isscalar(fator_vento):
                    fator_vento_array = np.ones(self.NGWD) * fator_vento
                else:
                    fator_vento_array = np.asarray(fator_vento)
                    assert fator_vento_array.shape == (self.NGWD,)
                self.PGWIND_AVAIL = np.array(self.sistema.PGWIND_disponivel) * fator_vento_array
            else:
                self.PGWIND_AVAIL = np.array([])
        
        def processa_soc_init(fator_soc_init):
            if self.NBESS == 0:
                return        
            self.soc_inicial_list = []
            for i, bus in enumerate(self.battery_buses):
                frac = soc_baterias.get(bus, 0.5) if soc_baterias else 0.5
                self.soc_inicial_list.append(frac * self.battery_capacity[i])

        processa_fator_carga(fator_carga)
        processa_fator_vento(fator_vento)
        processa_soc_init(fator_soc_init)

    def _add_VARS(self):

        def _create_VARx_V():
            self.var_lists['v_pu'] = []
            for b in range(self.NBAR):
                var = pyo.Var(bounds=(0.95, 1.05), initialize=1.0)
                setattr(self.model, f"V_pu_BAR_{b+1}", var)
                self.var_lists['v_pu'].append(var)
            self.var_indices['v_pu'] = {b: v for b, v in enumerate(self.var_lists['v_pu'])}

            for b, var in enumerate(self.var_lists['v_pu']):
                self.V_dict[(0, b)] = var

        def _create_VARx_ANG():
            self.var_lists['ang_pu'] = []
            for b in range(self.NBAR):
                var = pyo.Var(bounds=(-np.pi, np.pi), initialize=0.0)
                setattr(self.model, f"ANG_pu_BAR_{b+1}", var)
                self.var_lists['ang_pu'].append(var)
                self.ANG_dict[(0, b)] = var
            self.model.fix_slack_angle = pyo.Constraint(
                expr=self.var_lists['ang_pu'][self.slack_bus] == 0.0
            )
            self.var_indices['ang_pu'] = {b: v for b, v in enumerate(self.var_lists['ang_pu'])}

        def _create_VARx_PGER():

            self.var_lists['PGER_UTE'] = []
           
            for g in range(self.NUTE):
                p_var = pyo.Var(bounds=(self.thermal_pmin[g], self.thermal_pmax[g]), initialize=0.0)
               
                setattr(self.model, f"PGER_UTE_{g+1}", p_var)
                self.var_lists['PGER_UTE'].append(p_var)                
                self.PGER_dict[(0, g)] = p_var                
            self.var_indices['PGER_UTE'] = {g: v for g, v in enumerate(self.var_lists['PGER_UTE'])}
            
        def _create_VARx_QGER():

            self.var_lists['QGER_UTE'] = []

            for g in range(self.NUTE):
                q_var = pyo.Var(bounds=(self.thermal_qmin[g], self.thermal_qmax[g]), initialize=0.0)
                setattr(self.model, f"QGER_UTE_{g+1}", q_var)
                self.var_lists['QGER_UTE'].append(q_var)
                self.QGER_dict[(0, g)] = q_var
            self.var_indices['QGER_UTE'] = {g: v for g, v in enumerate(self.var_lists['QGER_UTE'])}

        def _create_VARx_DEF():
            self.var_lists['deficit'] = []
            for b in range(self.NBAR):
                var = pyo.Var(bounds=(0, 1e6), initialize=0.0)
                setattr(self.model, f"DEFICT_BAR_{b+1}", var)
                self.var_lists['deficit'].append(var)
                self.DEFICIT_dict[(0, b)] = var
            self.var_indices['deficit'] = {b: v for b, v in enumerate(self.var_lists['deficit'])}

        def _create_VARx_GWD():
            if self.NGWD == 0:
                return
            self.var_lists['p_wind'] = []
            self.var_lists['curtailment'] = []
            for w in range(self.NGWD):
                avail = self.PGWIND_AVAIL[w]
                p_wind = pyo.Var(bounds=(0, avail), initialize=0.0)
                curtail = pyo.Var(bounds=(0, avail), initialize=0.0)
                setattr(self.model, f"PGWD_{w}", p_wind)
                setattr(self.model, f"CURTAILMENT_{w}", curtail)
                self.var_lists['p_wind'].append(p_wind)
                self.var_lists['curtailment'].append(curtail)
                self.PGWIND_dict[(0, w)] = p_wind
                self.CURTAILMENT_dict[(0, w)] = curtail
            self.var_indices['p_wind'] = {w: v for w, v in enumerate(self.var_lists['p_wind'])}
            self.var_indices['curtailment'] = {w: v for w, v in enumerate(self.var_lists['curtailment'])}

        def _create_VARx_BESS():
            if self.NBESS == 0:
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
                setattr(self.model, f"charge_{bus+1}", ch)
                setattr(self.model, f"discharge_{bus+1}", dch)
                setattr(self.model, f"soc_{bus+1}", soc)
                setattr(self.model, f"battery_op_{bus+1}", op)
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

        
        _create_VARx_V()
        _create_VARx_ANG()
        _create_VARx_PGER()
        _create_VARx_QGER()
        _create_VARx_DEF()
        _create_VARx_GWD()
        _create_VARx_BESS()
   
    def _add_CONS(self):

        wind_gen_to_bar = None
        battery_list = None

        if self.NUTE > 0:
            ThermalGeneratorConstraints.add_constraints(
                model=self.model,
                T=1,
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

        if self.NGWD > 0:

            wind_gen_to_bar = self.wind_bus.tolist()

            WindGeneratorConstraints.add_constraints(
                model=self.model,
                T=1,
                NGER_GWD=self.NGWD,
                PGWIND=self.PGWIND_dict,
                CURTAILMENT=self.CURTAILMENT_dict,
                PGWIND_AVAIL=self.PGWIND_AVAIL.reshape(1, -1)
            )

        if self.NBESS > 0:

            battery_list = self.battery_buses.tolist()
            BatteryConstraints.add_constraints(
                model=self.model,
                sistema=self.sistema,
                T=1,
                battery_list=self.battery_buses.tolist(),
                CHARGE=self.CHARGE_dict,
                DISCHARGE=self.DISCHARGE_dict,
                SOC=self.SOC_dict,
                BatteryOperation=self.BatteryOperation_dict,
                soc_inicial_list=self.soc_inicial_list,
                soc_final_list=None,
                daily_reset_to_initial=False
            )
        

        AC_BalanceConstraints.add_constraints(
            model=self.model,
            sistema=self.sistema,
            HORA=1,
            G=self.G,
            B=self.B,
            V=self.V_dict,                
            ANG=self.ANG_dict,
            PGER=self.PGER_dict,
            QGER=self.QGER_dict,
            PLOAD=self.PLOAD.reshape(1, -1),
            QLOAD=self.QLOAD.reshape(1, -1),
            DEFICIT=self.DEFICIT_dict,
            PGWIND=self.PGWIND_dict if self.NGWD > 0 else None,
            BESS_SOC_op=self.BatteryOperation_dict if self.NBESS > 0 else None,
            conv_gen_to_bar=self.thermal_bus.tolist(),
            wind_gen_to_bar=wind_gen_to_bar,
            battery_list=battery_list
        )

    def _add_FOB(self):
        from FOB import AC_PerdasMinimas      
        FOB = AC_PerdasMinimas.AC_PerdasMinimas(_self=self)
        self.model.FOB = pyo.Objective(expr=FOB, sense=pyo.minimize)

    def _build_AC_OPF(self,
                      fator_carga: Optional[Union[float, np.ndarray]] = None,
                      fator_vento: Optional[Union[float, np.ndarray]] = None,
                      soc_baterias: Optional[Dict[int, float]] = None) -> None:
        
        self.model = pyo.ConcreteModel(name="ACOPF_Snapshot")

        self._processa_multiplicadores(fator_carga, fator_vento, soc_baterias)

        self._add_VARS()        
        self._add_CONS()
        self._add_FOB()

        self._solved = False

    def _build_Cenario(self, fator_carga=None, fator_vento=None, soc_baterias=None):
        self._build_AC_OPF(fator_carga, fator_vento, soc_baterias)

    def solve(self, solver_name='ipopt', write_lp=False, **solver_args):
        if self.model is None:
            raise RuntimeError("Modelo não construído.")
        
        opt = pyo.SolverFactory(solver_name)
        try:
            opt.set_executable('/home/lucasedbraga/anaconda3/envs/otm_venv/bin/ipopt')
        except:
            pass
        results = opt.solve(self.model, tee=True)

        self._solved = (results.solver.status == pyo.SolverStatus.ok and
                        results.solver.termination_condition == pyo.TerminationCondition.optimal)
        return results
    
    def solve_snapshot(self, solver_name='ipopt', fator_carga=None, fator_vento=None,
                       soc_baterias=None, cost_function=None, hora=0, dia=0,
                       cen_id=None, write_lp=False, verify=False):
        
        self._build_Cenario(fator_carga, fator_vento, soc_baterias)
        self.model.pprint()
        results = self.solve(solver_name, write_lp=write_lp)

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
            self.print_verification_report()
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
            QLOAD_vals = self.QLOAD.tolist()
            PGER_vals = [pyo.value(self.var_lists['PGER_UTE'][g]) for g in range(self.NUTE)]
            QGER_vals = [pyo.value(self.var_lists['QGER_UTE'][g]) for g in range(self.NUTE)]

            if self.NGWD > 0:
                PGWIND_disponivel = self.PGWIND_AVAIL.tolist()
                PGWIND_vals = [pyo.value(self.var_lists['p_wind'][w]) for w in range(self.NGWD)]
                CURTAILMENT_vals = [pyo.value(self.var_lists['curtailment'][w]) for w in range(self.NGWD)]
            else:
                PGWIND_disponivel = PGWIND_vals = CURTAILMENT_vals = []

            DEFICIT_vals = [pyo.value(self.var_lists['deficit'][b]) for b in range(self.NBAR)]

            SOC_init = [0.0] * self.NBAR
            SOC_atual = [0.0] * self.NBAR
            BESS_operation = [0.0] * self.NBAR
            if self.NBESS > 0:
                for i, bus in enumerate(self.battery_buses):
                    SOC_init[bus] = self.soc_inicial_list[i]
                    SOC_atual[bus] = pyo.value(self.SOC_dict[(0, bus)])
                    charge = pyo.value(self.CHARGE_dict[(0, bus)])
                    discharge = pyo.value(self.DISCHARGE_dict[(0, bus)])
                    BESS_operation[bus] = discharge - charge

            V = [pyo.value(self.var_lists['v_pu'][b]) for b in range(self.NBAR)]
            ANG = [pyo.value(self.var_lists['ang_pu'][b]) for b in range(self.NBAR)]

            P_flow = np.zeros(self.NLIN)
            Q_flow = np.zeros(self.NLIN)
            
            for e in range(self.NLIN):
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

            PERDAS_TOTAIS =  sum(PGER_vals) - sum(PLOAD_vals) - sum(DEFICIT_vals) \
                            + sum(PGWIND_vals) - sum(CURTAILMENT_vals) \
                            + sum(BESS_operation)
                            
            
            custo_deficit_pu = getattr(s, 'Custo_DEFICT', 1000.0)
            CUSTO = [d * custo_deficit_pu for d in DEFICIT_vals]

            CMO = [0.0]

            return OPF_SnapshotResult(
                dia=dia, dia_semana=dia_semana, hora=hora, sucesso=True,
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
                V=V, ANG=ANG,
                FLUXO_LIN=P_flow.tolist(),
                REATIVO_LIN=Q_flow.tolist(),
                CUSTO=CUSTO, CMO=CMO,
                PERDAS_TOTAIS=PERDAS_TOTAIS,
                dia_semana_nome='snapshot'
            )

        except Exception as e:
            print(f"Erro ao extrair snapshot: {e}")
            traceback.print_exc()
            return OPF_SnapshotResult(
                dia=dia, hora=hora, sucesso=False, mensagem=str(e),
                dia_semana=dia_semana, dia_semana_nome=dia_semana_nome
            )


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
    print("SIMULAÇÃO SNAPSHOT - AC OPF")
    print("=" * 70)

    print("\n1. Carregando dados do sistema...")
    json_path = "DATA/input/ieee14_BESS.json"
    if not os.path.exists(json_path):
        print(f"ERRO: Arquivo não encontrado: {json_path}")
        sys.exit(1)

    sistema = SistemaLoader(json_path)
    print(f"   ✓ Sistema carregado: {json_path}")
    print(f"   ✓ Potência base: {sistema.SB:.1f} MVA")
    print(f"   ✓ Barras: {sistema.NBAR}")
    print(f"   ✓ Geradores convencionais: {sistema.NGER_UTE}")
    print(f"   ✓ Geradores eólicos: {sistema.NGER_GWD}")
    print(f"   ✓ Baterias: {len(getattr(sistema, 'BARRAS_COM_BATERIA', []))}")

    print("\n2. Configurando banco de dados...")
    db_handler = OPF_DBHandler('DATA/output/resultados_ACOPF_snapshot.db')
    db_handler.create_tables()
    cen_id = datetime.now().strftime('%Y%m%d%H%M%S') + "_ACOPF_snapshot"
    print(f"   ✓ Cenário ID: {cen_id}")

    modelo = ACOPF_Snapshot(sistema=sistema, db_handler=db_handler)

    hora_desejada = 0

    seed = secrets.randbits(32)    
    avaliador = EvaluateFactors(sistema=sistema, n_dias=1, n_horas=1,
                                carga_incerteza=0.2, vento_variacao=0.1, seed=seed)
    
    fatores_carga_completo, fatores_vento_completo = avaliador.gerar_tudo()

    fator_carga_hora = fatores_carga_completo[0, hora_desejada, :]
    fator_vento_hora = fatores_vento_completo[0, hora_desejada, :] if sistema.NGER_GWD > 0 else 1.0

    print(f"\n3. Parâmetros para Hora {hora_desejada}:")
    print(f"   Fator de carga médio: {np.mean(fator_carga_hora):.3f}")
    if sistema.NGER_GWD > 0:
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
        print(f"   Demanda Ativa :  {sum(resultado.PLOAD):.3f} pu")
        print(f"   Geração Ativa (UTE): {sum(resultado.PGER):.3f} pu")
        print(f"   Demanda Reativa :  {sum(resultado.QLOAD):.3f} pu")
        print(f"   Geração Reativa (UTE): {sum(resultado.QGER):.3f} pu")
        if sistema.NGER_GWD > 0:
            print(f"   Geração WIND total:   {sum(resultado.PGWIND):.3f} pu")
            print(f"   WIND Curtailment total:      {sum(resultado.CURTAILMENT):.3f} pu")
        print(f"   Déficit total:          {sum(resultado.DEFICIT):.3f} pu")
        print(f"   Perdas ativas totais:   {resultado.PERDAS_TOTAIS:.3f} pu")
        print(f"   Tensão mínima: {min(resultado.V):.3f} pu, máxima: {max(resultado.V):.3f} pu")
        print(f"   Ângulo slack (ref): {resultado.ANG[sistema.slack_idx]:.3f} rad")
        if sistema.BARRAS_COM_BATERIA:
            for b in sistema.BARRAS_COM_BATERIA:
                print(f"   Bateria barra {b}:")
                print(f"      operação = {resultado.BESS_operation[b]:.3f} pu")
                print(f"      SOC inicial = {resultado.SOC_init[b]:.3f} pu")
                print(f"      SOC final   = {resultado.SOC_atual[b]:.3f} pu")
        #print(f"   CMO (barra slack): {resultado.CMO[0]:.2f} $/MWh")

    print("\n" + "=" * 70)
    print("EXECUÇÃO CONCLUÍDA")
    print("=" * 70)