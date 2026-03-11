import os
import sys
import numpy as np
import pyoptinterface as poi
from pyoptinterface import highs
from typing import List, Union, Optional, Tuple, Dict
import traceback

# Ajusta o path para encontrar os módulos do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports corrigidos
from SOLVER.OPF_DC.RES.BatteryConstraints import BatteryConstraints
from SOLVER.OPF_DC.RES.ThermalGeneratorConstraints import ThermalGeneratorConstraints
from SOLVER.OPF_DC.RES.WindGeneratorConstraints import WindGeneratorConstraints
from SOLVER.OPF_DC.RES.EletricConstraints import ElectricConstraints
from DB.DBmodel_OPF import TimeCoupledOPFResult, TimeCoupledOPFSnapshotResult


class DCOPFSnapshot:
    """
    Modelo de otimização para UMA ÚNICA HORA (snapshot).
    Baseado no TimeCoupledOPFModel, mas sem acoplamento temporal.
    As baterias são tratadas com SOC fixo (não há evolução temporal).
    """

    def __init__(self,
                 sistema,
                 db_handler=None,
                 considerar_perdas: bool = True):
        """
        Inicializa o modelo para uma única hora.
        
        Args:
            sistema: Objeto SistemaLoader com dados do sistema
            db_handler: Handler para banco de dados (opcional)
            considerar_perdas: Se True, considera perdas nas linhas
        """
        self.sistema = sistema
        self.db_handler = db_handler
        self.considerar_perdas = considerar_perdas

        # Modelo PyOptInterface
        self.model = None
        self._solved = False
        self._battery_list: List[int] = []
        self._battery_index: Dict[int, int] = {}
        self._perdas_calculadas: Optional[np.ndarray] = None
        self._soc_fixo_list: List[float] = []  # SOC fixo para cada bateria

        # Dicionários para armazenar variáveis (sem índice temporal)
        self.PGER: Dict[int, poi.Variable] = {}           # índice: gerador
        self.PGWIND: Dict[int, poi.Variable] = {}         # índice: gerador eólico
        self.CURTAILMENT: Dict[int, poi.Variable] = {}    # índice: gerador eólico
        self.DEFICIT: Dict[int, poi.Variable] = {}        # índice: barra
        self.V: Dict[int, poi.Variable] = {}              # índice: barra
        self.ANG: Dict[int, poi.Variable] = {}            # índice: barra
        self.FLUXO_LIN: Dict[int, poi.Variable] = {}      # índice: linha
        self.CHARGE: Dict[int, poi.Variable] = {}         # índice: barra com bateria
        self.DISCHARGE: Dict[int, poi.Variable] = {}      # índice: barra com bateria
        self.BatteryOperation: Dict[int, poi.Variable] = {}  # índice: barra com bateria

        # Parâmetros
        self.PLOAD: Optional[np.ndarray] = None           # demanda por barra
        self.PGWIND_AVAIL: Optional[np.ndarray] = None    # disponibilidade eólica por gerador

        # Lista de restrições de balanço (para remoção/recriação)
        self.balance_constraints: List[poi.Constraint] = []

    # -------------------------------------------------------------------------
    # Construção do modelo
    # -------------------------------------------------------------------------
    def build(self,
              fator_carga: Optional[Union[float, np.ndarray]] = None,
              fator_vento: Optional[Union[float, np.ndarray]] = None,
              soc_baterias: Optional[Dict[int, float]] = None) -> None:
        """
        Constrói o modelo para uma única hora.
        
        Args:
            fator_carga: Fator de carga (escalar ou array por barra)
            fator_vento: Fator de vento (escalar ou array por gerador eólico)
            soc_baterias: Dicionário com SOC fixo para cada barra com bateria
                         Ex: {3: 0.5, 5: 0.3} (50% e 30% da capacidade)
        """
        s = self.sistema
        T = 1  # Apenas um período

        # ==================== Processamento dos fatores ====================
        # Fator de carga
        if fator_carga is not None:
            if np.isscalar(fator_carga):
                fator_carga_array = np.ones(s.NBAR) * fator_carga
            elif isinstance(fator_carga, np.ndarray) and fator_carga.shape == (s.NBAR,):
                fator_carga_array = fator_carga
            elif isinstance(fator_carga, np.ndarray) and fator_carga.size == 1:
                fator_carga_array = np.ones(s.NBAR) * fator_carga.item()
            else:
                raise ValueError(f"fator_carga deve ser escalar ou array de shape ({s.NBAR},)")
        else:
            fator_carga_array = np.ones(s.NBAR)

        # Fator de vento
        if fator_vento is not None:
            if s.NGER_EOL == 0:
                fator_vento_array = np.array([])
            elif np.isscalar(fator_vento):
                fator_vento_array = np.ones(s.NGER_EOL) * fator_vento
            elif isinstance(fator_vento, np.ndarray) and fator_vento.shape == (s.NGER_EOL,):
                fator_vento_array = fator_vento
            elif isinstance(fator_vento, np.ndarray) and fator_vento.size == 1:
                fator_vento_array = np.ones(s.NGER_EOL) * fator_vento.item()
            else:
                raise ValueError(f"fator_vento deve ser escalar ou array de shape ({s.NGER_EOL},)")
        else:
            fator_vento_array = np.ones(s.NGER_EOL) if s.NGER_EOL > 0 else np.array([])

        # ==================== Inicializar modelo ====================
        self.model = highs.Model()

        # ==================== Parâmetros ====================
        # Demanda por barra
        self.PLOAD = np.zeros(s.NBAR)
        for b in range(s.NBAR):
            self.PLOAD[b] = s.PLOAD[b] * fator_carga_array[b]

        # Disponibilidade eólica
        if s.NGER_EOL > 0:
            self.PGWIND_AVAIL = np.zeros(s.NGER_EOL)
            for w in range(s.NGER_EOL):
                self.PGWIND_AVAIL[w] = s.PGWIND_disponivel[w] * fator_vento_array[w]
        else:
            self.PGWIND_AVAIL = np.array([])

        # ==================== SOC fixo das baterias ====================
        self._battery_list = list(s.BARRAS_COM_BATERIA)
        self._battery_index = {b: i for i, b in enumerate(self._battery_list)}

        # Garantir arrays (valores em MW/MWh)
        if not hasattr(s, 'BATTERY_CAPACITY'):
            s.BATTERY_CAPACITY = np.zeros(s.NBAR)
        if not hasattr(s, 'BATTERY_MIN_SOC'):
            s.BATTERY_MIN_SOC = np.zeros(s.NBAR)
        if not hasattr(s, 'BATTERY_POWER_LIMIT'):
            s.BATTERY_POWER_LIMIT = np.zeros(s.NBAR)
        if not hasattr(s, 'BATTERY_CHARGE_EFF'):
            s.BATTERY_CHARGE_EFF = 0.9
        if not hasattr(s, 'BATTERY_DISCHARGE_EFF'):
            s.BATTERY_DISCHARGE_EFF = 0.9

        # Processar SOC das baterias
        self._soc_fixo_list = []
        for i, b in enumerate(self._battery_list):
            if soc_baterias is not None and b in soc_baterias:
                soc_frac = soc_baterias[b]
            else:
                soc_frac = 0.5  # valor padrão: 50%
            self._soc_fixo_list.append(soc_frac * s.BATTERY_CAPACITY[b])

        # ==================== Variáveis ====================
        # Geração térmica
        for g in range(s.NGER_CONV):
            self.PGER[g] = self.model.add_variable(
                lb=s.PGMIN_CONV[g] * s.SB,
                ub=s.PGMAX_CONV[g] * s.SB,
                name=f"PGER_{g}"
            )

        # Geração eólica e curtailment
        if s.NGER_EOL > 0:
            for w in range(s.NGER_EOL):
                self.PGWIND[w] = self.model.add_variable(
                    lb=0,
                    ub=self.PGWIND_AVAIL[w],
                    name=f"PGWIND_{w}"
                )
                self.CURTAILMENT[w] = self.model.add_variable(
                    lb=0,
                    ub=self.PGWIND_AVAIL[w],
                    name=f"CURTAILMENT_{w}"
                )

        # Déficit
        for b in range(s.NBAR):
            self.DEFICIT[b] = self.model.add_variable(
                lb=0,
                ub=1e6,
                name=f"DEFICIT_{b}"
            )

        # Tensão, ângulo, fluxo
        for b in range(s.NBAR):
            self.V[b] = self.model.add_variable(
                lb=0.95,
                ub=1.05,
                name=f"V_{b}"
            )
            self.model.add_linear_constraint(self.V[b] == 1.0, name=f"fix_V_{b}")

        for b in range(s.NBAR):
            self.ANG[b] = self.model.add_variable(
                lb=-np.pi,
                ub=np.pi,
                name=f"ANG_{b}"
            )
            if b == s.slack_idx:
                self.model.add_linear_constraint(self.ANG[b] == 0.0, name=f"fix_ANG_slack")

        for e in range(s.NLIN):
            self.FLUXO_LIN[e] = self.model.add_variable(
                lb=-s.FLIM[e],
                ub=s.FLIM[e],
                name=f"FLUXO_LIN_{e}"
            )

        # ==================== Baterias ====================
        # Para snapshot, a bateria opera com SOC fixo
        # Criamos variáveis de operação sem restrições de evolução temporal
        for b in self._battery_list:
            power_limit = s.BATTERY_POWER_LIMIT[b] * s.SB
            
            # Potência de carga (positiva)
            self.CHARGE[b] = self.model.add_variable(
                lb=0,
                ub=power_limit,
                name=f"CHARGE_{b}"
            )
            
            # Potência de descarga (positiva)
            self.DISCHARGE[b] = self.model.add_variable(
                lb=0,
                ub=power_limit,
                name=f"DISCHARGE_{b}"
            )
            
            # Operação líquida (positivo = descarga, negativo = carga)
            self.BatteryOperation[b] = self.model.add_variable(
                lb=-power_limit,
                ub=power_limit,
                name=f"BatteryOperation_{b}"
            )
            
            # Relação entre as variáveis: BatteryOperation = DISCHARGE - CHARGE
            self.model.add_linear_constraint(
                self.BatteryOperation[b] - self.DISCHARGE[b] + self.CHARGE[b] == 0,
                name=f"battery_op_link_{b}"
            )
            
            # Restrição de SOC mínimo (não pode descarregar abaixo do mínimo)
            # Para snapshot, usamos o SOC fixo para limitar a descarga máxima
            soc_atual_mwh = self._soc_fixo_list[self._battery_index[b]]
            soc_min_mwh = s.BATTERY_MIN_SOC[b] * s.BATTERY_CAPACITY[b]
            
            # Energia disponível para descarga (respeitando SOC mínimo)
            energia_disponivel = max(0, soc_atual_mwh - soc_min_mwh)
            
            # A descarga em 1 hora não pode exceder a energia disponível
            self.model.add_linear_constraint(
                self.DISCHARGE[b] <= energia_disponivel,
                name=f"battery_max_discharge_{b}"
            )
            
            # Capacidade disponível para carga (espaço na bateria)
            capacidade_livre = s.BATTERY_CAPACITY[b] - soc_atual_mwh
            
            # A carga em 1 hora não pode exceder a capacidade livre
            self.model.add_linear_constraint(
                self.CHARGE[b] <= capacidade_livre,
                name=f"battery_max_charge_{b}"
            )

        # ==================== Restrições ====================
        self._add_all_constraints()
        self.add_FOB()

        self._solved = False

    def _add_all_constraints(self) -> None:
        """Adiciona todas as restrições delegando para os módulos específicos."""
        s = self.sistema
        T = 1  # Apenas um período

        # Geradores térmicos (versão adaptada para 1 período)
        self._add_thermal_constraints()

        # Geradores eólicos
        if s.NGER_EOL > 0:
            self._add_wind_constraints()

        # Mapeamento da barra de cada gerador eólico
        if hasattr(s, 'bus_wind'):
            wind_gen_to_bar = s.bus_wind
        elif hasattr(s, 'BARPG_EOL'):
            wind_gen_to_bar = s.BARPG_EOL
        else:
            wind_gen_to_bar = [0] * s.NGER_EOL

        # Restrições elétricas (adaptadas para 1 período)
        self.balance_constraints = self._add_electric_constraints(wind_gen_to_bar)

    def _add_thermal_constraints(self) -> None:
        """Adiciona restrições dos geradores térmicos para 1 período."""
        s = self.sistema
        
        # Para snapshot, apenas limites de geração (já estão nos bounds das variáveis)
        # Não há restrições de rampa
        pass

    def _add_wind_constraints(self) -> None:
        """Adiciona restrições dos geradores eólicos para 1 período."""
        s = self.sistema
        
        # Relação entre geração e curtailment
        for w in range(s.NGER_EOL):
            self.model.add_linear_constraint(
                self.PGWIND[w] + self.CURTAILMENT[w] == self.PGWIND_AVAIL[w],
                name=f"wind_curtail_link_{w}"
            )

    def _add_electric_constraints(self, wind_gen_to_bar) -> None:
        """
        Adiciona restrições elétricas (balanço de potência) para 1 período.
        Retorna lista de constraints de balanço.
        """
        s = self.sistema
        balance_constrs = []

        # Para cada barra, balanço de potência
        for b in range(s.NBAR):
            # Coeficientes e variáveis para o balanço
            expr = 0.0
            
            # Geração convencional nesta barra
            for g, barra_gen in enumerate(s.BARPG_CONV):
                if barra_gen == b:
                    expr += self.PGER[g]
            
            # Geração eólica nesta barra
            if s.NGER_EOL > 0:
                for w, barra_wind in enumerate(wind_gen_to_bar):
                    if barra_wind == b:
                        expr += self.PGWIND[w]
            
            # Baterias nesta barra (descarga - carga)
            if b in self._battery_list:
                expr += self.DISCHARGE[b] - self.CHARGE[b]
            
            # Déficit (entra como geração)
            expr += self.DEFICIT[b]
            
            # Perdas (se consideradas)
            if self.considerar_perdas and self._perdas_calculadas is not None:
                expr -= self._perdas_calculadas[b]
            
            # Fluxo nas linhas que chegam/saem
            for e in range(s.NLIN):
                if s.line_fr[e] == b:
                    expr -= self.FLUXO_LIN[e]
                if s.line_to[e] == b:
                    expr += self.FLUXO_LIN[e]
            
            # Demanda
            expr -= self.PLOAD[b]
            
            # Criar constraint
            constr = self.model.add_linear_constraint(expr == 0, name=f"balance_{b}")
            balance_constrs.append(constr)
        
        # Relação entre fluxo e ângulos
        for e in range(s.NLIN):
            i = s.line_fr[e]
            j = s.line_to[e]
            x = sistema.x_line[e]
            
            self.model.add_linear_constraint(
                self.FLUXO_LIN[e] == (self.ANG[i] - self.ANG[j])/x,
                name=f"power_flow_{e}"
            )
        
        return balance_constrs

    def add_FOB(self) -> None:
        """Função objetivo: minimizar custo de geração térmica + penalidade de déficit."""
        s = self.sistema
        
        # Custo da geração térmica (assumindo custo linear)
        custo_termico = 0
        for g in range(s.NGER_CONV):
            # Usar custo médio ou custo definido
            custo = getattr(s, 'CUSTO_GER', [50.0] * s.NGER_CONV)[g]
            custo_termico += custo * self.PGER[g]
        
        # Penalidade do déficit
        penalidade_deficit = 0
        for b in range(s.NBAR):
            custo_deficit = getattr(s, 'CPG_DEFICIT', 1000.0)
            penalidade_deficit += custo_deficit * self.DEFICIT[b]
        
        self.model.set_objective(custo_termico + penalidade_deficit)

    # -------------------------------------------------------------------------
    # Métodos para perdas iterativas
    # -------------------------------------------------------------------------
    def calculate_losses(self) -> np.ndarray:
        """Calcula perdas nas linhas baseado na solução atual."""
        s = self.sistema
        perdas_barra = np.zeros(s.NBAR)
        
        for e in range(s.NLIN):
            i = s.line_fr[e]
            j = s.line_to[e]
            fluxo_val = self.model.get_value(self.FLUXO_LIN[e])
            r = s.r_line[e]
            perdas_linha = r * (fluxo_val ** 2) * s.SB
            perdas_barra[i] += perdas_linha / 2
            perdas_barra[j] += perdas_linha / 2
        
        self._perdas_calculadas = perdas_barra
        return perdas_barra

    def update_losses(self, perdas_barra: np.ndarray) -> None:
        """Atualiza o vetor de perdas."""
        self._perdas_calculadas = perdas_barra

    def solve_iterative(self, solver_name: str = 'highs', tol: float = 1e-4,
                        max_iter: int = 50, write_lp: bool = True, **solver_args):
        """
        Resolve o modelo com iterações de perdas (ponto fixo).
        """
        if not self.considerar_perdas:
            return self.solve(solver_name, write_lp=write_lp, **solver_args)

        # Primeira solução
        raw = self.solve(solver_name, write_lp=write_lp, **solver_args)
        if not self._solved or self.model.get_model_attribute(
                poi.ModelAttribute.TerminationStatus) != poi.TerminationStatusCode.OPTIMAL:
            print("Primeira iteração: solução não ótima.")
            return raw

        ang_prev = np.array([self.model.get_value(self.ANG[b]) for b in range(self.sistema.NBAR)])

        for it in range(1, max_iter):
            # Calcular perdas com a solução atual
            perdas = self.calculate_losses()
            self.update_losses(perdas)

            # Remover restrições de balanço antigas
            for constr in self.balance_constraints:
                self.model.delete_constraint(constr)

            # Recriar restrições de balanço com as novas perdas
            s = self.sistema
            if hasattr(s, 'bus_wind'):
                wind_gen_to_bar = s.bus_wind
            elif hasattr(s, 'BARPG_EOL'):
                wind_gen_to_bar = s.BARPG_EOL
            else:
                wind_gen_to_bar = [0] * s.NGER_EOL
            
            self.balance_constraints = self._add_electric_constraints(wind_gen_to_bar)

            # Resolver novamente
            raw = self.solve(solver_name, write_lp=False, **solver_args)
            if not self._solved or self.model.get_model_attribute(
                    poi.ModelAttribute.TerminationStatus) != poi.TerminationStatusCode.OPTIMAL:
                print(f"Iteração {it+1}: solução não ótima.")
                break

            ang_curr = np.array([self.model.get_value(self.ANG[b]) for b in range(self.sistema.NBAR)])
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
        """
        if self.model is None:
            raise RuntimeError("Modelo não construído. Chame build() primeiro.")
        
        if write_lp:
            import inspect
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            caller_filename = caller_frame.f_code.co_filename
            base = os.path.splitext(os.path.basename(caller_filename))[0]
            lp_filename = f"DATA/output/{base}_snapshot.lp"
            self.model.write(lp_filename)
            print(f"Modelo escrito em {lp_filename}")
        
        self.model.optimize()
        self._solved = True
        return self.model

    # -------------------------------------------------------------------------
    # Extração de resultados
    # -------------------------------------------------------------------------
    def extract_results(self, hora: int = 0, dia: int = 0, 
                       cen_id: Optional[str] = None) -> TimeCoupledOPFSnapshotResult:
        """
        Extrai os resultados da solução para a hora atual.
        
        Args:
            hora: Hora do dia (para referência)
            dia: Dia da simulação (para referência)
            cen_id: ID do cenário (opcional)
        
        Returns:
            TimeCoupledOPFSnapshotResult com os resultados
        """
        if not self._solved or self.model is None:
            raise RuntimeError("Modelo não resolvido. Execute solve() primeiro.")

        s = self.sistema
        dias_nomes = ["domingo", "segunda", "terça", "quarta", "quinta", "sexta", "sábado"]
        dia_semana = ((dia) % 7) + 1
        dia_semana_nome = dias_nomes[dia_semana-1]

        try:
            # Demanda
            PLOAD_vals = self.PLOAD.tolist()
            demanda_total = sum(PLOAD_vals)

            # Geração térmica
            PGER_vals = [self.model.get_value(self.PGER[g]) for g in range(s.NGER_CONV)]
            ger_term_total = sum(PGER_vals)

            # Eólica
            if s.NGER_EOL > 0:
                PGWIND_disponivel = self.PGWIND_AVAIL.tolist()
                PGWIND_vals = [self.model.get_value(self.PGWIND[w]) for w in range(s.NGER_EOL)]
                CURTAILMENT_vals = [self.model.get_value(self.CURTAILMENT[w]) for w in range(s.NGER_EOL)]
                ger_eol_total = sum(PGWIND_vals)
            else:
                PGWIND_disponivel = PGWIND_vals = CURTAILMENT_vals = []
                ger_eol_total = 0.0

            # Déficit
            DEFICIT_vals = [self.model.get_value(self.DEFICIT[b]) for b in range(s.NBAR)]
            deficit_total = sum(DEFICIT_vals)

            # Baterias
            SOC_init = [0.0] * s.NBAR
            SOC_atual = [0.0] * s.NBAR
            BESS_operation = [0.0] * s.NBAR
            carga_total = 0.0
            descarga_total = 0.0

            if self._battery_list:
                for i, b in enumerate(self._battery_list):
                    # SOC inicial (fixo informado)
                    SOC_init[b] = self._soc_fixo_list[i]
                    
                    # Para snapshot, SOC atual = SOC inicial (não há evolução)
                    SOC_atual[b] = SOC_init[b]
                    
                    # Operação
                    charge = self.model.get_value(self.CHARGE[b]) if b in self.CHARGE else 0.0
                    discharge = self.model.get_value(self.DISCHARGE[b]) if b in self.DISCHARGE else 0.0
                    BESS_operation[b] = discharge - charge
                    carga_total += charge
                    descarga_total += discharge

            # Tensão, ângulo, fluxo
            V = [self.model.get_value(self.V[b]) for b in range(s.NBAR)]
            ANG = [self.model.get_value(self.ANG[b]) for b in range(s.NBAR)]
            FLUXO_LIN = [self.model.get_value(self.FLUXO_LIN[e]) for e in range(s.NLIN)]

            # Custos (déficit)
            CUSTO = [DEFICIT_vals[b] * getattr(s, 'CPG_DEFICIT', 1000.0) for b in range(s.NBAR)]

            # CMO (preço marginal) – obtido da restrição da barra slack
            CMO = 0.0
            # for b, constr in enumerate(self.balance_constraints):
            #     if b == s.slack_idx:
            #         CMO = self.model.get_dual(constr)
            #         break

            # Perdas
            if self.considerar_perdas and self._perdas_calculadas is not None:
                PERDAS_BARRA = self._perdas_calculadas.tolist()
                perdas_total = sum(PERDAS_BARRA)
            else:
                PERDAS_BARRA = [0.0] * s.NBAR
                perdas_total = 0.0

            # Criar objeto de resultado
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
                CMO=[CMO],
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

    # -------------------------------------------------------------------------
    # Verificação de restrições e balanço de massa
    # -------------------------------------------------------------------------
    def verify_constraints(self, tol: float = 1e-4) -> Dict[str, List[str]]:
        """
        Verifica se todas as restrições do modelo estão sendo atendidas pela solução atual.
        
        Args:
            tol: Tolerância para comparações
            
        Returns:
            Dicionário com listas de violações por categoria
        """
        if not self._solved or self.model is None:
            raise RuntimeError("Modelo não resolvido. Execute solve() primeiro.")
        
        s = self.sistema
        violacoes = {
            'balanco': [],
            'termica': [],
            'eolica': [],
            'bateria': [],
            'fluxo': []
        }
        
        # ==================== 1. BALANÇO DE POTÊNCIA POR BARRA ====================
        for b in range(s.NBAR):
            # Calcular LHS (geração)
            lhs = 0.0
            
            # Geração térmica nesta barra
            for g, barra_gen in enumerate(s.BARPG_CONV):
                if barra_gen == b:
                    lhs += self.model.get_value(self.PGER[g])
            
            # Geração eólica nesta barra
            if s.NGER_EOL > 0:
                wind_gen_to_bar = getattr(s, 'bus_wind', getattr(s, 'BARPG_EOL', [0] * s.NGER_EOL))
                for w, barra_wind in enumerate(wind_gen_to_bar):
                    if barra_wind == b:
                        lhs += self.model.get_value(self.PGWIND[w])
            
            # Baterias nesta barra (descarga - carga)
            if b in self._battery_list:
                lhs += self.model.get_value(self.DISCHARGE[b]) - self.model.get_value(self.CHARGE[b])
            
            # Déficit (entra como geração)
            lhs += self.model.get_value(self.DEFICIT[b])
            
            # Fluxo nas linhas (positivo saindo, negativo entrando)
            for e in range(s.NLIN):
                fluxo = self.model.get_value(self.FLUXO_LIN[e])
                if s.line_fr[e] == b:
                    lhs -= fluxo  # potência saindo da barra
                if s.line_to[e] == b:
                    lhs += fluxo  # potência entrando na barra
            
            # RHS (demanda)
            rhs = self.PLOAD[b]
            
            # Perdas (se consideradas)
            if self.considerar_perdas and self._perdas_calculadas is not None:
                rhs += self._perdas_calculadas[b]
            
            # Verificar balanço
            diff = lhs - rhs
            if abs(diff) > tol:
                violacoes['balanco'].append(
                    f"Barra {b}: balanço não fecha. LHS={lhs:.6f}, RHS={rhs:.6f}, diff={diff:.6f}"
                )
        
        # ==================== 2. LIMITES DOS GERADORES TÉRMICOS ====================
        for g in range(s.NGER_CONV):
            pger = self.model.get_value(self.PGER[g])
            pmin = s.PGMIN_CONV[g] * s.SB
            pmax = s.PGMAX_CONV[g] * s.SB
            
            if pger < pmin - tol:
                violacoes['termica'].append(
                    f"Gerador {g}: PGER={pger:.4f} < mínimo {pmin:.4f}"
                )
            if pger > pmax + tol:
                violacoes['termica'].append(
                    f"Gerador {g}: PGER={pger:.4f} > máximo {pmax:.4f}"
                )
        
        # ==================== 3. LIMITES DOS GERADORES EÓLICOS ====================
        if s.NGER_EOL > 0:
            wind_gen_to_bar = getattr(s, 'bus_wind', getattr(s, 'BARPG_EOL', [0] * s.NGER_EOL))
            for w in range(s.NGER_EOL):
                pgwind = self.model.get_value(self.PGWIND[w])
                curtail = self.model.get_value(self.CURTAILMENT[w])
                disponivel = self.PGWIND_AVAIL[w]
                
                # Verificar se geração + curtailment = disponível
                if abs(pgwind + curtail - disponivel) > tol:
                    violacoes['eolica'].append(
                        f"Eólica {w}: PGWIND({pgwind:.4f}) + CURTAIL({curtail:.4f}) != disponível({disponivel:.4f})"
                    )
                
                # Verificar limites individuais
                if pgwind < -tol:
                    violacoes['eolica'].append(f"Eólica {w}: PGWIND={pgwind:.4f} < 0")
                if pgwind > disponivel + tol:
                    violacoes['eolica'].append(f"Eólica {w}: PGWIND={pgwind:.4f} > disponível {disponivel:.4f}")
                if curtail < -tol:
                    violacoes['eolica'].append(f"Eólica {w}: CURTAILMENT={curtail:.4f} < 0")
                if curtail > disponivel + tol:
                    violacoes['eolica'].append(f"Eólica {w}: CURTAILMENT={curtail:.4f} > disponível {disponivel:.4f}")
        
        # ==================== 4. LIMITES DAS BATERIAS ====================
        if self._battery_list:
            for i, b in enumerate(self._battery_list):
                charge = self.model.get_value(self.CHARGE[b]) if b in self.CHARGE else 0.0
                discharge = self.model.get_value(self.DISCHARGE[b]) if b in self.DISCHARGE else 0.0
                operation = self.model.get_value(self.BatteryOperation[b]) if b in self.BatteryOperation else 0.0
                
                power_limit = s.BATTERY_POWER_LIMIT[b] * s.SB
                soc_atual_mwh = self._soc_fixo_list[i]
                soc_min_mwh = s.BATTERY_MIN_SOC[b] * s.BATTERY_CAPACITY[b]
                
                # Verificar relação operation = discharge - charge
                if abs(operation - (discharge - charge)) > tol:
                    violacoes['bateria'].append(
                        f"Bateria {b}: operation({operation:.4f}) != discharge({discharge:.4f}) - charge({charge:.4f})"
                    )
                
                # Verificar limites de potência
                if charge < -tol:
                    violacoes['bateria'].append(f"Bateria {b}: CHARGE={charge:.4f} < 0")
                if charge > power_limit + tol:
                    violacoes['bateria'].append(f"Bateria {b}: CHARGE={charge:.4f} > limite {power_limit:.4f}")
                if discharge < -tol:
                    violacoes['bateria'].append(f"Bateria {b}: DISCHARGE={discharge:.4f} < 0")
                if discharge > power_limit + tol:
                    violacoes['bateria'].append(f"Bateria {b}: DISCHARGE={discharge:.4f} > limite {power_limit:.4f}")
                
                # Verificar limites de energia
                energia_disponivel = max(0, soc_atual_mwh - soc_min_mwh)
                if discharge > energia_disponivel + tol:
                    violacoes['bateria'].append(
                        f"Bateria {b}: DISCHARGE={discharge:.4f} > energia disponível {energia_disponivel:.4f} "
                        f"(SOC={soc_atual_mwh:.4f}, min={soc_min_mwh:.4f})"
                    )
                
                capacidade_livre = s.BATTERY_CAPACITY[b] - soc_atual_mwh
                if charge > capacidade_livre + tol:
                    violacoes['bateria'].append(
                        f"Bateria {b}: CHARGE={charge:.4f} > capacidade livre {capacidade_livre:.4f}"
                    )
        
        # ==================== 5. LIMITES DE FLUXO NAS LINHAS ====================
        for e in range(s.NLIN):
            fluxo = self.model.get_value(self.FLUXO_LIN[e])
            flim = s.FLIM[e]
            
            if abs(fluxo) > flim + tol:
                violacoes['fluxo'].append(
                    f"Linha {e}: |FLUXO|={abs(fluxo):.4f} > limite {flim:.4f}"
                )
        
        return violacoes

    def print_verification_report(self, tol: float = 1e-4) -> None:
        """
        Imprime um relatório detalhado da verificação de restrições.
        """
        print("\n" + "=" * 70)
        print("RELATÓRIO DE VERIFICAÇÃO DE RESTRIÇÕES")
        print("=" * 70)
        
        try:
            violacoes = self.verify_constraints(tol)
            
            total_violacoes = sum(len(v) for v in violacoes.values())
            
            if total_violacoes == 0:
                print("\n✅ NENHUMA VIOLAÇÃO ENCONTRADA.")
                print("   O snapshot respeita todas as restrições dentro da tolerância.")
            else:
                print(f"\n❌ TOTAL DE VIOLAÇÕES: {total_violacoes}")
                
                # Balanço de potência
                if violacoes['balanco']:
                    print(f"\n--- BALANÇO DE POTÊNCIA ({len(violacoes['balanco'])} violações) ---")
                    for v in violacoes['balanco']:
                        print(f"   {v}")
                
                # Geração térmica
                if violacoes['termica']:
                    print(f"\n--- GERAÇÃO TÉRMICA ({len(violacoes['termica'])} violações) ---")
                    for v in violacoes['termica']:
                        print(f"   {v}")
                
                # Geração eólica
                if violacoes['eolica']:
                    print(f"\n--- GERAÇÃO EÓLICA ({len(violacoes['eolica'])} violações) ---")
                    for v in violacoes['eolica']:
                        print(f"   {v}")
                
                # Baterias
                if violacoes['bateria']:
                    print(f"\n--- BATERIAS ({len(violacoes['bateria'])} violações) ---")
                    for v in violacoes['bateria']:
                        print(f"   {v}")
                
                # Fluxo nas linhas
                if violacoes['fluxo']:
                    print(f"\n--- FLUXO NAS LINHAS ({len(violacoes['fluxo'])} violações) ---")
                    for v in violacoes['fluxo']:
                        print(f"   {v}")
            
            # Resumo do balanço por barra (mesmo sem violações)
            print("\n--- RESUMO DO BALANÇO POR BARRA ---")
            s = self.sistema
            for b in range(s.NBAR):
                # Calcular injeção líquida
                inj_liquida = 0.0
                
                # Geração térmica
                for g, barra_gen in enumerate(s.BARPG_CONV):
                    if barra_gen == b:
                        inj_liquida += self.model.get_value(self.PGER[g])
                
                # Geração eólica
                if s.NGER_EOL > 0:
                    wind_gen_to_bar = getattr(s, 'bus_wind', getattr(s, 'BARPG_EOL', [0] * s.NGER_EOL))
                    for w, barra_wind in enumerate(wind_gen_to_bar):
                        if barra_wind == b:
                            inj_liquida += self.model.get_value(self.PGWIND[w])
                
                # Baterias
                if b in self._battery_list:
                    inj_liquida += self.model.get_value(self.DISCHARGE[b]) - self.model.get_value(self.CHARGE[b])
                
                # Déficit
                inj_liquida += self.model.get_value(self.DEFICIT[b])
                
                # Fluxo nas linhas
                for e in range(s.NLIN):
                    fluxo = self.model.get_value(self.FLUXO_LIN[e])
                    if s.line_fr[e] == b:
                        inj_liquida -= fluxo
                    if s.line_to[e] == b:
                        inj_liquida += fluxo
                
                # Demanda + perdas
                demanda = self.PLOAD[b]
                perdas = self._perdas_calculadas[b] if self._perdas_calculadas is not None else 0.0
                
                diff = inj_liquida - demanda - perdas
                
                status = "✅" if abs(diff) < tol else "❌"
                print(f"   Barra {b}: {status} Injeção={inj_liquida:8.4f} | Demanda={demanda:8.4f} | Perdas={perdas:8.4f} | Diff={diff:8.4e}")
            
            print("=" * 70)
            
        except Exception as e:
            print(f"\n❌ Erro durante verificação: {e}")
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # Método de conveniência
    # -------------------------------------------------------------------------
    def solve_snapshot(self,
                        solver_name: str = 'highs',
                        fator_carga: Optional[Union[float, np.ndarray]] = None,
                        fator_vento: Optional[Union[float, np.ndarray]] = None,
                        soc_baterias: Optional[Dict[int, float]] = None,
                        hora: int = 0,
                        dia: int = 0,
                        cen_id: Optional[str] = None,
                        tol: float = 1e-4,
                        max_iter: int = 50,
                        write_lp: bool = True,
                        verify: bool = True):  # <-- NOVO PARÂMETRO
            """
            Executa todo o processo para um snapshot: construir, resolver e salvar.
            
            Args:
                solver_name: Nome do solver
                fator_carga: Fator de carga
                fator_vento: Fator de vento
                soc_baterias: Dicionário com SOC das baterias
                hora: Hora do dia (para referência)
                dia: Dia (para referência)
                cen_id: ID do cenário
                tol: Tolerância para iterações de perdas
                max_iter: Máximo de iterações de perdas
                write_lp: Se True, escreve arquivo LP
                verify: Se True, verifica restrições após resolver
            
            Returns:
                raw: Modelo resolvido
            """
            self.build(fator_carga, fator_vento, soc_baterias)

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
            
            # Verificar restrições se solicitado
            if verify:
                self.print_verification_report(tol=1e-4)

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
    print("SIMULAÇÃO SNAPSHOT (UMA HORA) - DC OPF")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Carregar sistema
    # -------------------------------------------------------------------------
    print("\n1. Carregando dados do sistema...")
    json_path = "DATA/input/B6L8_BASE.json"  # Ajuste conforme necessário
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
    db_handler = OPF_DBHandler('DATA/output/resultados_snapshot.db')
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
    hora_desejada = 4  # Hora 4 (0-23)
    
    # Fatores de carga e vento (podem vir de EvaluateFactors ou ser definidos manualmente)
    seed = secrets.randbits(32)
    from UTILS.EvaluateFactors import EvaluateFactors
    
    # Gerar fatores para 1 dia (para ter um valor para a hora desejada)
    avaliador = EvaluateFactors(
        sistema=sistema,
        n_dias=1,
        n_horas=24,
        carga_incerteza=0.2,
        vento_variacao=0.1,
        seed=seed
    )
    fatores_carga_completo, fatores_vento_completo = avaliador.gerar_tudo()
    
    # Extrair fatores para a hora desejada
    fator_carga_hora = fatores_carga_completo[0, hora_desejada, :]  # shape: (n_barras,)
    fator_vento_hora = fatores_vento_completo[0, hora_desejada, :] if sistema.NGER_EOL > 0 else 1.0
    
    print(f"\n3. Parâmetros para Hora {hora_desejada}:")
    print(f"   Fator de carga médio: {np.mean(fator_carga_hora):.3f}")
    if sistema.NGER_EOL > 0:
        print(f"   Fator de vento médio: {np.mean(fator_vento_hora):.3f}")

    # SOC das baterias (exemplo)
    soc_baterias = {}
    for b in sistema.BARRAS_COM_BATERIA:
        soc_baterias[b] = 0.5  # 50% para todas

    # -------------------------------------------------------------------------
    # 4. Resolver
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
        write_lp=True
    )

    status = modelo.model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
    print(f"\nStatus da solução: {status}")

    # -------------------------------------------------------------------------
    # 5. Extrair e exibir resultados
    # -------------------------------------------------------------------------
    if status == poi.TerminationStatusCode.OPTIMAL:
        resultado = modelo.extract_results(hora=hora_desejada, cen_id=cen_id)
        
        print(f"\n5. Resultados para Hora {hora_desejada}:")
        print(f"   Demanda total: {sum(resultado.PLOAD):.2f} MW")
        print(f"   Geração térmica total: {sum(resultado.PGER):.2f} MW")
        if sistema.NGER_EOL > 0:
            print(f"   Geração eólica total: {sum(resultado.PGWIND):.2f} MW")
            print(f"   Curtailment total: {sum(resultado.CURTAILMENT):.2f} MW")
        print(f"   Déficit total: {sum(resultado.DEFICIT):.2f} MW")
        
        if sistema.BARRAS_COM_BATERIA:
            for b in sistema.BARRAS_COM_BATERIA:
                print(f"   Bateria barra {b}: operação = {resultado.BESS_operation[b]:.2f} MW (SOC={resultado.SOC_init[b]/sistema.BATTERY_CAPACITY[b]:.1%})")
        
        print(f"   CMO (barra slack): {resultado.CMO[0]:.2f} $/MWh")

    print("\n" + "=" * 70)
    print("EXECUÇÃO CONCLUÍDA")
    print("=" * 70)