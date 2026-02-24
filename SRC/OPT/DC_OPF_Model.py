from pyomo.environ import *
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class OPFResult:
    """Resultado do OPF"""
    sucesso: bool
    PGER: List[float] = field(default_factory=list)               # geração convencional (índice = gerador conv)
    PGWIND_disponivel: List[float] = field(default_factory=list) # disponibilidade eólica (índice = eólico)
    PGWIND: List[float] = field(default_factory=list)            # geração eólica utilizada (índice = eólico)
    CURTAILMENT: List[float] = field(default_factory=list)       # corte eólico (índice = eólico)
    SOC_init: List[float] = field(default_factory=list)          # SOC inicial (por barra)
    BESS_operation: List[float] = field(default_factory=list)    # operação da bateria (por barra)
    SOC_atual: List[float] = field(default_factory=list)         # SOC final (por barra)
    DEFICIT: List[float] = field(default_factory=list)           # déficit por barra

    V: List[float] = field(default_factory=list)
    ANG: List[float] = field(default_factory=list)
    FLUXO_LIN: List[float] = field(default_factory=list)

    CUSTO: List[float] = field(default_factory=list)
    CMO: List[float] = field(default_factory=list)
    PERDAS_BARRA: List[float] = field(default_factory=list)

    mensagem: str = ""
    timestamp: Optional[datetime] = None
    tempo_execucao: float = 0.0

    def to_dict(self) -> Dict:
        """Converte resultado para dicionário"""
        return {
            'sucesso': self.sucesso,
            'PGER': self.PGER,
            'PGWIND': self.PGWIND,
            'CURTAILMENT': self.CURTAILMENT,
            'SOC_init': self.SOC_init,
            'SOC_atual': self.SOC_atual,
            'DEFICIT': self.DEFICIT,
            'V': self.V,
            'ANG': self.ANG,
            'FLUXO_LIN': self.FLUXO_LIN,
            'CUSTO': self.CUSTO,
            'CMO': self.CMO,
            'PERDAS_BARRA': self.PERDAS_BARRA,
            'mensagem': self.mensagem,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'tempo_execucao': self.tempo_execucao,
        }


@dataclass
class DC_OPF_Model:
    """Modelo físico do sistema (apenas restrições, sem objetivo)"""
    def __init__(self):
            self.model = None          # será criado em build()
            self.sistema = None
            self.considerar_perdas = True

    def build(self, sistema, considerar_perdas=False):
        """Constrói modelo físico com todas as restrições, usando a nova estrutura de dados."""
        self.sistema = sistema
        self.considerar_perdas = considerar_perdas

        m = ConcreteModel()
        self.model = m
        s = self.sistema

        m.clear()

        # === CONJUNTOS ===
        m.BUSES = Set(initialize=range(s.NBAR))
        m.CONV_GENERATORS = Set(initialize=range(s.NGER_CONV))   # geradores convencionais
        m.LINES = Set(initialize=range(s.NLIN))

        # Geradores eólicos (se houver)
        if s.NGER_EOL > 0:
            m.WIND_GENERATORS = Set(initialize=range(s.NGER_EOL))
        else:
            m.WIND_GENERATORS = Set(initialize=[])

        # Baterias (se houver)
        if hasattr(s, 'BATTERIES') and len(s.BATTERIES) > 0:
            m.BATTERIES = Set(initialize=s.BATTERIES)   # índices das barras com bateria
        else:
            m.BATTERIES = Set(initialize=[])

        # === VARIÁVEIS ===
        # Geração convencional
        m.PGER = Var(m.CONV_GENERATORS, within=NonNegativeReals)

        # Geração eólica e curtailment (se existirem)
        if len(m.WIND_GENERATORS) > 0:
            m.PGWIND = Var(m.WIND_GENERATORS, within=NonNegativeReals)
            m.PGWIND_disponivel = Var(m.WIND_GENERATORS, within=NonNegativeReals)
            m.CURTAILMENT = Var(m.WIND_GENERATORS, within=NonNegativeReals)

        # Déficit por barra
        m.DEFICIT = Var(m.BUSES, within=NonNegativeReals)

        # Variáveis elétricas
        m.V = Var(m.BUSES, within=NonNegativeReals, bounds=(0.95, 1.05))
        m.ANG = Var(m.BUSES, within=Reals, bounds=(-3.14, 3.14))
        m.FLUXO_LIN = Var(m.LINES, within=Reals)

        # Parâmetro para perdas (mutável)
        if considerar_perdas:
            # Verifica se os parâmetros de perda já existem, se não, cria
            if not hasattr(m, 'PERDAS_LINHA'):
                m.PERDAS_LINHA = Param(m.LINES, mutable=True, initialize=0.0)
            if not hasattr(m, 'PERDAS_BARRA'):
                m.PERDAS_BARRA = Param(m.BUSES, mutable=True, initialize=0.0)

        # Fixações iniciais
        for bus in m.BUSES:
            m.V[bus].fix(1.0)
        m.ANG[s.slack_idx].fix(0.0)

        # === RESTRIÇÕES ===
        self.add_constraints()

        return m

    def add_constraints(self):
        """Adiciona todas as restrições ao modelo."""
        from SOLVER.RES.EletricConstraints import DCElectricConstraints
        from SOLVER.RES.WindGeneratorConstraints import WindGeneratorConstraints
        from SOLVER.RES.BatteryConstraints import BatteryConstraints

        m = self.model
        s = self.sistema

        # Restrições eólicas (se houver)
        if len(m.WIND_GENERATORS) > 0:
            WindGeneratorConstraints.add_wind_generator_constraints(m, s)

        # Restrições de bateria (se houver)
        if len(m.BATTERIES) > 0:
            BatteryConstraints.add_battery_constraints(m, s)

        # Limites dos geradores convencionais
        DCElectricConstraints.add_generator_limits_constraints(m, s)

        # Déficit
        DCElectricConstraints.add_deficit_constraints(m, s)

        # Fluxo nas linhas
        DCElectricConstraints.add_line_flow_constraints(m, s)

        # Balanço de potência (inclui geração eólica e baterias)
        DCElectricConstraints.add_power_balance_constraints(m, s, self.considerar_perdas)

    def update_losses(self, perdas_barra: np.ndarray):
        """Atualiza parâmetros de perdas no modelo (para iterações)."""
        if not self.considerar_perdas:
            return
        m = self.model
        for i in m.BUSES:
            if i < len(perdas_barra):
                m.PERDAS_BARRA[i] = perdas_barra[i]

    def calculate_losses(self) -> np.ndarray:
        """Calcula perdas nas linhas baseadas na solução atual."""
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
        """
        Resolve o modelo considerando perdas de forma iterativa (ponto fixo).
        
        Args:
            solver (str): Nome do solver (ex: 'glpk', 'cbc', 'gurobi').
            tol (float): Tolerância para convergência (diferença máxima nos ângulos).
            max_iter (int): Número máximo de iterações.
            **solver_args: Argumentos adicionais para o solver.

        Returns:
            OPFResult: Resultado da última iteração (convergida ou não).
        """
        import numpy as np
        from pyomo.opt import SolverFactory, TerminationCondition

        if not self.considerar_perdas:
            # Sem perdas: resolve uma única vez
            solver_instance = SolverFactory(solver)
            results = solver_instance.solve(self.model, **solver_args)
            return self.extract_results(results)

        m = self.model
        s = self.sistema

        # Inicializa ângulos anteriores (zero na primeira iteração)
        ang_prev = np.zeros(s.NBAR)

        for it in range(max_iter):
            # Resolve o modelo com as perdas atuais
            solver_instance = SolverFactory(solver)
            results = solver_instance.solve(m, **solver_args)

            # Verifica se a solução é ótima
            if results.solver.termination_condition != TerminationCondition.optimal:
                return self.extract_results(results)

            # Obtém os ângulos atuais
            ang_curr = np.array([value(m.ANG[b]) for b in m.BUSES])

            # Calcula as perdas com base nos fluxos atuais e atualiza PERDAS_LINHA
            perdas_barra = self.calculate_losses()  # retorna vetor de perdas por barra

            # Atualiza o parâmetro PERDAS_BARRA no modelo
            self.update_losses(perdas_barra)

            # Verifica convergência (diferença máxima nos ângulos)
            diff = np.max(np.abs(ang_curr - ang_prev))
            if diff < tol:
                print(f"Convergência alcançada na iteração {it+1} (diff = {diff:.6f})")
                break

            ang_prev = ang_curr.copy()

            if it == max_iter - 1:
                print(f"Atenção: número máximo de iterações ({max_iter}) atingido. diff = {diff:.6f}")

        # Extrai e retorna os resultados finais
        return self.extract_results(results)

    def extract_results(self, results, iteracoes: int = 1) -> OPFResult:
        """Extrai resultados da solução."""
        m = self.model
        s = self.sistema

        if results.solver.termination_condition != TerminationCondition.optimal:
            return OPFResult(
                sucesso=False,
                PGER=[0.0] * s.NGER_CONV,
                PGWIND_disponivel=[0.0] * s.NGER_EOL,
                PGWIND=[0.0] * s.NGER_EOL,
                CURTAILMENT=[0.0] * s.NGER_EOL,
                SOC_init=[0.0] * s.NBAR,
                BESS_operation=[0.0] * s.NBAR,
                SOC_atual=[0.0] * s.NBAR,
                DEFICIT=[0.0] * s.NBAR,
                V=[0.0] * s.NBAR,
                ANG=[0.0] * s.NBAR,
                FLUXO_LIN=[0.0] * s.NLIN,
                CUSTO=[0.0] * s.NBAR,
                CMO=[0.0] * s.NBAR,
                PERDAS_BARRA=[0.0] * s.NBAR,
                mensagem=f"Solver terminou com: {results.solver.termination_condition}",
                timestamp=datetime.now()
            )

        # Geração convencional
        PGER = [value(m.PGER[g]) for g in m.CONV_GENERATORS]

        # Geração eólica e curtailment (se existirem)
        PGWIND = []
        PGWIND_disponivel = []
        CURTAILMENT = []
        if len(m.WIND_GENERATORS) > 0:
            PGWIND = [value(m.PGWIND[g]) for g in m.WIND_GENERATORS]
            PGWIND_disponivel = [value(m.PGWIND_disponivel[g]) for g in m.WIND_GENERATORS]
            CURTAILMENT = [value(m.CURTAILMENT[g]) for g in m.WIND_GENERATORS]

        # Baterias (por barra)
        SOC_init = [0.0] * s.NBAR
        BESS_operation = [0.0] * s.NBAR
        SOC_atual = [0.0] * s.NBAR
        if len(m.BATTERIES) > 0 and hasattr(m, 'SOC_init') and hasattr(m, 'SOC_atual') and hasattr(m, 'CHARGE') and hasattr(m, 'DISCHARGE'):
            for b in m.BATTERIES:
                SOC_init[b] = value(m.SOC_init[b])
                SOC_atual[b] = value(m.SOC_atual[b])
                charge = value(m.CHARGE[b]) if hasattr(m, 'CHARGE') else 0.0
                discharge = value(m.DISCHARGE[b]) if hasattr(m, 'DISCHARGE') else 0.0
                BESS_operation[b] = charge - discharge

        # Déficit
        DEFICIT = [value(m.DEFICIT[b]) for b in m.BUSES]

        # Tensão, ângulo, fluxo
        V = [value(m.V[b]) for b in m.BUSES]
        ANG = [value(m.ANG[b]) for b in m.BUSES]
        FLUXO_LIN = [value(m.FLUXO_LIN[e]) for e in m.LINES]

        # Custos (exemplo simples: déficit penalizado)
        CUSTO = [DEFICIT[b] * s.CPG_DEFICIT for b in m.BUSES]

        # CMO (dual da restrição de balanço na barra slack)
        CMO = 0.0
        if hasattr(m, 'dual') and hasattr(m, 'PowerBalance'):
            try:
                CMO = m.dual[m.PowerBalance[s.slack_idx]]
            except:
                CMO = 0.0

        # Perdas totais
        PERDAS_BARRA = 0.0
        if self.considerar_perdas:
            PERDAS_BARRA = sum(value(m.PERDAS_BARRA[b]) for b in m.BUSES)

        return OPFResult(
            sucesso=True,
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
            CMO=CMO,
            PERDAS_BARRA=PERDAS_BARRA,
            mensagem="",
            timestamp=datetime.now(),
            tempo_execucao=results.solver.time if hasattr(results.solver, 'time') else 0.0
        )