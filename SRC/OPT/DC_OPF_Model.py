from pyomo.environ import *
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class OPFResult:
    """Resultado do OPF"""
    sucesso: bool
    PGER: List[float] = field(default_factory=list)
    PGWIND: List[float] = field(default_factory=list)
    CURTAILMENT: List[float] = field(default_factory=list)
    SOC_init: List[float] = field(default_factory=list)
    BESS_operation: List[float] = field(default_factory=list)
    SOC_atual: List[float] = field(default_factory=list)
    DEFICIT: List[float] = field(default_factory=list)

    V: List[float] = field(default_factory=list)
    ANG: List[float] = field(default_factory=list)
    FLUXO_LIN: List[float] = field(default_factory=list)

    CUSTO: List[float] = field(default_factory=list)
    CMO: List[float] = field(default_factory=list)
    PERDAS: List[float] = field(default_factory=list)

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
            'PERDAS': self.PERDAS,

            'mensagem': self.mensagem,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'tempo_execucao': self.tempo_execucao,
        }

@dataclass
class DC_OPF_Model:
    """Modelo físico do sistema (apenas restrições, sem objetivo)"""
    model: ConcreteModel = field(default_factory=ConcreteModel)
    sistema: Optional[object] = None
    considerar_perdas: bool = False
    
    def build(self, sistema, considerar_perdas=False):
        """Constrói modelo físico com todas as restrições"""
        self.sistema = sistema
        self.considerar_perdas = considerar_perdas

        m = self.model
        s = self.sistema
        
        # Limpar modelo se já existir
        m.clear()
        
        # === CONJUNTOS ===
        m.BUSES = Set(initialize=range(s.NBAR))
        m.GENERATORS = Set(initialize=range(s.NGER))
        m.LINES = Set(initialize=range(s.NLIN))

        # === VARIÁVEIS ===
        m.PGER = Var(m.GENERATORS, within=NonNegativeReals)
        m.DEFICIT = Var(m.BUSES, within=NonNegativeReals)

        m.V = Var(m.BUSES, within=NonNegativeReals, bounds=(0.95, 1.05))
        m.ANG = Var(m.BUSES, within=Reals, bounds=(-3.14, 3.14))
        m.FLUXO_LIN = Var(m.LINES, within=Reals)

        # Conjunto de geradores eólicos (se houver)
        if hasattr(s, 'BAR_GWD') and len(s.BAR_GWD) > 0:
            m.GWD_GENERATORS = Set(initialize=s.BAR_GWD)
        else:
            m.GWD_GENERATORS = Set(initialize=[])

        # Conjunto de baterias (se houver)
        if hasattr(s, 'BATTERIES') and len(s.BATTERIES) > 0:
            m.BATTERIES = Set(initialize=s.BATTERIES)
        else:
            m.BATTERIES = Set(initialize=[])
        
        # Parâmetro para perdas (mutável para iteração)
        if considerar_perdas:
            m.PERDAS = Param(m.BUSES, mutable=True, default=0.0)

        # Fixar tensão
        for bus in m.BUSES:
            m.V[bus].fix(1.0)
        # Fixar ângulo da barra slack
        m.ANG[s.slack_idx].fix(0.0)

        # === RESTRIÇÕES ===
        self.add_constraints()
        
        return m
    
    def add_constraints(self):
        """Adiciona todas as restrições ao modelo"""
        # Importar módulos de restrições
        from SRC.SOLVER.RES.EletricConstraints import DCElectricConstraints
        from SRC.SOLVER.RES.WindGeneratorConstraints import WindGeneratorConstraints
        from SRC.SOLVER.RES.BatteryConstraints import BatteryConstraints
        

        if len(self.model.GWD_GENERATORS) > 0:
            WindGeneratorConstraints.add_wind_generator_constraints(
                self.model, self.sistema
            )

        # Adicionar restrições de bateria (se houver)
        if len(self.model.BATTERIES) > 0:
            BatteryConstraints.add_battery_constraints(
                self.model, self.sistema
            )
        
        # Adicionar restrições elétricas
        DCElectricConstraints.add_generator_limits_constraints(
            self.model, self.sistema
        )

        DCElectricConstraints.add_deficit_constraints(
            self.model, self.sistema
        )
        DCElectricConstraints.add_line_flow_constraints(
            self.model, self.sistema
        )

        DCElectricConstraints.add_power_balance_constraints(
            self.model, self.sistema, self.considerar_perdas
        )
    
    def update_losses(self, perdas_barra: np.ndarray):
        """Atualiza parâmetros de perdas no modelo"""
        if not self.considerar_perdas:
            return
        
        m = self.model
        for i in m.BUSES:
            if i < len(perdas_barra):
                m.PERDAS[i] = perdas_barra[i]
    
    def calculate_losses(self) -> np.ndarray:
        """Calcula perdas nas linhas baseadas na solução atual"""
        s = self.sistema
        perdas_barra = np.zeros(s.NBAR)
        
        for e in self.model.LINES:
            i = s.line_fr[e]
            j = s.line_to[e]
            
            # Obter fluxo da solução
            fluxo_val = value(self.model.FLUXO[e]) if hasattr(self.model, 'FLUXO') else 0
            
            # Calcular perdas na linha (DC simplificado)
            r = s.r_line[e]
            perdas_linha = r * (fluxo_val ** 2)
            
            # Distribuir igualmente entre barras
            perdas_barra[i] += perdas_linha / 2
            perdas_barra[j] += perdas_linha / 2
        
        return perdas_barra
    
    def extract_results(self, results, iteracoes: int = 1) -> OPFResult:
        """Extrai resultados da solução"""
        m = self.model
        s = self.sistema
        
        # Verificar status
        if results.solver.termination_condition != TerminationCondition.optimal:
            return OPFResult(
                sucesso=False,
                PGER=[0.0] * s.NGER,
                PGWIND=[0.0] * len(s.BAR_GWD),
                CURTAILMENT=[0.0] * len(s.BAR_GWD),
                SOC_init=[0.0] * len(s.BATTERIES),
                BESS_operation=[0.0] * len(s.BATTERIES),
                SOC_atual=[0.0] * len(s.BATTERIES),
                DEFICIT=[0.0] * s.NBAR,

                V= [0.0] * s.NBAR,
                ANG=[0.0] * s.NBAR,
                FLUXO_LIN= [0.0] * s.NLIN,

                CUSTO=[0.0] * s.NBAR,
                CMO=[0.0] * s.NBAR,
                PERDAS=[0.0] * s.NLIN,

                mensagem=f"Solver terminou com: {results.solver.termination_condition}",
                timestamp=datetime.now()
            )
        
        # Extrair valores
        PGER_val = [value(m.PGER[g]) for g in m.GENERATORS]
        PGWIND = [value(m.PGWIND[g]) for g in m.GWD_GENERATORS]
        CURTAILMENT = [value(m.CURTAILMENT[g]) for g in m.GWD_GENERATORS]
        SOC_init = [value(m.SOC_init[b]) for b in m.BATTERIES]
        SOC_atual = [value(m.SOC_atual[b]) for b in m.BATTERIES]
        DEFICIT = [value(m.DEFICIT[b]) for b in m.BUSES]

        V = [value(m.V[b]) for b in m.BUSES]
        ANG = [value(m.ANG[b]) for b in m.BUSES]
        FLUXO_LIN = [value(m.FLUXO_LIN[e]) for e in m.LINES]

        # Calcular o valor líquido para cada gerador
        PGER = [] 
        curtailment_dict = {g: value(m.CURTAILMENT[g]) for g in m.GWD_GENERATORS}

        for i, g in enumerate(m.GENERATORS):
            if g in m.GWD_GENERATORS:
                # Para geradores GWD: Geração Bruta - Curtailment
                liquid_value = PGER_val[i] - curtailment_dict[g]
                PGER.append(liquid_value)

            else:
                # Para outros geradores: mantém a geração bruta
                PGER.append(PGER_val[i])
        
        # Custos
        CUSTO = [DEFICIT[b] * 5000.0 for b in m.BUSES]

        # CMO (Custo Marginal de Operação) - dual da restrição de balanço
        CMO = 0.0
        if hasattr(m, 'dual') and hasattr(m, 'PowerBalance'):
            try:
                CMO = m.dual[m.PowerBalance[s.slack_idx]]
            except:
                CMO = 0.0

        # Perdas
        PERDAS = 0.0
        if self.considerar_perdas:
            PERDAS = sum(value(m.PERDAS[b]) for b in m.BUSES)
        
        # Adiciona extração de estado da bateria (SOC, operação, potência)
        BESS_operation = []
        if hasattr(m, 'BUSES') and len(m.BUSES) > 0:
            for b in m.BUSES:
                if b in m.BATTERIES:
                    charge = value(m.CHARGE[b]) if hasattr(m, 'CHARGE') else 0.0
                    discharge = value(m.DISCHARGE[b]) if hasattr(m, 'DISCHARGE') else 0.0
                    pot_inst_bess = charge - discharge
                    BESS_operation.append(pot_inst_bess)
                else:
                    BESS_operation.append(0.0)

        return OPFResult(
            sucesso=True,

            PGER=PGER,
            PGWIND=PGWIND,
            CURTAILMENT=CURTAILMENT,
            SOC_init=SOC_init,
            SOC_atual=SOC_atual,
            BESS_operation=BESS_operation,
            DEFICIT=DEFICIT,

            V=V,
            ANG=ANG,
            FLUXO_LIN=FLUXO_LIN,

            CUSTO=CUSTO,
            CMO=CMO,
            PERDAS=PERDAS,

            mensagem="",
            timestamp=datetime.now(),
            tempo_execucao=results.solver.time if hasattr(results.solver, 'time') else 0.0
        )