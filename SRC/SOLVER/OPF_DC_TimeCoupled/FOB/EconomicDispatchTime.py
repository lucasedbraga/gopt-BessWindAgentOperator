
from pyomo.environ import *
# =============================================================================
# Função objetivo (
# integrada no script, conforme solicitado)
# =============================================================================
class ObjectiveFunction:
    """Função objetivo para minimização do custo total de operação."""
    
    @staticmethod
    def add_objective(model, sistema):
        """
        Adiciona a função objetivo ao modelo.
        Considera custos de geração térmica, penalidade de corte eólico, déficit e operação das baterias.
        """
        def objective_rule(m):
            total = 0
            # Custo dos geradores térmicos
            if hasattr(sistema, 'CPG_CONV'):
                for t in m.T:
                    for g in m.CONV_GENERATORS:
                        total += sistema.CPG_CONV[g] * m.PGER[t, g]
            # Penalidade por corte eólico
            if hasattr(m, 'WIND_GENERATORS') and len(m.WIND_GENERATORS) > 0 and hasattr(sistema, 'CPG_CURTAILMENT'):
                for t in m.T:
                    for w in m.WIND_GENERATORS:
                        total += sistema.CPG_CURTAILMENT[w] * m.CURTAILMENT[t, w]
            # Custo do déficit
            if hasattr(sistema, 'CPG_DEFICIT'):
                for t in m.T:
                    for b in m.BUSES:
                        total += sistema.CPG_DEFICIT * m.DEFICIT[t, b]
            # Custo de operação das baterias (desgaste)
            if hasattr(m, 'BATTERIES') and len(m.BATTERIES) > 0 and hasattr(sistema, 'BATTERY_COST'):
                for t in m.T:
                    for b in m.BATTERIES:
                        # Supondo que BATTERY_COST seja um vetor indexado por barra
                        custo_bateria = sistema.BATTERY_COST[b] if hasattr(sistema.BATTERY_COST, '__getitem__') else sistema.BATTERY_COST
                        total += custo_bateria * (m.CHARGE[t, b] + m.DISCHARGE[t, b])
            return total
        
        model.TotalCost = Objective(rule=objective_rule, sense=minimize)

