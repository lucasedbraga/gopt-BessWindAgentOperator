from pyomo.environ import *
import numpy as np

class WindGeneratorConstraints:
    """Restrições de Geradores Eólicos"""


    @staticmethod
    def add_wind_generator_constraints(model, sistema):
        
        # Verificar se há baterias no sistema
        if not hasattr(model, 'BATTERIES') or len(model.BATTERIES) == 0:
            return
        
        # Variáveis de bateria (apenas se houver baterias)
        if len(model.GWD_GENERATORS) > 0:
            model.PGWIND = Var(model.GWD_GENERATORS, within=NonNegativeReals)
            model.CURTAILMENT = Var(model.GWD_GENERATORS, within=NonNegativeReals)
        
        """Adiciona restrições para geradores eólicos"""
        # Geração utilizada não pode exceder disponível
        def wind_balance_rule(m, g):
            return m.PGWIND[g] <= m.PGER[g]
        
        model.WindBalance = Constraint(model.GWD_GENERATORS, rule=wind_balance_rule)
        
        # Definição de curtailment
        def curtailment_definition_rule(m, g):
            return m.CURTAILMENT[g] == m.PGER[g] - m.PGWIND[g]
        
        model.CurtailmentDefinition = Constraint(model.GWD_GENERATORS, rule=curtailment_definition_rule)