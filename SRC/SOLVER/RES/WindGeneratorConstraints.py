from pyomo.environ import *
import numpy as np

class WindGeneratorConstraints:
    """Restrições de Geradores Eólicos"""

    @staticmethod
    def add_wind_generator_constraints(model, sistema):
        """
        Adiciona restrições para geradores eólicos ao modelo Pyomo.
        As variáveis são definidas como não negativas e a equação de balanço
        aloca toda a energia disponível entre geração e curtailment.
        """
        # Verificar se há geradores eólicos
        if not hasattr(model, 'WIND_GENERATORS') or len(model.WIND_GENERATORS) == 0:
            return

        # Criar variáveis para geradores eólicos (todas não negativas)
        model.PGWIND = Var(model.WIND_GENERATORS, within=NonNegativeReals,
                           doc='Geração eólica utilizada (MW)')
        model.PGWIND_disponivel = Var(model.WIND_GENERATORS, within=NonNegativeReals,
                                      doc='Geração eólica disponível (MW) – fixa')
        model.CURTAILMENT = Var(model.WIND_GENERATORS, within=NonNegativeReals,
                                doc='Curtailment eólico (MW)')

        # Restrição: fixar a disponibilidade eólica com base no perfil de vento
        def wind_available_power_rule(m, g):
            # g é o índice do gerador eólico no conjunto (0..NGER_EOL-1)
            return m.PGWIND_disponivel[g] == sistema.PGWIND_disponivel[g]
        model.WindAvailablePower = Constraint(model.WIND_GENERATORS,
                                              rule=wind_available_power_rule)

        # Restrição fundamental: geração utilizada + curtailment = disponível
        def wind_energy_balance_rule(m, g):
            return m.PGWIND[g] + m.CURTAILMENT[g] == m.PGWIND_disponivel[g]
        model.WindEnergyBalance = Constraint(model.WIND_GENERATORS,
                                             rule=wind_energy_balance_rule)

        # Limites superiores individuais (redundantes, mas ajudam o solver)
        def pgwind_upper_bound_rule(m, g):
            return m.PGWIND[g] <= m.PGWIND_disponivel[g]
        model.PGWIND_UpperBound = Constraint(model.WIND_GENERATORS,
                                             rule=pgwind_upper_bound_rule)

        def curtailment_upper_bound_rule(m, g):
            return m.CURTAILMENT[g] <= m.PGWIND_disponivel[g]
        model.Curtailment_UpperBound = Constraint(model.WIND_GENERATORS,
                                                  rule=curtailment_upper_bound_rule)