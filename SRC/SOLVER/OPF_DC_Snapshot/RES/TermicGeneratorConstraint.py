from pyomo.environ import *

class TermicGeneratorConstraints:
    """
    Restrições para geradores térmicos em um snapshot, com limites de geração
    e restrições de rampa baseadas na geração do período anterior.
    O vetor de geração anterior (last_generation) deve ser fornecido a cada chamada
    e atualizado externamente com a solução do snapshot.
    """

    @staticmethod
    def add_ramp_constraints(model, sistema):
        """
        Adiciona restrições de rampa de subida e descida para cada gerador térmico,
        limitando a variação da geração atual em relação ao valor anterior.

        Args:
            model: Modelo Pyomo contendo:
                - Conjunto TERMIC_GENERATORS: índices dos geradores térmicos
                - Variável PGER[g]: geração atual do gerador g (MW)
            sistema: Objeto com atributos:
                - ramp_up_MW_h[g]: taxa máxima de aumento (MW/h)
                - ramp_down_MW_h[g]: taxa máxima de redução (MW/h)
            last_generation: dict {g: valor} com a geração de cada térmica no snapshot anterior.
                             Deve conter todos os geradores em TERMIC_GENERATORS.
            delta_t: Intervalo de tempo entre este snapshot e o anterior (horas). Padrão = 1.
        """
        if not hasattr(model, 'PGER'):
            raise AttributeError("Modelo precisa da variável 'PGER[g]'.")

        # Restrição de subida (ramp up)
        def ramp_up_rule(m, g):
            # Geração atual não pode exceder a anterior + ramp_up
            return m.PGER[g] <= sistema.last_generation[g] + sistema.RAMP_UP[g]

        model.RampUp = Constraint(model.CONV_GENERATORS, rule=ramp_up_rule)

        # Restrição de descida (ramp down)
        def ramp_down_rule(m, g):
            # Geração atual não pode ser inferior à anterior - ramp_down
            return m.PGER[g] >= sistema.last_generation[g] - sistema.RAMP_DOWN[g]

        model.RampDown = Constraint(model.CONV_GENERATORS, rule=ramp_down_rule)

    @staticmethod
    def add_generator_limits_constraints(model, sistema):
        """
        Adiciona limites mínimo e máximo de geração para geradores térmicos.

        Args:
            model: Modelo Pyomo com TERMIC_GENERATORS e PGER[g]
            sistema: Objeto com atributos:
                - PGMIN_TERM[g]: geração mínima do gerador g (MW)
                - PGMAX_TERM[g]: geração máxima do gerador g (MW)
        """
        if not hasattr(model, 'TERMIC_GENERATORS') or not hasattr(model, 'PGER'):
            return  # Conjuntos ausentes, nada a fazer

        def gen_limits_rule(m, g):
            return inequality(sistema.PGMIN_TERM[g], m.PGER[g], sistema.PGMAX_TERM[g])

        model.TermicGenLimits = Constraint(model.TERMIC_GENERATORS, rule=gen_limits_rule)