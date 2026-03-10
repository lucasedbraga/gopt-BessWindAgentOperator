from pyomo.environ import *

class ThermalGeneratorConstraints:
    @staticmethod
    def add_constraints(model, sistema):
        if not hasattr(model, 'CONV_GENERATORS') or len(model.CONV_GENERATORS) == 0:
            return

        # Converte rampas de MW/h para pu por hora (1 período = 1h)
        ramp_up_pu = sistema.RAMP_UP / sistema.SB
        ramp_down_pu = sistema.RAMP_DOWN / sistema.SB

        # Limites de geração (já em pu)
        def gen_limits_rule(m, t, g):
            return inequality(sistema.PGMIN_CONV[g], m.PGER[t, g], sistema.PGMAX_CONV[g])
        model.ThermalGenLimits = Constraint(model.T, model.CONV_GENERATORS, rule=gen_limits_rule)

        # Primeiro período: comparar com geração inicial
        def first_ramp_up_rule(m, g):
            return m.PGER[0, g] <= sistema.PGER_INICIAL_CONV[g] + ramp_up_pu[g]
        model.FirstRampUp = Constraint(model.CONV_GENERATORS, rule=first_ramp_up_rule)

        def first_ramp_down_rule(m, g):
            return m.PGER[0, g] >= sistema.PGER_INICIAL_CONV[g] - ramp_down_pu[g]
        model.FirstRampDown = Constraint(model.CONV_GENERATORS, rule=first_ramp_down_rule)

        # Períodos seguintes: comparar com período anterior
        def ramp_up_rule(m, t, g):
            if t == 0:
                return Constraint.Skip
            return m.PGER[t, g] <= m.PGER[t-1, g] + ramp_up_pu[g]
        model.RampUp = Constraint(model.T, model.CONV_GENERATORS, rule=ramp_up_rule)

        def ramp_down_rule(m, t, g):
            if t == 0:
                return Constraint.Skip
            return m.PGER[t, g] >= m.PGER[t-1, g] - ramp_down_pu[g]
        model.RampDown = Constraint(model.T, model.CONV_GENERATORS, rule=ramp_down_rule)