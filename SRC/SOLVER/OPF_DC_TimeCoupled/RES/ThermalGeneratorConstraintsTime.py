#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições para geradores térmicos (convencionais) em modelo multi-período.
"""
from pyomo.environ import *

class ThermalGeneratorConstraints:
    """Restrições de geradores térmicos."""

    @staticmethod
    def add_constraints(model, sistema):
        """
        Adiciona restrições para geradores térmicos.
        Assume que model já possui:
        - T (conjunto de períodos)
        - CONV_GENERATORS (conjunto de geradores)
        - PGER (variável indexada por T e gerador)
        """
        if not hasattr(model, 'CONV_GENERATORS') or len(model.CONV_GENERATORS) == 0:
            return

        # Limites de geração (mínimo e máximo)
        def gen_limits_rule(m, t, g):
            return inequality(sistema.PGMIN_CONV[g], m.PGER[t, g], sistema.PGMAX_CONV[g])
        model.ThermalGenLimits = Constraint(model.T, model.CONV_GENERATORS, rule=gen_limits_rule)

        # Outras restrições podem ser adicionadas aqui (rampas, etc.)