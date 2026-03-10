#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições para geradores eólicos em modelo multi-período.
"""
from pyomo.environ import *

class WindGeneratorConstraints:
    """Restrições de geradores eólicos."""

    @staticmethod
    def add_constraints(model, sistema):
        """
        Adiciona restrições para geradores eólicos.
        Assume que model possui:
        - T
        - WIND_GENERATORS (conjunto)
        - PGWIND (variável)
        - CURTAILMENT (variável)
        - PGWIND_AVAIL (parâmetro)
        """
        if not hasattr(model, 'WIND_GENERATORS') or len(model.WIND_GENERATORS) == 0:
            return

        # Balanço: geração + corte = disponível
        def wind_balance_rule(m, t, w):
            return m.PGWIND[t, w] + m.CURTAILMENT[t, w] == m.PGWIND_AVAIL[t, w]
        model.WindBalance = Constraint(model.T, model.WIND_GENERATORS, rule=wind_balance_rule)

      
        def curtailment_limit(m, t, w): 
            return m.CURTAILMENT[t, w] <= m.PGWIND_AVAIL[t, w]
        model.CurtailmentLimit = Constraint(model.T, model.WIND_GENERATORS, rule=curtailment_limit)