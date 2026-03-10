#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições elétricas: fluxo nas linhas, limites, balanço de potência.
"""
from pyomo.environ import *

class ElectricConstraints:
    """Restrições da rede elétrica."""

    @staticmethod
    def add_constraints(model, sistema, considerar_perdas=False):
        """
        Adiciona restrições de fluxo, limites de linha e balanço de potência.
        - model: modelo Pyomo (deve ter T, BUSES, LINES, ANG, FLUXO_LIN, DEFICIT, etc.)
        - sistema: objeto com dados da rede
        - considerar_perdas: se True, inclui perdas no balanço (model deve ter PERDAS_BARRA)
        """
        # --------------------------------------------------------------------
        # Definição do fluxo de potência (DC)
        # --------------------------------------------------------------------
        def flow_def_rule(m, t, e):
            i = sistema.line_fr[e]
            j = sistema.line_to[e]
            return m.FLUXO_LIN[t, e] == (m.ANG[t, i] - m.ANG[t, j]) / sistema.x_line[e]
        model.FlowDef = Constraint(model.T, model.LINES, rule=flow_def_rule)

        # --------------------------------------------------------------------
        # Limites de fluxo
        # --------------------------------------------------------------------
        def flow_limits_rule(m, t, e):
            return inequality(-sistema.FLIM[e], m.FLUXO_LIN[t, e], sistema.FLIM[e])
        model.FlowLimits = Constraint(model.T, model.LINES, rule=flow_limits_rule)

        # --------------------------------------------------------------------
        # Limites de déficit (opcional)
        # --------------------------------------------------------------------
        def deficit_limits_rule(m, t, b):
            return m.DEFICIT[t, b] <= 2 * m.PLOAD[t, b]  # exemplo: até 2x a carga
        model.DeficitLimits = Constraint(model.T, model.BUSES, rule=deficit_limits_rule)

        # --------------------------------------------------------------------
        # Balanço de potência em cada barra
        # --------------------------------------------------------------------
        def power_balance_rule(m, t, b):
            # Geração térmica na barra
            ger_conv = sum(m.PGER[t, g] for g in m.CONV_GENERATORS if sistema.BARPG_CONV[g] == b)

            # Geração eólica na barra
            ger_eol = 0
            if hasattr(m, 'WIND_GENERATORS') and len(m.WIND_GENERATORS) > 0:
                ger_eol = sum(m.PGWIND[t, w] for w in m.WIND_GENERATORS if sistema.BARPG_EOL[w] == b)

            # Déficit
            deficit = m.DEFICIT[t, b]

            # Baterias (se existirem)
            bateria = 0
            if hasattr(m, 'BATTERIES') and b in m.BATTERIES:
                bateria = m.DISCHARGE[t, b] - m.CHARGE[t, b]

            # Fluxo líquido (injetado - retirado)
            fluxo_liquido = 0
            for e in m.LINES:
                if sistema.line_fr[e] == b:
                    fluxo_liquido += m.FLUXO_LIN[t, e]
                elif sistema.line_to[e] == b:
                    fluxo_liquido -= m.FLUXO_LIN[t, e]

            # Carga
            carga = m.PLOAD[t, b]

            # Perdas (se ativadas)
            perdas = m.PERDAS_BARRA[t, b] if considerar_perdas else 0.0

            return ger_conv + ger_eol + deficit + bateria - fluxo_liquido == carga + perdas
        model.PowerBalance = Constraint(model.T, model.BUSES, rule=power_balance_rule)