from pyomo.environ import *
import numpy as np
from typing import Dict, List, Tuple

class DCElectricConstraints:
    """Restrições elétricas DC - adaptadas para nova estrutura (convencionais e eólicos separados)"""
    
    @staticmethod
    def add_power_balance_constraints(model, sistema, considerar_perdas=False):
        """Adiciona restrições de balanço de potência"""
        def power_balance_rule(m, i):
            # Geração convencional na barra i
            geracao_conv = 0.0
            if hasattr(m, 'CONV_GENERATORS'):
                for g in m.CONV_GENERATORS:
                    if sistema.BARPG_CONV[g] == i:
                        geracao_conv += m.PGER[g]

            # Geração eólica na barra i
            geracao_eol = 0.0
            if hasattr(m, 'WIND_GENERATORS') and len(m.WIND_GENERATORS) > 0:
                for g in m.WIND_GENERATORS:
                    if sistema.BARPG_EOL[g] == i:
                        geracao_eol += m.PGWIND[g]

            # Déficit na barra
            deficit_barra = m.DEFICIT[i]

            # Fluxos líquidos (injetado - retirado)
            fluxo_liquido = 0.0
            for e in m.LINES:
                if sistema.line_fr[e] == i:
                    fluxo_liquido += m.FLUXO_LIN[e]
                elif sistema.line_to[e] == i:
                    fluxo_liquido -= m.FLUXO_LIN[e]

            perdas = 0.0
            if considerar_perdas:
                perdas = m.PERDAS_BARRA[i]

            # Contribuição das baterias (se houver)
            contribuicao_bateria = 0.0
            if hasattr(m, 'BATTERIES') and i in m.BATTERIES:
                contribuicao_bateria = m.BatteryOperation[i]

            # Balanço: geração total + déficit + bateria - fluxo líquido - perdas = carga
            return (geracao_conv + geracao_eol + deficit_barra + contribuicao_bateria 
                    - fluxo_liquido - perdas == sistema.PLOAD[i])

        model.PowerBalance = Constraint(model.BUSES, rule=power_balance_rule)

    @staticmethod
    def add_line_flow_constraints(model, sistema):
        """
        Adiciona restrições de fluxo nas linhas.
        Cria o parâmetro mutável PERDAS_LINHA se não existir.
        """
        # Garantir que PERDAS_LINHA exista (inicializado com zero)
        if not hasattr(model, 'PERDAS_LINHA'):
            model.PERDAS_LINHA = Param(model.LINES, mutable=True, initialize=0.0)

        def fluxo_definition_rule(m, e):
            i = sistema.line_fr[e]
            j = sistema.line_to[e]
            return m.FLUXO_LIN[e] == (m.ANG[i] - m.ANG[j]) / sistema.x_line[e]

        model.FluxoDefinition = Constraint(model.LINES, rule=fluxo_definition_rule)

        def fluxo_max_pos_rule(m, e):
            # Fluxo + metade da perda <= limite positivo
            return m.FLUXO_LIN[e] + m.PERDAS_LINHA[e]/2 <= sistema.FLIM[e]

        def fluxo_max_neg_rule(m, e):
            # Fluxo + metade da perda >= limite negativo
            return m.FLUXO_LIN[e] + m.PERDAS_LINHA[e]/2 >= -sistema.FLIM[e]

        model.FluxoMaxPos = Constraint(model.LINES, rule=fluxo_max_pos_rule)
        model.FluxoMaxNeg = Constraint(model.LINES, rule=fluxo_max_neg_rule)

    @staticmethod
    def add_generator_limits_constraints(model, sistema):
        """Adiciona limites de geração para geradores CONVENCIONAIS (não eólicos)"""
        if not hasattr(model, 'CONV_GENERATORS'):
            return

        def conv_limits_rule(m, g):
            # g pertence a CONV_GENERATORS
            return inequality(sistema.PGMIN_CONV[g], m.PGER[g], sistema.PGMAX_CONV[g])

        model.ConvGenLimits = Constraint(model.CONV_GENERATORS, rule=conv_limits_rule)

    @staticmethod
    def add_deficit_constraints(model, sistema):
        """Adiciona restrições para déficit (inalterado)"""
        def deficit_limits_rule(m, i):
            # Déficit limitado a um múltiplo da carga (aqui 2x, pode ajustar)
            return m.DEFICIT[i] <= sistema.PLOAD[i] * 2.0

        model.DeficitLimits = Constraint(model.BUSES, rule=deficit_limits_rule)