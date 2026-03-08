#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições para baterias em modelo multi-período (com índice temporal).
Estilo simplificado, similar à versão sem tempo.
"""
from pyomo.environ import *
import numpy as np

class BatteryConstraintsTime:
    """Restrições de bateria para modelos com horizonte de tempo."""

    @staticmethod
    def add_constraints(model, sistema):
        if not hasattr(model, 'BATTERIES') or len(model.BATTERIES) == 0:
            return

        # Criar variáveis de bateria indexadas no tempo
        model.SOC = Var(model.T, model.BATTERIES, within=NonNegativeReals)
        model.CHARGE = Var(model.T, model.BATTERIES, within=NonNegativeReals)
        model.DISCHARGE = Var(model.T, model.BATTERIES, within=NonNegativeReals)

        # Eficiências (com valores padrão)
        eff_carga = getattr(sistema, 'BATTERY_CHARGE_EFF', 0.9)
        eff_descarga = getattr(sistema, 'BATTERY_DISCHARGE_EFF', 0.9)

        # Função auxiliar para acessar arrays por índice da barra
        def get_val(attr, barra, default=0):
            val = getattr(sistema, attr, None)
            if val is None:
                return default
            if isinstance(val, (list, tuple, np.ndarray)):
                if barra < len(val):
                    return val[barra]
                else:
                    return default
            elif isinstance(val, dict):
                return val.get(barra, default)
            else:
                return val

        # --------------------------------------------------------------------
        # 1. Limites operacionais da bateria (SOC mínimo e máximo)
        # --------------------------------------------------------------------
        def soc_max_rule(m, t, b):
            return m.SOC[t, b] <= get_val('BATTERY_CAPACITY', b, 0)
        model.BatterySOCMax = Constraint(model.T, model.BATTERIES, rule=soc_max_rule)

        def soc_min_rule(m, t, b):
            # Se BATTERY_MIN_SOC não existir, usa 0.1 como padrão
            min_soc = get_val('BATTERY_MIN_SOC', b, 0.1)
            return m.SOC[t, b] >= min_soc
        model.BatterySOCMin = Constraint(model.T, model.BATTERIES, rule=soc_min_rule)

        # --------------------------------------------------------------------
        # 2. Limites de potência de carga/descarga
        # --------------------------------------------------------------------
        def charge_limit_rule(m, t, b):
            return m.CHARGE[t, b] <= get_val('BATTERY_POWER_LIMIT', b, 0)
        model.BatteryChargeLimit = Constraint(model.T, model.BATTERIES, rule=charge_limit_rule)

        def discharge_limit_rule(m, t, b):
            return m.DISCHARGE[t, b] <= get_val('BATTERY_POWER_OUT', b, 0)
        model.BatteryDischargeLimit = Constraint(model.T, model.BATTERIES, rule=discharge_limit_rule)

        # --------------------------------------------------------------------
        # 3. Condição inicial (SOC do primeiro período)
        # --------------------------------------------------------------------
        def soc_initial_rule(m, b):
            soc_init = get_val('SOC_inicial', b, 0)
            return m.SOC[0, b] == soc_init
        model.BatterySOCInitial = Constraint(model.BATTERIES, rule=soc_initial_rule)

        # --------------------------------------------------------------------
        # 4. Evolução do SOC (acoplamento temporal)
        # --------------------------------------------------------------------
        def soc_evolution_rule(m, t, b):
            if t == 0:
                return Constraint.Skip  # já tratado no initial
            else:
                return m.SOC[t, b] == m.SOC[t-1, b] + eff_carga * m.CHARGE[t, b] - (m.DISCHARGE[t, b] / eff_descarga)
        model.BatterySOCEvolution = Constraint(model.T, model.BATTERIES, rule=soc_evolution_rule)

        # --------------------------------------------------------------------
        # 5. Definição da operação líquida (opcional, para extração)
        # --------------------------------------------------------------------
        # Cria uma variável para a operação líquida (descarga - carga) em cada período
        model.BatteryOperation = Var(model.T, model.BATTERIES, within=Reals)

        def battery_operation_rule(m, t, b):
            return m.BatteryOperation[t, b] == m.DISCHARGE[t, b] - m.CHARGE[t, b]
        model.BatteryOperationDef = Constraint(model.T, model.BATTERIES, rule=battery_operation_rule)