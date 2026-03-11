#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições para baterias em modelo multi‑período (com índice temporal) – PyOptInterface.
"""
import numpy as np
import pyoptinterface as poi
from pyoptinterface import highs

class BatteryConstraintsTime:
    """Restrições de bateria para modelos com horizonte de tempo (PyOptInterface)."""

    @staticmethod
    def add_constraints(model, sistema, T, battery_list, battery_index,
                        CHARGE, DISCHARGE, SOC, BatteryOperation,
                        soc_inicial_list, soc_final_list=None):
        """
        Adiciona variáveis e restrições de bateria ao modelo PyOptInterface.

        Parâmetros:
        -----------
        model : highs.Model
            Modelo PyOptInterface.
        sistema : SistemaLoader
            Objeto contendo dados do sistema (BATTERY_CAPACITY, BATTERY_MIN_SOC, etc.)
        T : int
            Número de períodos.
        battery_list : list
            Lista de índices das barras que possuem bateria.
        battery_index : dict
            Mapeamento de barra -> índice na lista (opcional, usado para SOC inicial).
        CHARGE, DISCHARGE, SOC, BatteryOperation : dict
            Dicionários que serão preenchidos com as variáveis criadas.
            As chaves serão (t, b) para t em range(T) e b em battery_list.
        soc_inicial_list : list
            Lista com os valores iniciais de SOC (MWh) para cada bateria na ordem de battery_list.
        soc_final_list : list, optional
            Lista com os valores finais desejados de SOC (MWh) para cada bateria.
            Se fornecida, adiciona restrição de SOC final.
        """
        if not battery_list:
            return

        # Eficiências (valores padrão)
        eff_carga = getattr(sistema, 'BATTERY_CHARGE_EFF', 1)
        eff_descarga = getattr(sistema, 'BATTERY_DISCHARGE_EFF', 1)

        # Função auxiliar para obter parâmetro por barra
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
        # 1. Criar variáveis
        # --------------------------------------------------------------------
        for t in range(T):
            for b in battery_list:
                # Capacidade e limites
                cap = get_val('BATTERY_CAPACITY', b, 1.0)          # MWh
                min_soc = get_val('BATTERY_MIN_SOC', b, 0.1)      # pu da capacidade
                min_soc_abs = min_soc * cap                        # MWh
                power_limit = get_val('BATTERY_POWER_LIMIT', b, cap)  # MW
                power_out = get_val('BATTERY_POWER_OUT', b, cap)      # MW

                SOC[t, b] = model.add_variable(
                    lb=min_soc_abs,
                    ub=cap,
                    name=f"SOC_{t}_{b}"
                )
                CHARGE[t, b] = model.add_variable(
                    lb=0,
                    ub=power_limit,
                    name=f"CHARGE_{t}_{b}"
                )
                DISCHARGE[t, b] = model.add_variable(
                    lb=0,
                    ub=power_out,
                    name=f"DISCHARGE_{t}_{b}"
                )
                BatteryOperation[t, b] = model.add_variable(
                    lb=-power_out,      # pode ser negativa (carga líquida)
                    ub=power_out,
                    name=f"BatteryOperation_{t}_{b}"
                )

        # --------------------------------------------------------------------
        # 2. Restrições
        # --------------------------------------------------------------------
        # 2a. Limites de SOC (já definidos nos bounds, mas podemos reforçar se necessário)

        # 2b. Limites de potência já estão nos bounds, não precisam de restrição extra.

        # 2c. Condição inicial
        for i, b in enumerate(battery_list):
            model.add_linear_constraint(
                SOC[0, b] == soc_inicial_list[i] + eff_carga * CHARGE[t, b] -
                    DISCHARGE[t, b] / eff_descarga,
                name=f"SOC_init_{b}"
            )

        # 2d. Evolução do SOC
        for t in range(1, T):
            for b in battery_list:
                expr = SOC[t, b] - (
                    SOC[t-1, b] +
                    eff_carga * CHARGE[t, b] -
                    DISCHARGE[t, b] / eff_descarga
                )
                model.add_linear_constraint(expr == 0, name=f"SOC_evolution_{t}_{b}")

        # 2e. Definição da operação líquida (BatteryOperation = DISCHARGE - CHARGE)
        for t in range(T):
            for b in battery_list:
                model.add_linear_constraint(
                    BatteryOperation[t, b] == DISCHARGE[t, b] - CHARGE[t, b],
                    name=f"BatteryOperation_def_{t}_{b}"
                )

        # 2f. SOC final (opcional)
        if soc_final_list is not None:
            for i, b in enumerate(battery_list):
                model.add_linear_constraint(
                    SOC[T-1, b] == soc_final_list[i],
                    name=f"SOC_final_{b}"
                )