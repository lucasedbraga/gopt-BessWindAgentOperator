#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições elétricas: fluxo nas linhas, limites, balanço de potência (PyOptInterface).
"""
import numpy as np
import pyoptinterface as poi
from pyoptinterface import highs
from typing import Dict, List, Optional

class ElectricConstraints:
    """Restrições da rede elétrica para PyOptInterface."""

    @staticmethod
    def add_constraints(
        model: highs.Model,
        sistema,
        T: int,
        ANG: Dict,
        FLUXO_LIN: Dict,
        DEFICIT: Dict,
        PLOAD: np.ndarray,
        PGER: Dict,
        conv_gen_to_bar: List[int],
        PGWIND: Optional[Dict] = None,
        wind_gen_to_bar: Optional[List[int]] = None,
        CHARGE: Optional[Dict] = None,
        DISCHARGE: Optional[Dict] = None,
        battery_list: Optional[List[int]] = None,
        PERDAS_BARRA: Optional[np.ndarray] = None,
        considerar_perdas: bool = False
    ) -> List:
        """
        Adiciona restrições de fluxo DC, limites de linha e balanço de potência.
        Retorna a lista de restrições de balanço (para duais).
        """
        balance_constraints = []

        # 1. Definição do fluxo de potência (DC)
        for t in range(T):
            for e in range(sistema.NLIN):
                i = sistema.line_fr[e]
                j = sistema.line_to[e]
                x = sistema.x_line[e]
                expr = FLUXO_LIN[t, e] - (ANG[t, i] - ANG[t, j]) / x
                model.add_linear_constraint(expr == 0, name=f"flow_def_{t}_{e}")

        # 2. Limites de fluxo
        for t in range(T):
            for e in range(sistema.NLIN):
                flim = sistema.FLIM[e]
                model.add_linear_constraint(FLUXO_LIN[t, e] >= -flim, name=f"flow_lb_{t}_{e}")
                model.add_linear_constraint(FLUXO_LIN[t, e] <= flim, name=f"flow_ub_{t}_{e}")

        # 3. Limites de déficit (opcional)
        for t in range(T):
            for b in range(sistema.NBAR):
                model.add_linear_constraint(
                    DEFICIT[t, b] <= 2.0 * PLOAD[t, b],
                    name=f"deficit_limit_{t}_{b}"
                )

        # 4. Balanço de potência em cada barra
        for t in range(T):
            for b in range(sistema.NBAR):
                # Geração térmica na barra
                thermal_sum = 0
                for g in range(sistema.NGER_CONV):
                    if conv_gen_to_bar[g] == b:
                        thermal_sum += PGER[t, g]

                # Geração eólica na barra
                wind_sum = 0
                if PGWIND is not None and wind_gen_to_bar is not None:
                    for w in range(sistema.NGER_EOL):
                        if wind_gen_to_bar[w] == b:
                            wind_sum += PGWIND[t, w]

                # Bateria (descarga - carga)
                battery_net = 0
                if battery_list is not None and CHARGE is not None and DISCHARGE is not None:
                    if b in battery_list:
                        battery_net = DISCHARGE[t, b] - CHARGE[t, b]

                # Déficit
                deficit = DEFICIT[t, b]

                # Fluxo líquido entrando na barra (positivo se entra)
                flow_net = 0
                for e in range(sistema.NLIN):
                    if sistema.line_to[e] == b:
                        flow_net += FLUXO_LIN[t, e]      # fluxo chegando
                    elif sistema.line_fr[e] == b:
                        flow_net -= FLUXO_LIN[t, e]      # fluxo saindo

                # Carga
                load = PLOAD[t, b]

                # Perdas (alocadas na barra)
                losses = PERDAS_BARRA[t, b] if (considerar_perdas and PERDAS_BARRA is not None) else 0.0

                # Equação: geração + (descarga - carga) + fluxo_entrando == carga + perdas
                expr = thermal_sum + wind_sum + deficit + battery_net + flow_net - load - losses
                constr = model.add_linear_constraint(expr == 0, name=f"balance_{t}_{b}")
                balance_constraints.append((t, b, constr))

        return balance_constraints