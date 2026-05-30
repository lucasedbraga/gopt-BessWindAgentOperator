#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições elétricas AC – coordenadas polares (V, θ) para multi‑período.
Usa conjunto TIME_BUS para indexar as restrições.
"""
import numpy as np
import pyomo.environ as pyo
from typing import Dict, List, Optional, Tuple


class ACElectricConstraints:
    @staticmethod
    def add_constraints(
        model: pyo.ConcreteModel,
        sistema,
        T: int,
        G: np.ndarray,
        B: np.ndarray,
        V: dict,
        ANG: dict,
        PGER: dict,
        QGER: dict,
        PGWIND: Optional[dict],
        conv_gen_to_bar: list,
        wind_gen_to_bar: Optional[list],
        PLOAD: np.ndarray,
        QLOAD: np.ndarray,
        DEFICIT: dict,
        CHARGE: Optional[dict],
        DISCHARGE: Optional[dict],
        battery_list: Optional[list]
    ) -> List[Tuple[int, int, pyo.Constraint]]:
        n_bus = sistema.NBAR
        balance_constraints = []

        # Conjunto de índices (t, i) para todos os períodos e barras
        model.TIME_BUS = pyo.Set(initialize=[(t, i) for t in range(T) for i in range(n_bus)])

        thermal_at_bus = [[] for _ in range(n_bus)]
        for g, bus in enumerate(conv_gen_to_bar):
            thermal_at_bus[bus].append(g)

        wind_at_bus = [[] for _ in range(n_bus)]
        if PGWIND is not None and wind_gen_to_bar is not None:
            for w, bus in enumerate(wind_gen_to_bar):
                wind_at_bus[bus].append(w)

        battery_set = set(battery_list) if battery_list else set()

        def p_balance_rule(m, t, i):
            Vi = V[t, i]
            theta_i = ANG[t, i]
            sum_active = 0.0
            for j in range(n_bus):
                g_ij = G[i, j]
                b_ij = B[i, j]
                if g_ij == 0.0 and b_ij == 0.0:
                    continue
                Vj = V[t, j]
                delta = theta_i - ANG[t, j]
                sum_active += Vj * (g_ij * pyo.cos(delta) + b_ij * pyo.sin(delta))
            P_inj = Vi * sum_active

            gen_active = sum(PGER[t, g] for g in thermal_at_bus[i])
            if PGWIND is not None:
                gen_active += sum(PGWIND[t, w] for w in wind_at_bus[i])
            battery_net = 0.0
            if battery_list is not None and i in battery_set:
                battery_net = DISCHARGE[t, i] - CHARGE[t, i]
            deficit = DEFICIT[t, i]
            pload = PLOAD[t, i]

            return P_inj - (gen_active + battery_net + deficit - pload) == 0

        def q_balance_rule(m, t, i):
            Vi = V[t, i]
            theta_i = ANG[t, i]
            sum_reactive = 0.0
            for j in range(n_bus):
                g_ij = G[i, j]
                b_ij = B[i, j]
                if g_ij == 0.0 and b_ij == 0.0:
                    continue
                Vj = V[t, j]
                delta = theta_i - ANG[t, j]
                sum_reactive += Vj * (g_ij * pyo.sin(delta) - b_ij * pyo.cos(delta))
            Q_inj = Vi * sum_reactive

            gen_reactive = sum(QGER[t, g] for g in thermal_at_bus[i])
            qload = QLOAD[t, i]

            return Q_inj - (gen_reactive - qload) == 0

        model.P_balance = pyo.Constraint(model.TIME_BUS, rule=p_balance_rule)
        model.Q_balance = pyo.Constraint(model.TIME_BUS, rule=q_balance_rule)

        # Retornar referências para uso externo (ex.: perdas iterativas, embora não usadas aqui)
        for t in range(T):
            for i in range(n_bus):
                balance_constraints.append((t, i, model.P_balance[t, i]))

        return balance_constraints