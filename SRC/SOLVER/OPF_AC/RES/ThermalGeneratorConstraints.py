#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições para geradores térmicos – versão Pyomo.
Inclui limites de potência ativa e reativa, e rampas de geração ativa.
"""
from __future__ import annotations
import numpy as np
import pyomo.environ as pyo
from typing import Dict, List, Tuple, Optional


class ThermalGeneratorConstraints:
    """
    Restrições de geradores térmicos para modelos Pyomo.
    """

    @staticmethod
    def add_constraints(
        model: pyo.ConcreteModel,
        T: int,
        NGER_UTE: int,
        PGER: Dict[Tuple[int, int], pyo.Var],
        QGER: Optional[Dict[Tuple[int, int], pyo.Var]] = None,
        PGER_MIN_UTE: List[float] = None,
        PGER_MAX_UTE: List[float] = None,
        qgmin_conv: Optional[List[float]] = None,
        qgmax_conv: Optional[List[float]] = None,
        PGER_inicial_UTE: Optional[List[float]] = None,
        ramp_up_mw: Optional[List[float]] = None,
        ramp_down_mw: Optional[List[float]] = None,
        SB: float = 100.0
    ) -> None:
        """
        Adiciona limites de geração ativa/reativa e restrições de rampa.

        Parâmetros
        ----------
        model : pyo.ConcreteModel
            Modelo Pyomo.
        T : int
            Número de períodos.
        NGER_UTE : int
            Número de geradores convencionais.
        PGER : dict
            Variáveis de potência ativa (pyo.Var), chave (t, g).
        QGER : dict, opcional
            Variáveis de potência reativa (pyo.Var), chave (t, g).
        PGER_MIN_UTE, PGER_MAX_UTE : list
            Limites de potência ativa (pu).
        qgmin_conv, qgmax_conv : list, opcional
            Limites de potência reativa (pu).
        PGER_inicial_UTE : list, opcional
            Geração ativa inicial (antes do período 0) para rampa.
        ramp_up_mw, ramp_down_mw : list, opcional
            Taxas de rampa em MW/h (convertidas para pu/h com SB).
        SB : float
            Potência base (MVA).
        """
        if NGER_UTE == 0:
            return

        if PGER_MIN_UTE is None:
            PGER_MIN_UTE = [0.0] * NGER_UTE
        if PGER_MAX_UTE is None:
            PGER_MAX_UTE = [1.0] * NGER_UTE

        # Limites de potência ativa
        for t in range(T):
            for g in range(NGER_UTE):
                setattr(model, f"gen_active_lb_{t}_{g}",
                        pyo.Constraint(expr=PGER[t, g] >= PGER_MIN_UTE[g]))
                setattr(model, f"gen_active_ub_{t}_{g}",
                        pyo.Constraint(expr=PGER[t, g] <= PGER_MAX_UTE[g]))

        # Limites de potência reativa
        if QGER is not None and qgmin_conv is not None and qgmax_conv is not None:
            for t in range(T):
                for g in range(NGER_UTE):
                    setattr(model, f"gen_reactive_lb_{t}_{g}",
                            pyo.Constraint(expr=QGER[t, g] >= qgmin_conv[g]))
                    setattr(model, f"gen_reactive_ub_{t}_{g}",
                            pyo.Constraint(expr=QGER[t, g] <= qgmax_conv[g]))

        # # Restrições de rampa
        # if ramp_up_mw is not None and ramp_down_mw is not None and PGER_inicial_UTE is not None:
        #     ramp_up_pu = [r / SB for r in ramp_up_mw]
        #     ramp_down_pu = [r / SB for r in ramp_down_mw]

        #     # Primeiro período
        #     for g in range(NGER_UTE):
        #         setattr(model, f"first_ramp_up_{g}",
        #                 pyo.Constraint(expr=PGER[0, g] <= PGER_inicial_UTE[g] + ramp_up_pu[g]))
        #         setattr(model, f"first_ramp_down_{g}",
        #                 pyo.Constraint(expr=PGER[0, g] >= PGER_inicial_UTE[g] - ramp_down_pu[g]))

        #     # Demais períodos
        #     for t in range(1, T):
        #         for g in range(NGER_UTE):
        #             setattr(model, f"ramp_up_{t}_{g}",
        #                     pyo.Constraint(expr=PGER[t, g] <= PGER[t-1, g] + ramp_up_pu[g]))
        #             setattr(model, f"ramp_down_{t}_{g}",
        #                     pyo.Constraint(expr=PGER[t, g] >= PGER[t-1, g] - ramp_down_pu[g]))