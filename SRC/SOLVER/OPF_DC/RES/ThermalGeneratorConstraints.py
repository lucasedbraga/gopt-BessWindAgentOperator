#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restrições para geradores térmicos em modelo multi-período (PyOptInterface).
Todos os valores de potência em pu (por unidade).
"""
from __future__ import annotations
import numpy as np
import pyoptinterface as poi
from pyoptinterface import highs
from typing import Dict, List, Tuple


class ThermalGeneratorConstraints:
    """
    Restrições de geradores térmicos: limites de geração e rampas.
    """

    @staticmethod
    def add_constraints(
        model: highs.Model,
        T: int,
        NGER_CONV: int,
        PGER: Dict[Tuple[int, int], poi.Variable],
        pgmin_conv: List[float],
        pgmax_conv: List[float],
        pger_inicial_conv: List[float],
        ramp_up_mw: List[float],
        ramp_down_mw: List[float],
        SB: float
    ) -> None:
        """
        Adiciona restrições de limites e rampa para geradores térmicos.

        Parâmetros:
        -----------
        model : highs.Model
            Modelo PyOptInterface.
        T : int
            Número de períodos.
        NGER_CONV : int
            Número de geradores convencionais.
        PGER : dict
            Dicionário de variáveis de geração térmica, chave (t, g) (valor em pu).
        pgmin_conv : list
            Limite mínimo de geração (pu) para cada gerador.
        pgmax_conv : list
            Limite máximo de geração (pu) para cada gerador.
        pger_inicial_conv : list
            Geração inicial (pu) antes do primeiro período.
        ramp_up_mw : list
            Rampa de subida (MW/h) para cada gerador.
        ramp_down_mw : list
            Rampa de descida (MW/h) para cada gerador.
        SB : float
            Potência base (MVA) para conversão de MW para pu.
        """
        if NGER_CONV == 0:
            return

        # Converte rampas de MW/h para pu/h (1 período = 1h)
        ramp_up_pu = [r / SB for r in ramp_up_mw]
        ramp_down_pu = [r / SB for r in ramp_down_mw]

        # Limites de geração (explícitos, embora já estejam nos bounds)
        for t in range(T):
            for g in range(NGER_CONV):
                model.add_linear_constraint(
                    PGER[t, g] >= pgmin_conv[g],
                    name=f"gen_lb_{t}_{g}"
                )
                model.add_linear_constraint(
                    PGER[t, g] <= pgmax_conv[g],
                    name=f"gen_ub_{t}_{g}"
                )

        # Primeiro período: comparar com geração inicial
        for g in range(NGER_CONV):
            model.add_linear_constraint(
                PGER[0, g] <= pger_inicial_conv[g] + ramp_up_pu[g],
                name=f"first_ramp_up_{g}"
            )
            model.add_linear_constraint(
                PGER[0, g] >= pger_inicial_conv[g] - ramp_down_pu[g],
                name=f"first_ramp_down_{g}"
            )

        # Períodos seguintes: comparar com período anterior
        for t in range(1, T):
            for g in range(NGER_CONV):
                model.add_linear_constraint(
                    PGER[t, g] <= PGER[t-1, g] + ramp_up_pu[g],
                    name=f"ramp_up_{t}_{g}"
                )
                model.add_linear_constraint(
                    PGER[t, g] >= PGER[t-1, g] - ramp_down_pu[g],
                    name=f"ramp_down_{t}_{g}"
                )