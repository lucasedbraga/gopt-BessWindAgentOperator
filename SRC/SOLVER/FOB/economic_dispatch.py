from pyomo.environ import *
import numpy as np
import time
from typing import Dict, Optional
from datetime import datetime

from OPT.DC_OPF_Model import DC_OPF_Model, OPFResult

class DC_OPF_EconomicDispatch_Solver:
    """Solver genérico para OPF com nova estrutura de dados (convencionais e eólicos separados)"""
    
    def __init__(self, sistema):
        """
        Args:
            sistema: Objeto com dados do sistema processados
                     (deve ter os atributos definidos em SistemaLoader)
        """
        self.sistema = sistema
        
    def build_model(self, considerar_perdas=False) -> DC_OPF_Model:
        """Constrói modelo físico"""
        modelo = DC_OPF_Model(considerar_perdas=considerar_perdas)
        modelo.build(self.sistema, considerar_perdas)
        return modelo
    
    def add_objective(self, modelo: DC_OPF_Model):
        """Adiciona função objetivo ao modelo, usando custos específicos para convencionais e eólicos"""
        m = modelo.model
        s = self.sistema

        # Custo de geração convencional
        custo_geracao = 0.0
        if hasattr(m, 'CONV_GENERATORS') and len(m.CONV_GENERATORS) > 0:
            custo_geracao = sum(
                m.PGER[g] * s.CPG_CONV[g]
                for g in m.CONV_GENERATORS
            )

        # Custo de curtailment (penalidade por eólico)
        custo_curtailment = 0.0
        if hasattr(m, 'WIND_GENERATORS') and len(m.WIND_GENERATORS) > 0:
            custo_curtailment = sum(
                m.CURTAILMENT[g] * s.CPG_CURTAILMENT[g]
                for g in m.WIND_GENERATORS
            )

        # Custo de déficit (penalidade muito alta, pode ser um vetor por barra, mas aqui usamos escalar)
        custo_deficit = 0.0
        if hasattr(m, 'DEFICIT'):
            custo_deficit = sum(
                m.DEFICIT[b] * s.CPG_DEFICIT
                for b in m.BUSES
            )

        # Custo de operação das baterias (se houver)
        custo_bateria = 0.0
        if hasattr(m, 'BATTERIES') and len(m.BATTERIES) > 0:
            # Exemplo: custo associado à carga/descarga, pode ser negativo se for benefício
            # Usando BATTERY_COST por barra (se definido) ou um valor padrão
            for b in m.BATTERIES:
                # Custo pode ser modelado como penalidade por uso: ex: 10 * (CHARGE + DISCHARGE)
                # Ou benefício por descarga: -benef * DISCHARGE
                # Ajuste conforme seu modelo econômico
                custo_bateria += m.CHARGE[b] * s.BATTERY_COST[b] - m.DISCHARGE[b] * s.BATTERY_COST[b]  # Exemplo simples
                # Se preferir apenas penalidade: custo_bateria += (m.CHARGE[b] + m.DISCHARGE[b]) * s.BATTERY_COST[b]

        # Função objetivo total
        m.obj = Objective(
            expr=custo_geracao + custo_curtailment + custo_deficit + custo_bateria,
            sense=minimize
        )

        return m.obj