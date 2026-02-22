from pyomo.environ import *
import numpy as np
import time
from typing import Dict, Optional
from datetime import datetime

from OPT.DC_OPF_Model import DC_OPF_Model, OPFResult

class DC_OPF_EconomicDispatch_Solver:
    """Solver genérico para OPF"""
    
    def __init__(self, sistema):
        """
        Args:
            sistema: Objeto com dados do sistema processados
                     (deve ter os atributos definidos em SistemaLoader)
        """
        self.sistema = sistema
        
    def build_model(self, considerar_perdas=False) -> DC_OPF_Model:
        """Constrói modelo físico"""
        modelo = DC_OPF_Model()
        modelo.build(self.sistema, considerar_perdas)
        return modelo
    
    def add_objective(self, modelo: DC_OPF_Model):
        """Adiciona função objetivo ao modelo"""
        m = modelo.model
        
        # Custo de geração convencional
        custo_geracao = sum(
            m.PGER[g] * self.sistema.CPG[g] 
            for g in m.GENERATORS 
            if g not in m.GWD_GENERATORS
        )
        
        # Custo de curtailment (penalidade alta)
        custo_curtailment = sum(
            m.CURTAILMENT[g] * 1000.0
            for g in m.GWD_GENERATORS
        )
        
        # Custo de déficit (penalidade muito alta)
        custo_deficit = sum(
            m.DEFICIT[b] * 5000.0
            for b in m.BUSES
        )
        
        # Custo de operação das baterias (se houver)
        custo_bateria = 0.0
        if hasattr(m, 'BATTERIES') and len(m.BATTERIES) > 0:
            custo_bateria = sum(
                (-0.1*m.CHARGE[b] + 0.5*m.DISCHARGE[b]) #* getattr(self.sistema, 'BATTERY_COST', [0])[b]
                for b in m.BATTERIES
            )
        
        # Função objetivo total
        m.obj = Objective(
            expr=custo_geracao + custo_curtailment + custo_deficit + custo_bateria, 
            sense=minimize
        )
        
        return m.obj