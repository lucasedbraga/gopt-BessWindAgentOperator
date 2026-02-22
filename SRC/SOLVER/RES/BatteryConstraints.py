from pyomo.environ import *
import numpy as np

class BatteryConstraints:
    """Restrições de Bateria"""
    @staticmethod
    def add_battery_constraints(model, sistema):
        """
        Adiciona restrições para baterias usando apenas variáveis contínuas.
        
        Parâmetros:
        - model: modelo Pyomo
        - sistema: objeto com parâmetros (capacidade, limites de potência, eficiências, SOC inicial e final)
        - is_last_period: bool, indica se o período atual é o último da simulação
        """
        # Verificar se há baterias no sistema
        if not hasattr(model, 'BATTERIES') or len(model.BATTERIES) == 0:
            return
        
        # Variáveis de bateria (apenas se houver baterias)
        if len(model.BATTERIES) > 0:
            model.CHARGE = Var(model.BATTERIES, within=NonNegativeReals)  # Carga
            model.DISCHARGE = Var(model.BATTERIES, within=NonNegativeReals)  # Descarga
            model.SOC_init = Var(model.BATTERIES, within=NonNegativeReals)  # Estado de carga anterior
            model.SOC_atual = Var(model.BATTERIES, within=NonNegativeReals)  # Estado de carga atual

        # Verificar se as variáveis necessárias existem no modelo
        if not hasattr(model, 'CHARGE') or not hasattr(model, 'DISCHARGE') or \
        not hasattr(model, 'SOC_atual') or not hasattr(model, 'SOC_init'):
            print("  Variáveis de bateria não definidas - pulando restrições")
            return
        
        # Eficiências (podem ser parametrizadas no sistema)
        eff_carga = getattr(sistema, 'BATTERY_CHARGE_EFF', 1)
        eff_descarga = getattr(sistema, 'BATTERY_DISCHARGE_EFF', 1)
        
        # ------------------------------------------------------------------------
        # 1. Limites operacionais da bateria (SOC mínimo e máximo)
        # ------------------------------------------------------------------------
        def soc_max_rule(m, b):
            return m.SOC_atual[b] <= sistema.BATTERY_CAPACITY[b]
        model.BatterySOCMax = Constraint(model.BATTERIES, rule=soc_max_rule)
        
        def soc_min_rule(m, b):
            return m.SOC_atual[b] >= sistema.BATTERY_MIN_SOC[b]
        model.BatterySOCMin = Constraint(model.BATTERIES, rule=soc_min_rule)
        
        # ------------------------------------------------------------------------
        # 2. Condição inicial (SOC do período anterior)
        # ------------------------------------------------------------------------
        def soc_initial_rule(m):
            # Fixar SOC inicial das baterias 
            for b in m.BATTERIES:
                m.SOC_init[b].fix((sistema.SOC_init)[b])
        soc_initial_rule(model)  # Chamar a função para fixar os valores iniciais

        # ------------------------------------------------------------------------
        # 3. Balanço de energia (atualização do SOC)
        # ------------------------------------------------------------------------
        def soc_update_rule(m, b):
            # SOC final = SOC inicial + (carga * eficiência) - (descarga / eficiência_descarga)
            return m.SOC_atual[b] == m.SOC_init[b] + eff_carga * m.CHARGE[b] - (m.DISCHARGE[b] / eff_descarga)
        model.BatterySOCUpdate = Constraint(model.BATTERIES, rule=soc_update_rule)
        
        # ------------------------------------------------------------------------
        # Limites de potência de carga e descarga
        # ------------------------------------------------------------------------
        def charge_limit_rule(m, b):
            return m.CHARGE[b] <= sistema.BATTERY_POWER_LIMIT[b]
        model.BatteryChargeLimit = Constraint(model.BATTERIES, rule=charge_limit_rule)
        
        def discharge_limit_rule(m, b):
            return m.DISCHARGE[b] <= sistema.BATTERY_POWER_OUT[b]
        model.BatteryDischargeLimit = Constraint(model.BATTERIES, rule=discharge_limit_rule)