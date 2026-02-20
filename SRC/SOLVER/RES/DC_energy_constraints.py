from pyomo.environ import *
import numpy as np

class DCEnergyConstraints:
    """Restrições energéticas"""
    
    @staticmethod
    def add_deficit_constraints(model, sistema):
        """Adiciona restrições para déficit"""
        def deficit_limits_rule(m, i):
            # Déficit limitado pela carga da barra
            carga_barra = sistema.PLOAD[i]
            return m.DEFICIT[i] <= carga_barra * 2.0
        
        model.DeficittLimits = Constraint(model.BUSES, rule=deficit_limits_rule)
    
    @staticmethod
    def add_wind_generator_constraints(model, sistema):
        """Adiciona restrições para geradores eólicos"""
        # Geração utilizada não pode exceder disponível
        def wind_balance_rule(m, g):
            return m.PG_WIND_USED[g] <= m.PG[g]
        
        model.WindBalance = Constraint(model.GWD_GENERATORS, rule=wind_balance_rule)
        
        # Definição de curtailment
        def curtailment_definition_rule(m, g):
            return m.CURTAILMENT[g] == m.PG[g] - m.PG_WIND_USED[g]
        
        model.CurtailmentDefinition = Constraint(model.GWD_GENERATORS, rule=curtailment_definition_rule)
    
    @staticmethod
    def add_battery_constraints(model, sistema, is_last_period=False):
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
        
        # Verificar se as variáveis necessárias existem no modelo
        if not hasattr(model, 'CHARGE') or not hasattr(model, 'DISCHARGE') or \
        not hasattr(model, 'SOC') or not hasattr(model, 'SOC_INI'):
            print("  Variáveis de bateria não definidas - pulando restrições")
            return
        
        # Eficiências (podem ser parametrizadas no sistema)
        eff_carga = getattr(sistema, 'BATTERY_CHARGE_EFF', 0.95)
        eff_descarga = getattr(sistema, 'BATTERY_DISCHARGE_EFF', 0.95)
        
        # ------------------------------------------------------------------------
        # 1. Limites operacionais da bateria (SOC mínimo e máximo)
        # ------------------------------------------------------------------------
        def soc_max_rule(m, b):
            return m.SOC[b] <= sistema.BATTERY_CAPACITY[b]
        model.BatterySOCMax = Constraint(model.BATTERIES, rule=soc_max_rule)
        
        def soc_min_rule(m, b):
            return m.SOC[b] >= 0.1*sistema.BATTERY_CAPACITY[b]
        model.BatterySOCMin = Constraint(model.BATTERIES, rule=soc_min_rule)
        
        # ------------------------------------------------------------------------
        # 2. Condição inicial (SOC do período anterior)
        # ------------------------------------------------------------------------
        # def soc_initial_rule(m, b):
        #     return m.SOC_PREV[b] == sistema.BATTERY_INITIAL_SOC[b]
        # model.BatterySOCInitial = Constraint(model.BATTERIES, rule=soc_initial_rule)
        
        # ------------------------------------------------------------------------
        # 3. Balanço de energia (atualização do SOC)
        # ------------------------------------------------------------------------
        def soc_update_rule(m, b):
            # SOC final = SOC inicial + (carga * eficiência) - (descarga / eficiência_descarga)
            return m.SOC[b] == m.SOC_INI[b] + eff_carga * m.CHARGE[b] - (m.DISCHARGE[b] / eff_descarga)
        model.BatterySOCUpdate = Constraint(model.BATTERIES, rule=soc_update_rule)
        
        # ------------------------------------------------------------------------
        # 4. Limites de potência de carga e descarga (sem binárias)
        # ------------------------------------------------------------------------
        def charge_limit_rule(m, b):
            return m.CHARGE[b] <= sistema.BATTERY_POWER_LIMIT[b]   # ou BATTERY_CHARGE_LIMIT
        model.BatteryChargeLimit = Constraint(model.BATTERIES, rule=charge_limit_rule)
        
        def discharge_limit_rule(m, b):
            return m.DISCHARGE[b] <= sistema.BATTERY_POWER_OUT[b]  # ou BATTERY_DISCHARGE_LIMIT
        model.BatteryDischargeLimit = Constraint(model.BATTERIES, rule=discharge_limit_rule)
        
        # ------------------------------------------------------------------------
        # 5. Condição de último período (SOC final = parâmetro alvo)
        # ------------------------------------------------------------------------
        if is_last_period:
            # Certificar que o sistema possui o parâmetro BATTERY_FINAL_SOC
            if not hasattr(sistema, 'BATTERY_FINAL_SOC'):
                raise AttributeError("Parâmetro BATTERY_FINAL_SOC não definido no sistema para o último período.")
            
            def final_soc_rule(m, b):
                return m.SOC[b] == sistema.BATTERY_FINAL_SOC[b]
            model.BatteryFinalSOC = Constraint(model.BATTERIES, rule=final_soc_rule)