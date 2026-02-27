#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de otimização multi‑período acoplado (30 dias) com baterias.
Executa uma única otimização para todos os períodos e salva os resultados no banco SQLite.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pyomo.environ import *

# Adiciona o diretório SRC ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Union
from DB.OPF_Snapshot_DBmodel import *


# =============================================================================
# Classe principal do modelo integrado (acoplado no tempo)
# =============================================================================
class MultiDayOPFModelIntegrated:
    """
    Modelo de otimização multi‑período com acoplamento temporal integrado.
    As variáveis são indexadas no tempo (períodos = n_dias * n_horas).
    O SOC inicial e final podem ser escalares (para todas as baterias) ou listas individuais.
    """

    def __init__(self, sistema, n_horas=24, n_dias=1, db_handler=None):
        self.sistema = sistema
        self.n_horas = n_horas
        self.n_dias = n_dias
        self.T = n_horas * n_dias
        self.db_handler = db_handler
        self.model = None
        self._solved = False
        self._raw_results = None
        self._battery_map = {}  # mapeia barra -> índice na lista de baterias

    def _map_battery_data(self, attr_name, default=None):
        """
        Retorna um dicionário {barra: valor} para um atributo do sistema.
        Se o atributo for uma lista de tamanho NBAR, usa diretamente.
        Se for uma lista de tamanho igual ao número de baterias, mapeia na ordem de BARRAS_COM_BATERIA.
        """
        s = self.sistema
        if not hasattr(s, 'BARRAS_COM_BATERIA') or not hasattr(s, attr_name):
            return {}
        valores = getattr(s, attr_name)
        if len(valores) == s.NBAR:
            # indexado por barra
            return {b: valores[b] for b in s.BARRAS_COM_BATERIA}
        else:
            # assume que está na mesma ordem de BARRAS_COM_BATERIA
            return {b: valores[i] for i, b in enumerate(s.BARRAS_COM_BATERIA)}

    def build(self, fator_carga: np.ndarray = None, fator_vento: np.ndarray = None,
              soc_inicial: Union[float, List[float]] = 0.5,
              soc_final: Union[float, List[float]] = 0.5):
        """
        Constrói o modelo monolítico com variáveis temporais.

        Args:
            fator_carga: array (T, NBAR) ou (dias, horas, NBAR) com fatores de carga.
            fator_vento: array (T, NGER_EOL) com fatores eólicos (cada coluna um gerador).
            soc_inicial: escalar (mesmo valor para todas) ou lista com valores por bateria.
            soc_final: escalar ou lista com valores por bateria.
        """
        s = self.sistema
        T = self.T

        # Processa fator_carga para shape (T, NBAR)
        if fator_carga is not None:
            if fator_carga.ndim == 2 and fator_carga.shape[0] == self.n_dias and fator_carga.shape[1] == self.n_horas:
                # (dias, horas) -> expandir para (T, NBAR) repetindo o mesmo fator para todas as barras
                fator_carga = np.repeat(fator_carga.reshape((T, 1)), s.NBAR, axis=1)
            elif fator_carga.ndim == 3:
                fator_carga = fator_carga.reshape((T, -1))
            elif fator_carga.ndim == 2 and fator_carga.shape[1] != s.NBAR:
                # se veio com shape (T, algo) mas não NBAR, assume que é (T, 1) e expande
                fator_carga = np.repeat(fator_carga, s.NBAR, axis=1)
        else:
            fator_carga = np.ones((T, s.NBAR))

        # Processa fator_vento para shape (T, NGER_EOL)
        if fator_vento is not None:
            if fator_vento.ndim == 1:
                # (T,) -> expandir para (T, NGER_EOL) repetindo a mesma coluna
                fator_vento = np.repeat(fator_vento.reshape(-1, 1), s.NGER_EOL, axis=1)
            elif fator_vento.ndim == 2 and fator_vento.shape[0] == self.n_dias and fator_vento.shape[1] == self.n_horas:
                # (dias, horas) -> expandir para (T, NGER_EOL) repetindo
                fator_vento = np.repeat(fator_vento.reshape((T, 1)), s.NGER_EOL, axis=1)
            elif fator_vento.ndim == 3:
                fator_vento = fator_vento.reshape((T, -1))
            # caso contrário, assume que já está no formato correto (T, NGER_EOL)
        else:
            fator_vento = np.ones((T, s.NGER_EOL)) if s.NGER_EOL > 0 else np.ones((T, 0))

        # Verifica se há baterias
        tem_bateria = hasattr(s, 'BARRAS_COM_BATERIA') and len(s.BARRAS_COM_BATERIA) > 0
        if tem_bateria:
            self._battery_list = list(s.BARRAS_COM_BATERIA)
            self._battery_index = {b: i for i, b in enumerate(self._battery_list)}
            # Mapeia dados das baterias
            self.capacidade = self._map_battery_data('BATTERY_CAPACITY', 0)
            self.min_soc = self._map_battery_data('BATTERY_MIN_SOC', 0)
            self.power_charge = self._map_battery_data('BATTERY_POWER_LIMIT', 0)
            self.power_discharge = self._map_battery_data('BATTERY_POWER_OUT', 0)
            self.custo_bateria = self._map_battery_data('BATTERY_COST', 0)

            # Trata soc_inicial e soc_final
            if isinstance(soc_inicial, (int, float)):
                soc_inicial = [soc_inicial] * len(self._battery_list)
            if isinstance(soc_final, (int, float)):
                soc_final = [soc_final] * len(self._battery_list)

        self.model = ConcreteModel()
        m = self.model

        # Conjuntos
        m.T = RangeSet(0, T-1)
        m.BUSES = Set(initialize=range(s.NBAR))
        m.CONV_GENERATORS = Set(initialize=range(s.NGER_CONV))
        m.LINES = Set(initialize=range(s.NLIN))
        if s.NGER_EOL > 0:
            m.WIND_GENERATORS = Set(initialize=range(s.NGER_EOL))
        else:
            m.WIND_GENERATORS = Set(initialize=[])
        if tem_bateria:
            m.BATTERIES = Set(initialize=self._battery_list)
        else:
            m.BATTERIES = Set(initialize=[])

        # Parâmetros temporais
        m.PLOAD = Param(m.T, m.BUSES, mutable=True, initialize=0.0)
        if s.NGER_EOL > 0:
            m.PGWIND_AVAIL = Param(m.T, m.WIND_GENERATORS, mutable=True, initialize=0.0)

        # Preenche parâmetros
        for t in m.T:
            for b in m.BUSES:
                m.PLOAD[t, b] = s.PLOAD[b] * fator_carga[t, b]
            if s.NGER_EOL > 0:
                for w in m.WIND_GENERATORS:
                    if hasattr(s, 'PGMAX_EFETIVO_EOL'):
                        m.PGWIND_AVAIL[t, w] = s.PGMAX_EFETIVO_EOL[w] * fator_vento[t, w]
                    elif hasattr(s, 'PGWIND_disponivel'):
                        m.PGWIND_AVAIL[t, w] = s.PGWIND_disponivel[w] * fator_vento[t, w]
                    else:
                        m.PGWIND_AVAIL[t, w] = 0.0

        # Variáveis
        m.PGER = Var(m.T, m.CONV_GENERATORS, within=NonNegativeReals)
        if s.NGER_EOL > 0:
            m.PGWIND = Var(m.T, m.WIND_GENERATORS, within=NonNegativeReals)
            m.CURTAILMENT = Var(m.T, m.WIND_GENERATORS, within=NonNegativeReals)
        m.DEFICIT = Var(m.T, m.BUSES, within=NonNegativeReals)
        m.V = Var(m.T, m.BUSES, within=NonNegativeReals, bounds=(0.95, 1.05))
        m.ANG = Var(m.T, m.BUSES, within=Reals, bounds=(-3.14, 3.14))
        m.FLUXO_LIN = Var(m.T, m.LINES, within=Reals)

        if tem_bateria:
            m.SOC = Var(m.T, m.BATTERIES, within=NonNegativeReals)
            m.CHARGE = Var(m.T, m.BATTERIES, within=NonNegativeReals)
            m.DISCHARGE = Var(m.T, m.BATTERIES, within=NonNegativeReals)

        # Fixações iniciais
        for t in m.T:
            for b in m.BUSES:
                m.V[t, b].fix(1.0)
            m.ANG[t, s.slack_idx].fix(0.0)

        # ========== Restrições ==========
        # Eólicas
        if s.NGER_EOL > 0:
            def wind_balance_rule(m, t, w):
                return m.PGWIND[t, w] + m.CURTAILMENT[t, w] == m.PGWIND_AVAIL[t, w]
            m.WindBalance = Constraint(m.T, m.WIND_GENERATORS, rule=wind_balance_rule)

        # Limites de geração convencional
        def gen_limits_rule(m, t, g):
            return inequality(s.PGMIN_CONV[g], m.PGER[t, g], s.PGMAX_CONV[g])
        m.GenLimits = Constraint(m.T, m.CONV_GENERATORS, rule=gen_limits_rule)

        # Fluxo nas linhas
        def flow_def_rule(m, t, e):
            i = s.line_fr[e]
            j = s.line_to[e]
            return m.FLUXO_LIN[t, e] == (m.ANG[t, i] - m.ANG[t, j]) / s.x_line[e]
        m.FlowDef = Constraint(m.T, m.LINES, rule=flow_def_rule)

        def flow_limits_rule(m, t, e):
            return inequality(-s.FLIM[e], m.FLUXO_LIN[t, e], s.FLIM[e])
        m.FlowLimits = Constraint(m.T, m.LINES, rule=flow_limits_rule)

        # Déficit (limite opcional)
        def deficit_limits_rule(m, t, b):
            return m.DEFICIT[t, b] <= 2 * m.PLOAD[t, b]
        m.DeficitLimits = Constraint(m.T, m.BUSES, rule=deficit_limits_rule)

        # Balanço de potência
        def power_balance_rule(m, t, b):
            ger_conv = sum(m.PGER[t, g] for g in m.CONV_GENERATORS if s.BARPG_CONV[g] == b)
            ger_eol = 0
            if s.NGER_EOL > 0:
                ger_eol = sum(m.PGWIND[t, w] for w in m.WIND_GENERATORS if s.BARPG_EOL[w] == b)
            deficit = m.DEFICIT[t, b]
            bateria = 0
            if tem_bateria and b in m.BATTERIES:
                bateria = m.DISCHARGE[t, b] - m.CHARGE[t, b]
            fluxo_liquido = 0
            for e in m.LINES:
                if s.line_fr[e] == b:
                    fluxo_liquido += m.FLUXO_LIN[t, e]
                elif s.line_to[e] == b:
                    fluxo_liquido -= m.FLUXO_LIN[t, e]
            carga = m.PLOAD[t, b]
            return ger_conv + ger_eol + deficit + bateria - fluxo_liquido == carga
        m.PowerBalance = Constraint(m.T, m.BUSES, rule=power_balance_rule)

        # Restrições de bateria
        if tem_bateria:
            eff_carga = getattr(s, 'BATTERY_CHARGE_EFF', 0.9)
            eff_descarga = getattr(s, 'BATTERY_DISCHARGE_EFF', 0.9)

            # Limites de SOC
            def soc_max_rule(m, t, b):
                return m.SOC[t, b] <= self.capacidade[b]
            m.SOCMax = Constraint(m.T, m.BATTERIES, rule=soc_max_rule)

            def soc_min_rule(m, t, b):
                return m.SOC[t, b] >= self.min_soc[b]
            m.SOCMin = Constraint(m.T, m.BATTERIES, rule=soc_min_rule)

            # Limites de carga/descarga
            def charge_limit_rule(m, t, b):
                return m.CHARGE[t, b] <= self.power_charge[b]
            m.ChargeLimit = Constraint(m.T, m.BATTERIES, rule=charge_limit_rule)

            def discharge_limit_rule(m, t, b):
                return m.DISCHARGE[t, b] <= self.power_discharge[b]
            m.DischargeLimit = Constraint(m.T, m.BATTERIES, rule=discharge_limit_rule)

            # Evolução do SOC
            def soc_evolution_rule(m, t, b):
                idx = self._battery_index[b]
                if t == 0:
                    return m.SOC[t, b] == soc_inicial[idx]
                else:
                    return m.SOC[t, b] == m.SOC[t-1, b] + eff_carga * m.CHARGE[t, b] - (m.DISCHARGE[t, b] / eff_descarga)
            m.SOCEvolution = Constraint(m.T, m.BATTERIES, rule=soc_evolution_rule)

            # Condição final
            def soc_final_rule(m, b):
                idx = self._battery_index[b]
                return m.SOC[T-1, b] == soc_final[idx]
            m.SOCFinal = Constraint(m.BATTERIES, rule=soc_final_rule)

        # ========== Função Objetivo ==========
        def objective_rule(m):
            total = 0
            if hasattr(s, 'CPG_CONV'):
                for t in m.T:
                    for g in m.CONV_GENERATORS:
                        total += s.CPG_CONV[g] * m.PGER[t, g]
            if s.NGER_EOL > 0 and hasattr(s, 'CPG_CURTAILMENT'):
                for t in m.T:
                    for w in m.WIND_GENERATORS:
                        total += s.CPG_CURTAILMENT[w] * m.CURTAILMENT[t, w]
            if hasattr(s, 'CPG_DEFICIT'):
                for t in m.T:
                    for b in m.BUSES:
                        total += s.CPG_DEFICIT * m.DEFICIT[t, b]
            if tem_bateria and hasattr(s, 'BATTERY_COST'):
                for t in m.T:
                    for b in m.BATTERIES:
                        total += self.custo_bateria[b] * (m.CHARGE[t, b] + m.DISCHARGE[t, b])
            return total
        m.TotalCost = Objective(rule=objective_rule, sense=minimize)

        self._solved = False
        self._soc_inicial_list = soc_inicial if tem_bateria else []
        self._soc_final_list = soc_final if tem_bateria else []

    def solve(self, solver_name='glpk', **solver_args):
        if self.model is None:
            raise RuntimeError("Modelo não construído.")
        solver = SolverFactory(solver_name)
        self._raw_results = solver.solve(self.model, tee=False, **solver_args)
        self._solved = True
        return self._raw_results

    def extract_results(self) -> MultiDayOPFResult:
        if not self._solved or self.model is None:
            raise RuntimeError("Modelo não resolvido.")
        m = self.model
        s = self.sistema
        T = self.T
        snapshots = []

        for t in range(T):
            dia = t // self.n_horas
            hora = t % self.n_horas
            try:
                PGER = [value(m.PGER[t, g]) for g in m.CONV_GENERATORS]
                if s.NGER_EOL > 0:
                    PGWIND_disponivel = [value(m.PGWIND_AVAIL[t, w]) for w in m.WIND_GENERATORS]
                    PGWIND = [value(m.PGWIND[t, w]) for w in m.WIND_GENERATORS]
                    CURTAILMENT = [value(m.CURTAILMENT[t, w]) for w in m.WIND_GENERATORS]
                else:
                    PGWIND_disponivel = PGWIND = CURTAILMENT = []
                DEFICIT = [value(m.DEFICIT[t, b]) for b in m.BUSES]

                SOC_init = []
                SOC_atual = []
                BESS_operation = []
                if hasattr(m, 'BATTERIES') and len(m.BATTERIES) > 0:
                    for b in m.BATTERIES:
                        idx = self._battery_index[b]
                        if t == 0:
                            soc_init = self._soc_inicial_list[idx]
                        else:
                            soc_init = value(m.SOC[t-1, b])
                        SOC_init.append(soc_init)
                        SOC_atual.append(value(m.SOC[t, b]))
                        charge = value(m.CHARGE[t, b]) if m.CHARGE[t, b].value is not None else 0.0
                        discharge = value(m.DISCHARGE[t, b]) if m.DISCHARGE[t, b].value is not None else 0.0
                        BESS_operation.append(charge - discharge)
                else:
                    SOC_init = [0.0]*s.NBAR
                    SOC_atual = [0.0]*s.NBAR
                    BESS_operation = [0.0]*s.NBAR

                V = [value(m.V[t, b]) for b in m.BUSES]
                ANG = [value(m.ANG[t, b]) for b in m.BUSES]
                FLUXO_LIN = [value(m.FLUXO_LIN[t, e]) for e in m.LINES]

                CUSTO = [DEFICIT[b] * s.CPG_DEFICIT for b in m.BUSES] if hasattr(s, 'CPG_DEFICIT') else [0.0]*s.NBAR
                CMO = 0.0
                if hasattr(m, 'dual') and hasattr(m, 'PowerBalance'):
                    try:
                        CMO = m.dual[m.PowerBalance[t, s.slack_idx]]
                    except:
                        CMO = 0.0
                PERDAS_BARRA = [0.0]*s.NBAR

                snapshots.append(MultiDayOPFSnapshotResult(
                    dia=dia, hora=hora, sucesso=True,
                    PGER=PGER, PGWIND_disponivel=PGWIND_disponivel, PGWIND=PGWIND,
                    CURTAILMENT=CURTAILMENT, SOC_init=SOC_init, BESS_operation=BESS_operation,
                    SOC_atual=SOC_atual, DEFICIT=DEFICIT, V=V, ANG=ANG, FLUXO_LIN=FLUXO_LIN,
                    CUSTO=CUSTO, CMO=[CMO], PERDAS_BARRA=PERDAS_BARRA,
                    tempo_execucao=self._raw_results.solver.time if hasattr(self._raw_results.solver, 'time') else 0.0
                ))
            except Exception as e:
                snapshots.append(MultiDayOPFSnapshotResult(
                    dia=dia, hora=hora, sucesso=False, mensagem=str(e)
                ))
        sucesso_global = all(s.sucesso for s in snapshots)
        return MultiDayOPFResult(snapshots=snapshots, sucesso_global=sucesso_global,
                                 mensagem_global="OK" if sucesso_global else "Falhas")

    def solve_multiday(self, solver_name='glpk', fator_carga=None, fator_vento=None,
                       soc_inicial=0.5, soc_final=0.5, cen_id=None):
        """
        Método de conveniência: constrói, resolve e opcionalmente salva resultados.
        soc_inicial e soc_final podem ser escalares ou listas.
        """
        self.build(fator_carga, fator_vento, soc_inicial, soc_final)
        raw = self.solve(solver_name)

        if self.db_handler is not None and cen_id is not None:
            resultados = self.extract_results()
            for snap in resultados.snapshots:
                dia_str = f"{snap.dia+1}"
                # Recupera os fatores usados (precisa ter acesso aos arrays originais)
                if fator_carga is not None:
                    if fator_carga.ndim == 2 and fator_carga.shape[0] == self.n_dias and fator_carga.shape[1] == self.n_horas:
                        perfil_carga_avg = fator_carga[snap.dia, snap.hora]  # escalar
                    else:
                        perfil_carga_avg = np.mean(fator_carga[snap.dia * self.n_horas + snap.hora, :])
                else:
                    perfil_carga_avg = 1.0
                if fator_vento is not None and self.sistema.NGER_EOL > 0:
                    if fator_vento.ndim == 2 and fator_vento.shape[0] == self.n_dias and fator_vento.shape[1] == self.n_horas:
                        perfil_vento_avg = fator_vento[snap.dia, snap.hora]
                    else:
                        perfil_vento_avg = np.mean(fator_vento[snap.dia * self.n_horas + snap.hora, :])
                else:
                    perfil_vento_avg = 1.0
                self.db_handler.save_hourly_result(
                    resultado=snap,
                    sistema=self.sistema,
                    hora=snap.hora,
                    perfil_carga=perfil_carga_avg,
                    perfil_eolica=perfil_vento_avg,
                    solver_name=solver_name,
                    dia=dia_str,
                    cen_id=cen_id
                )
        return raw

# =============================================================================
# Configuração do banco de dados (simplificada)
# =============================================================================
import sqlite3
class OPF_DBHandler:
    """Handler simplificado para salvar resultados no SQLite."""
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def create_tables(self):
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hourly_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cen_id TEXT,
                dia TEXT,
                hora INTEGER,
                sucesso BOOLEAN,
                perfil_carga REAL,
                perfil_eolica REAL,
                solver_name TEXT,
                timestamp TEXT,
                tempo_execucao REAL,
                PGER TEXT,
                PGWIND_disponivel TEXT,
                PGWIND TEXT,
                CURTAILMENT TEXT,
                SOC_init TEXT,
                BESS_operation TEXT,
                SOC_atual TEXT,
                DEFICIT TEXT,
                V TEXT,
                ANG TEXT,
                FLUXO_LIN TEXT,
                CUSTO TEXT,
                CMO TEXT,
                PERDAS_BARRA TEXT,
                mensagem TEXT
            )
        ''')
        self.conn.commit()

    def save_hourly_result(self, resultado, sistema, hora, perfil_carga, perfil_eolica,
                          solver_name, dia, cen_id):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO hourly_results (
                cen_id, dia, hora, sucesso, perfil_carga, perfil_eolica, solver_name,
                timestamp, tempo_execucao, PGER, PGWIND_disponivel, PGWIND, CURTAILMENT,
                SOC_init, BESS_operation, SOC_atual, DEFICIT, V, ANG, FLUXO_LIN,
                CUSTO, CMO, PERDAS_BARRA, mensagem
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            cen_id, dia, hora, resultado.sucesso, perfil_carga, perfil_eolica, solver_name,
            datetime.now().isoformat(), resultado.tempo_execucao,
            str(resultado.PGER), str(resultado.PGWIND_disponivel), str(resultado.PGWIND),
            str(resultado.CURTAILMENT), str(resultado.SOC_init), str(resultado.BESS_operation),
            str(resultado.SOC_atual), str(resultado.DEFICIT), str(resultado.V), str(resultado.ANG),
            str(resultado.FLUXO_LIN), str(resultado.CUSTO), str(resultado.CMO),
            str(resultado.PERDAS_BARRA), resultado.mensagem
        ))
        self.conn.commit()

# =============================================================================
# Função auxiliar para carregar sistema (usa o loader real se disponível)
# =============================================================================
def carregar_sistema(json_path):
    """Tenta carregar o sistema usando o loader real; se falhar, usa dados fictícios."""
    try:
        # Tenta importar o loader real (caminho relativo ao projeto)
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from UTILS.systemLoader import SistemaLoader
        sistema = SistemaLoader(json_path)
        print("  ✓ Sistema carregado com loader real.")
        return sistema
    except:
        print("nenhum sistema encontrado")

# =============================================================================
# Script principal
# =============================================================================
def main():
    print("="*70)
    print("OTIMIZAÇÃO MULTI-PERÍODO ACOPLADA (30 DIAS) - COM BATERIAS")
    print("="*70)

    # -------------------------------------------------------------------------
    # 1. Carregar sistema
    # -------------------------------------------------------------------------
    json_path = "DATA/input/3barras_BASE.json"   # Ajuste conforme necessário
    if not os.path.exists(json_path):
        print(f"Aviso: Arquivo {json_path} não encontrado. Usando dados fictícios internos.")
        sistema = carregar_sistema(None)  # força uso do fictício
    else:
        sistema = carregar_sistema(json_path)

    print(f"\nSistema carregado:")
    print(f"  Barras: {sistema.NBAR}")
    print(f"  Geradores convencionais: {sistema.NGER_CONV}")
    print(f"  Geradores eólicos: {sistema.NGER_EOL}")
    print(f"  Baterias nas barras: {sistema.BARRAS_COM_BATERIA if hasattr(sistema, 'BARRAS_COM_BATERIA') else 'nenhuma'}")

    # -------------------------------------------------------------------------
    # 2. Parâmetros da simulação
    # -------------------------------------------------------------------------
    n_dias = 30
    n_horas = 24
    T = n_dias * n_horas
    print(f"\nSimulando {n_dias} dias x {n_horas} horas = {T} períodos.")

    # -------------------------------------------------------------------------
    # 3. Carregar dados de vento (ou gerar aleatórios)
    # -------------------------------------------------------------------------
    vento_file = "/home/lucasedbraga/repositorios/ufjf/mestrado_luedsbr/SRC/SOLVER/DB/getters/intermittent-renewables-production-france.csv"
    if os.path.exists(vento_file):
        print(f"\nCarregando dados de vento de: {vento_file}")
        df = pd.read_csv(vento_file)
        df['DateTime'] = pd.to_datetime(df['Date and Hour'].str.slice(stop=-6))
        df = df.sort_values('DateTime')
        df_wind = df[df['Source'] == 'Wind'].copy()
        if len(df_wind) > 0:
            max_prod = df_wind['Production'].max()
            df_wind['Factor'] = df_wind['Production'] / max_prod if max_prod > 0 else 0
            # Amostragem aleatória para T períodos
            fatores_vento_1d = df_wind['Factor'].sample(T, replace=True).values
            print(f"  Amostrados {len(fatores_vento_1d)} fatores de vento.")
            # Expandir para 2D: (T, NGER_EOL) - mesmo fator para todos os geradores
            fatores_vento = np.tile(fatores_vento_1d.reshape(-1,1), (1, sistema.NGER_EOL))
        else:
            print("  Aviso: sem dados eólicos, usando fatores aleatórios.")
            fatores_vento = np.random.uniform(0.6, 1.4, (T, sistema.NGER_EOL))
    else:
        print(f"\nAviso: arquivo de vento não encontrado. Gerando fatores aleatórios.")
        fatores_vento = np.random.uniform(0.6, 1.4, (T, sistema.NGER_EOL))

    # -------------------------------------------------------------------------
    # 4. Gerar fatores de carga (perfil horário + incerteza)
    # -------------------------------------------------------------------------
    # Perfil horário típico (fator multiplicativo da carga base)
    nivel_horario = np.array([
        0.6, 0.6, 0.6, 0.6, 0.6, 0.6,  # 0h-5h: leve
        0.7, 0.8, 0.9, 1.0, 1.0, 1.0,  # 6h-11h: médio
        1.1, 1.0, 1.0, 1.0, 1.0, 1.1,  # 12h-17h: médio com picos
        1.2, 1.3, 1.2, 1.1, 1.0, 0.8   # 18h-23h: pesado
    ])

    gu = 0.2  # incerteza global (±10% em torno do nível)
    lim_inf = nivel_horario - gu/2
    lim_sup = nivel_horario + gu/2

    # Gerar fatores para todos os dias: shape (n_dias, n_horas)
    fatores_carga = np.zeros((n_dias, n_horas))
    for d in range(n_dias):
        for h in range(n_horas):
            fatores_carga[d, h] = np.random.uniform(lim_inf[h], lim_sup[h])

    # -------------------------------------------------------------------------
    # 5. Configurar banco de dados
    # -------------------------------------------------------------------------
    db_path = "DATA/output/resultados_acoplado_30dias.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db_handler = OPF_DBHandler(db_path)
    db_handler.create_tables()
    cen_id = datetime.now().strftime('%Y%m%d%H%M%S')
    print(f"\nBanco de dados: {db_path}")
    print(f"Cenário ID: {cen_id}")

    # -------------------------------------------------------------------------
    # 6. Criar modelo e resolver
    # -------------------------------------------------------------------------
    print("\nConstruindo modelo acoplado...")
    modelo = MultiDayOPFModelIntegrated(sistema, n_horas=n_horas, n_dias=n_dias, db_handler=db_handler)

    # Definir SOC inicial e final (escalar para todas as baterias)
    SOC_inicial = 0.5   # 50% da capacidade
    SOC_final = 0.5

    print("Resolvendo... (pode levar vários minutos para 30 dias)")
    raw = modelo.solve_multiday(
        solver_name='glpk',
        fator_carga=fatores_carga,      # shape (dias, horas)
        fator_vento=fatores_vento,      # shape (T, NGER_EOL)
        soc_inicial=SOC_inicial,
        soc_final=SOC_final,
        cen_id=cen_id
    )

    status = raw.solver.termination_condition
    print(f"\nStatus da solução: {status}")

    # -------------------------------------------------------------------------
    # 7. Extrair e exibir resumo
    # -------------------------------------------------------------------------
    resultados = modelo.extract_results()
    sucesso_global = resultados.sucesso_global
    print(f"Sucesso global: {sucesso_global}")
    print(f"Snapshots extraídos: {len(resultados.snapshots)}")

    # Estatísticas rápidas
    custo_total = sum(sum(snap.CUSTO) for snap in resultados.snapshots if snap.sucesso)
    print(f"Custo total aproximado (apenas déficit): {custo_total:.2f} $")

    if hasattr(modelo.model, 'BATTERIES') and len(modelo.model.BATTERIES) > 0:
        soc_final_vals = [value(modelo.model.SOC[T-1, b]) for b in modelo.model.BATTERIES]
        print(f"SOC final das baterias: {soc_final_vals} MWh")

    # Plot opcional (se matplotlib instalado)
    try:
        import matplotlib.pyplot as plt
        if hasattr(modelo.model, 'BATTERIES') and len(modelo.model.BATTERIES) > 0:
            barra_bateria = list(modelo.model.BATTERIES)[0]
            soc_serie = [value(modelo.model.SOC[t, barra_bateria]) for t in range(T)]
            plt.figure(figsize=(12,4))
            plt.plot(soc_serie, linewidth=1)
            plt.title(f"Evolução do SOC - Barra {barra_bateria}")
            plt.xlabel("Período (hora)")
            plt.ylabel("SOC (MWh)")
            plt.grid(True)
            plt.show()
    except ImportError:
        pass

    print("\n" + "="*70)
    print("EXECUÇÃO CONCLUÍDA")
    print("="*70)

if __name__ == "__main__":
    main()