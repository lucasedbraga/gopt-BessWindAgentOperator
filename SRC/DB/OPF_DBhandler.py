import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

class OPF_DBHandler:
    """Gerenciador de banco de dados SQL para resultados do OPF"""
    
    def __init__(self, db_path: str = 'DATA/output/resultados_PL.db'):
        self.db_path = db_path
        self.conn = None
        self.cen_id = None
        
    def connect(self):
        """Conecta ao banco de dados"""
        self.conn = sqlite3.connect(self.db_path)
        return self.conn
    
    def disconnect(self):
        """Desconecta do banco de dados"""
        if self.conn:
            self.conn.close()
    
    def create_tables(self):
        """Cria tabelas necessárias com as novas colunas de cenário"""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Tabela principal com cen_id e unique constraint
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS resultados_opf (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cen_id TEXT,
            timestamp TEXT,
            data_simulacao TEXT,
            hora_simulacao INTEGER,
            sucesso INTEGER,
            custo_total REAL,
            deficit_total REAL,
            curtailment_total REAL,
            perdas_total REAL,
            carga_total REAL,
            eolica_disponivel REAL,
            eolica_utilizada REAL,
            fator_vento REAL,
            iteracoes INTEGER,
            tempo_execucao REAL,
            solver_cenario TEXT,
            sistema_cenario TEXT,
            PLOAD_cenario TEXT,
            PG_result TEXT,
            V_result TEXT,
            ANG_result TEXT,
            PBESS_soc_result TEXT,
            BESS_operation_result TEXT,
            PCURWIND_result TEXT,
            FluxLIN_result TEXT,
            UNIQUE(cen_id, data_simulacao, hora_simulacao)
        )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_resultados_cen ON resultados_opf(cen_id)')

        # Tabela de detalhes por barra
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS DBAR_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cen_id TEXT,
            timestamp TEXT,
            data_simulacao TEXT,
            hora_simulacao INTEGER,
                       
            BAR_id INTEGER,
            BAR_tipo TEXT,
                       
            PLOAD_cenario REAL,
            QLOAD_cenario REAL,
            PGER_total_result REAL,
            PLOSS_result REAL,
            PDEF_result REAL,
            V_result REAL,
            ANG_result REAL,
            PBESS_inst_result REAL,
            PBESS_soc_result REAL,
            CMO_result REAL,
                       
            FOREIGN KEY (cen_id, data_simulacao, hora_simulacao) REFERENCES resultados_opf(cen_id, data_simulacao, hora_simulacao)
        )
        ''')
        
        # Tabela de detalhes por gerador
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS DGER_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cen_id TEXT,
            timestamp TEXT,
            data_simulacao TEXT,
            hora_simulacao INTEGER,
                       
            GER_id INTEGER,
            BAR_id INTEGER,
            GER_tipo TEXT,
                       
            Custo_cenario REAL,
            PGER_result REAL,
            PMAX_result REAL,
            P_MIN_result REAL,
            PGWIND_result REAL,
            PCWIND_result REAL,
                       
            FOREIGN KEY (cen_id, data_simulacao, hora_simulacao) REFERENCES resultados_opf(cen_id, data_simulacao, hora_simulacao)
        )
        ''')
        
        # Tabela de detalhes por linha
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS DLIN_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cen_id TEXT,
            timestamp TEXT,
            data_simulacao TEXT,
            hora_simulacao INTEGER,
                       
            linha_id INTEGER,
            de_barra INTEGER,
            para_barra INTEGER,
            
            PLIM_FLUX REAL,
            FLUX_result REAL,
            PLOSS_result REAL,
            LIN_usage_result REAL,
            
            FOREIGN KEY (cen_id, data_simulacao, hora_simulacao) REFERENCES resultados_opf(cen_id, data_simulacao, hora_simulacao)
        )
        ''')
    
    def save_hourly_result(self,
                       resultado,
                       sistema, hora: int,
                       perfil_carga: float,
                       perfil_eolica: float,
                       solver_name: str = 'glpk',
                       dia: Optional[str] = None,
                       cen_id: Optional[str] = None) -> None:
        
        """Salva resultado no banco de dados conforme novo esquema de tabelas"""
        if cen_id is None:
            raise ValueError("cen_id é obrigatório para identificar a simulação")

        conn = self.connect()
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        SB = sistema.SB

        # ----- Métricas gerais -----
        carga_total = np.sum(sistema.PLOAD) * SB
        capacidade_eolica = sum(sistema.PGMAX_EFETIVO[g] for g in sistema.BAR_GWD) * SB

        eolica_utilizada = 0.0
        for g_idx in sistema.BAR_GWD:
            if g_idx < len(resultado.PG):
                eolica_utilizada += resultado.PG[g_idx] * SB

        # ----- Preparação dos campos JSON -----
        pg_result_json = json.dumps([float(x) for x in resultado.PG]) if hasattr(resultado, 'PG') else '[]'
        v_result_json = json.dumps([float(x) for x in resultado.V]) if hasattr(resultado, 'V') else '[]'
        ang_result_json = json.dumps([float(x) for x in resultado.ANG]) if hasattr(resultado, 'ANG') else '[]'

        # BESS SOC (pode ser dict ou lista)
        bess_soc_json = '[]'
        if hasattr(resultado, 'SOC') and resultado.SOC is not None:
            if isinstance(resultado.SOC, dict):
                bess_soc_json = json.dumps({str(k): float(v) for k, v in resultado.SOC.items()})
            else:
                bess_soc_json = json.dumps([float(x) for x in resultado.SOC])

        # BESS operation
        bess_op_json = '[]'
        if hasattr(resultado, 'BATTERY_OPERATION') and resultado.BATTERY_OPERATION is not None:
            if isinstance(resultado.BATTERY_OPERATION, dict):
                bess_op_json = json.dumps({str(k): v for k, v in resultado.BATTERY_OPERATION.items()})
            else:
                bess_op_json = json.dumps([str(x) for x in resultado.BATTERY_OPERATION])

        # Curtailment eólico
        pcurwind_json = '[]'
        if hasattr(resultado, 'CURTAILMENT') and resultado.CURTAILMENT is not None:
            pcurwind_json = json.dumps([float(x) for x in resultado.CURTAILMENT])

        # Fluxos nas linhas
        fluxlin_json = '[]'
        if hasattr(resultado, 'FLUXO') and resultado.FLUXO is not None:
            fluxlin_json = json.dumps([float(x) for x in resultado.FLUXO])

        # ----- Inserção na tabela principal (resultados_opf) -----
        cursor.execute('''
        INSERT INTO resultados_opf (
            cen_id,
            timestamp,
            data_simulacao,
            hora_simulacao,
            sucesso,
            custo_total,
            deficit_total,
            curtailment_total,
            perdas_total,
            carga_total,
            eolica_disponivel,
            eolica_utilizada,
            fator_vento,
            iteracoes,
            tempo_execucao,
            solver_cenario,
            sistema_cenario,
            PLOAD_cenario,
            PG_result,
            V_result,
            ANG_result,
            PBESS_soc_result,
            BESS_operation_result,
            PCURWIND_result,
            FluxLIN_result
            )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cen_id, data_simulacao, hora_simulacao) DO UPDATE SET
            timestamp = excluded.timestamp,
            sucesso = excluded.sucesso,
            custo_total = excluded.custo_total,
            deficit_total = excluded.deficit_total,
            curtailment_total = excluded.curtailment_total,
            perdas_total = excluded.perdas_total,
            carga_total = excluded.carga_total,
            eolica_disponivel = excluded.eolica_disponivel,
            eolica_utilizada = excluded.eolica_utilizada,
            fator_vento = excluded.fator_vento,
            iteracoes = excluded.iteracoes,
            tempo_execucao = excluded.tempo_execucao,
            solver_cenario = excluded.solver_cenario,
            sistema_cenario = excluded.sistema_cenario,
            PLOAD_cenario = excluded.PLOAD_cenario,
            PG_result = excluded.PG_result,
            V_result = excluded.V_result,
            ANG_result = excluded.ANG_result,
            PBESS_soc_result = excluded.PBESS_soc_result,
            BESS_operation_result = excluded.BESS_operation_result,
            PCURWIND_result = excluded.PCURWIND_result,
            FluxLIN_result = excluded.FluxLIN_result
        ''', 
            (
                cen_id,
                timestamp,
                dia,
                int(hora),
                int(resultado.sucesso),
                float(resultado.custo_total),
                float(sum(resultado.DEFICIT) * SB) if hasattr(resultado, 'DEFICIT') else 0.0,
                float(sum(resultado.CURTAILMENT) * SB) if hasattr(resultado, 'CURTAILMENT') else 0.0,
                float(resultado.perdas * SB) if hasattr(resultado, 'perdas') else 0.0,
                float(carga_total),
                float(capacidade_eolica),
                float(eolica_utilizada),
                float(perfil_eolica),
                int(resultado.iteracoes) if hasattr(resultado, 'iteracoes') else 0,
                float(getattr(resultado, 'tempo_execucao', 0.0)),
                solver_name,
                sistema.json_file_path if hasattr(sistema, 'json_file_path') else 'unknown',
                float(perfil_carga),                     # PLOAD_cenario
                pg_result_json,
                v_result_json,
                ang_result_json,
                bess_soc_json,
                bess_op_json,
                pcurwind_json,
                fluxlin_json
            )
        )

        # ----- Detalhes por barra (DBAR_results) -----
        for i in range(sistema.NBAR):
            barra_id = sistema.indice_para_barra[i]
            # Tipo da barra
            tipo_barra = "PQ"
            for b in sistema.barras:
                if b["ID_Barra"] == barra_id:
                    tipo_barra = b["tipo"]
                    break

            # Geração total na barra
            geracao_barra = 0.0
            for g_idx, barra_idx in enumerate(sistema.BARPG):
                if barra_idx == i and g_idx < len(resultado.PG):
                    geracao_barra += resultado.PG[g_idx] * SB

            deficit_barra = resultado.DEFICIT[i] * SB if i < len(resultado.DEFICIT) else 0.0
            v_barra = resultado.V[i] if i < len(resultado.V) else 0.0
            ang_barra = resultado.ANG[i] * 180 / np.pi if i < len(resultado.ANG) else 0.0

            # BESS na barra (se houver)
            pbess_inst = 0.0
            pbess_soc = 0.0
            for idx_bess, barra_bess in enumerate(sistema.BARRAS_COM_BATERIA):
                if barra_bess == i:
                    if hasattr(resultado, 'BATTERY_POWER') and idx_bess < len(resultado.BATTERY_POWER):
                        pbess_inst = resultado.BATTERY_POWER[barra_bess] * SB
                    if hasattr(resultado, 'SOC'):
                        if isinstance(resultado.SOC, (list, tuple)) and idx_bess < len(resultado.SOC):
                            pbess_soc = resultado.SOC[barra_bess]
                    break

            cmo = resultado.cmo_total if hasattr(resultado, 'cmo_total') else 0.0

            cursor.execute('''
            INSERT INTO DBAR_results (
                cen_id,
                timestamp,
                data_simulacao,
                hora_simulacao,
                           
                BAR_id,
                BAR_tipo,
                PLOAD_cenario,
                QLOAD_cenario,
                PGER_total_result,
                PLOSS_result,
                PDEF_result,
                V_result,
                ANG_result,
                PBESS_inst_result,
                PBESS_soc_result,
                CMO_result
            ) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', 
                (
                    cen_id,
                    timestamp,
                    dia,
                    int(hora),
                    barra_id,
                    tipo_barra,
                    float(sistema.PLOAD[i] * SB),
                    float(sistema.QLOAD[i] * SB) if i < len(sistema.QLOAD) else 0.0,
                    float(geracao_barra),
                    0.0,
                    float(deficit_barra),
                    float(v_barra),
                    float(ang_barra),
                    float(pbess_inst),
                    float(pbess_soc),
                    float(cmo)
                )
            )

        # ----- Detalhes por gerador (DGER_results) -----
        for g_idx in range(len(resultado.PG)):
            if g_idx < len(sistema.BARPG):
                barra_idx = sistema.BARPG[g_idx]
                barra_id = sistema.indice_para_barra[barra_idx]
                tipo_gerador = sistema.GER_TIPOS_COMBINADO[g_idx] if g_idx < len(sistema.GER_TIPOS_COMBINADO) else "UNKNOWN"

                custo_ger = sistema.CPG[g_idx] if g_idx < len(sistema.CPG) else 0.0

                pgwind = 0.0
                pcwind = 0.0
                if g_idx in sistema.BAR_GWD:
                    pgwind = resultado.PG[g_idx] * SB
                    idx_gwd = sistema.BAR_GWD.index(g_idx)
                    if hasattr(resultado, 'CURTAILMENT') and idx_gwd < len(resultado.CURTAILMENT):
                        pcwind = resultado.CURTAILMENT[idx_gwd] * SB

                cursor.execute('''
                INSERT INTO DGER_results (
                    cen_id,
                    timestamp,
                    data_simulacao,
                    hora_simulacao,
                               
                    GER_id,
                    BAR_id,
                    GER_tipo,
                    custo_cenario,
                    PGER_result,
                    PMAX_result,
                    P_MIN_result,
                    PGWIND_result,
                    PCWIND_result
                ) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', 
                    (
                        cen_id,
                        timestamp,
                        dia,
                        int(hora),
                        g_idx,
                        barra_id,
                        tipo_gerador,
                        float(custo_ger),
                        float(resultado.PG[g_idx] * SB),
                        float(sistema.PGMAX[g_idx] * SB) if g_idx < len(sistema.PGMAX) else 0.0,
                        float(sistema.PGMIN[g_idx] * SB) if g_idx < len(sistema.PGMIN) else 0.0,
                        float(pgwind),
                        float(pcwind)
                    )
                )

        # ----- Detalhes por linha (DLIN_results) -----
        for e_idx in range(len(resultado.FLUXO)):
            if e_idx < sistema.NLIN:
                de_barra = sistema.indice_para_barra[sistema.line_fr[e_idx]]
                para_barra = sistema.indice_para_barra[sistema.line_to[e_idx]]

                fluxo_mw = resultado.FLUXO[e_idx] * SB
                limite_mw = sistema.FLIM[e_idx] * SB
                carregamento = abs(fluxo_mw / limite_mw * 100) if limite_mw > 0 else 0.0

                r = sistema.r_line[e_idx]
                perdas_mw = r * (resultado.FLUXO[e_idx] ** 2) * SB

                cursor.execute('''
                INSERT INTO DLIN_results (
                    cen_id,
                    timestamp,
                    data_simulacao,
                    hora_simulacao,
                    linha_id,
                    de_barra,
                    para_barra,
                               
                    PLIM_FLUX,
                    FLUX_result,
                    PLOSS_result,
                    LIN_usage_result             
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', 
                    (
                        cen_id,
                        timestamp,
                        dia,
                        int(hora),
                        e_idx,
                        de_barra,
                        para_barra,
                        float(limite_mw),
                        float(fluxo_mw),
                        float(perdas_mw),
                        float(carregamento)
                    )
                )

        conn.commit()
        conn.close()

    def export_to_csv(self, output_path: str):
        """Exporta todos os resultados para CSV (inalterado)"""
        import os
        
        conn = self.connect()
        
        # Criar diretório se não existir
        os.makedirs(output_path, exist_ok=True)
        
        # Lista de tabelas
        tabelas = ['resultados_opf', 'DBAR_results', 'DGER_results', 'DLIN_results']
        
        for tabela in tabelas:
            df = pd.read_sql_query(f"SELECT * FROM {tabela}", conn)
            caminho_arquivo = os.path.join(output_path, f"{tabela}.csv")
            df.to_csv(caminho_arquivo, index=False)
            print(f"   ✓ {tabela}: {len(df)} registros exportados")
        
        conn.close()