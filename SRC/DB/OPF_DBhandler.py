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
            tempo_execucao REAL,
            solver_cenario TEXT,

            sistema_cenario TEXT,
            fator_vento_cenario REAL,
            PLOAD_cenario TEXT,
            BESS_soc_init_cenario TEXT,
                       
            PGER_result TEXT,
            PGWIND_result TEXT,
            CURTAILMENT_result TEXT,
            BESS_operation_result TEXT,
            BESS_soc_atual_result TEXT,
 
                                
            V_result TEXT,
            ANG_result TEXT,
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
            BESS_init_cenario REAL,
                       
            PGER_total_result REAL,
            PGWIND_total_result REAL,
            PCURTAILMENT_total_result REAL,
            BESS_operation_result REAL,
            BESS_soc_atual_result REAL,
            PLOSS_result REAL,
            PDEF_result REAL,
                       
            V_result REAL,
            ANG_result REAL,
                                              
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

        # ----- Preparação dos campos JSON -----
       
        pg_result_json = json.dumps([float(x) for x in resultado.PGER]) if hasattr(resultado, 'PGER') else '[]'
        pgwind_result_json = json.dumps([float(x) for x in resultado.PGWIND]) if hasattr(resultado, 'PGWIND') else '[]'
        curtailment_result_json = json.dumps([float(x) for x in resultado.CURTAILMENT]) if hasattr(resultado, 'CURTAILMENT') else '[]'
        BESS_soc_init_json = json.dumps([float(x) for x in resultado.SOC_init]) if hasattr(resultado, 'SOC_init') else '[]'
        BESS_operation_json = json.dumps([float(x) for x in resultado.BESS_operation]) if hasattr(resultado, 'BESS_operation') else '[]'
        BESS_soc_atual_json = json.dumps([float(x) for x in resultado.SOC_atual]) if hasattr(resultado, 'SOC_atual') else '[]'
        
        v_result_json = json.dumps([float(x) for x in resultado.V]) if hasattr(resultado, 'V') else '[]'
        ang_result_json = json.dumps([float(x) for x in resultado.ANG]) if hasattr(resultado, 'ANG') else '[]'
        fluxlin_json = json.dumps([float(x) for x in resultado.FLUXO_LIN]) if hasattr(resultado, 'FLUXO_LIN') else '[]'

        # ----- Inserção na tabela principal (resultados_opf) -----
        cursor.execute('''
        INSERT INTO resultados_opf (
            cen_id,
            timestamp,
            data_simulacao,
            hora_simulacao,
            sucesso,
            tempo_execucao,
            solver_cenario,
                       
            sistema_cenario,
            fator_vento_cenario,
            PLOAD_cenario,
            BESS_soc_init_cenario,
                       
            PGER_result,
            PGWIND_result,
            CURTAILMENT_result,
            BESS_operation_result,
            BESS_soc_atual_result,

                       
            V_result,
            ANG_result,
            FluxLIN_result
            )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', 
            (
                cen_id,
                timestamp,
                dia,
                int(hora),
                int(resultado.sucesso),
                float(getattr(resultado, 'tempo_execucao', 0.0)),
                solver_name,

                sistema.json_file_path if hasattr(sistema, 'json_file_path') else 'unknown',
                float(perfil_eolica),
                float(perfil_carga),
                BESS_soc_init_json,


                pg_result_json,
                pgwind_result_json,
                curtailment_result_json,
                BESS_operation_json,
                BESS_soc_atual_json,

                v_result_json,
                ang_result_json,
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
                if barra_idx == i and g_idx < len(resultado.PGER):
                    geracao_barra += resultado.PGER[g_idx] * SB

            # Geração eólica na barra
            geracao_gwind_barra = 0.0
            for g_idx in sistema.BAR_GWD:
                if g_idx == i and g_idx < len(resultado.PGER):
                    geracao_gwind_barra += resultado.PGER[g_idx] * SB

            # Curtailment eólico na barra
            curtailment_barra = 0.0
            for idx_gwd, barra_gwd in enumerate(sistema.BAR_GWD):
                if barra_gwd == i:
                    if hasattr(resultado, 'CURTAILMENT') and idx_gwd < len(resultado.CURTAILMENT):
                        curtailment_barra += resultado.CURTAILMENT[idx_gwd] * SB
                    
            # Perdas na barra (estimativa)
            perdas_barra = 0.0
            for e_idx in range(sistema.NLIN):
                if sistema.line_fr[e_idx] == i or sistema.line_to[e_idx] == i:
                    r = sistema.r_line[e_idx]
                    fluxo = resultado.FLUXO_LIN[e_idx] if e_idx < len(resultado.FLUXO_LIN) else 0.0
                    perdas_barra += r * (fluxo ** 2) * SB

            # Déficit na barra (se houver)      
            deficit_barra = resultado.DEFICIT[i] * SB if i < len(resultado.DEFICIT) else 0.0
            v_barra = resultado.V[i] if i < len(resultado.V) else 0.0
            ang_barra = resultado.ANG[i] * 180 / np.pi if i < len(resultado.ANG) else 0.0

            # BESS na barra (índice i)
            pbess_soc_init = resultado.SOC_init[i] if i < len(resultado.SOC_init) else 0.0
            pbess_soc_operation = resultado.BESS_operation[i] if i < len(resultado.BESS_operation) else 0.0
            pbess_soc_atual = resultado.SOC_atual[i] if i < len(resultado.SOC_atual) else 0.0

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
                BESS_init_cenario,
                PGER_total_result,
                PGWIND_total_result,
                PCURTAILMENT_total_result,
                BESS_operation_result,
                BESS_soc_atual_result,
                PLOSS_result,
                PDEF_result,
                V_result,
                ANG_result
            ) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                float(pbess_soc_init),           # BESS_init_cenario
                float(geracao_barra),             # PGER_total_result
                float(geracao_gwind_barra),       # PGWIND_total_result
                float(curtailment_barra),         # PCURTAILMENT_total_result
                float(pbess_soc_operation),       # BESS_operation_result
                float(pbess_soc_atual),           # BESS_soc_atual_result
                float(perdas_barra),              # PLOSS_result
                float(deficit_barra),             # PDEF_result
                float(v_barra),                   # V_result
                float(ang_barra)                   # ANG_result
            ))
        # ----- Detalhes por gerador (DGER_results) -----
        for g_idx in range(len(resultado.PGER)):
            if g_idx < len(sistema.BARPG):
                barra_idx = sistema.BARPG[g_idx]
                barra_id = sistema.indice_para_barra[barra_idx]
                tipo_gerador = sistema.GER_TIPOS_COMBINADO[g_idx] if g_idx < len(sistema.GER_TIPOS_COMBINADO) else "UNKNOWN"

                custo_ger = sistema.CPG[g_idx] if g_idx < len(sistema.CPG) else 0.0

                pgwind = 0.0
                pcwind = 0.0
                if g_idx in sistema.BAR_GWD:
                    pgwind = resultado.PGER[g_idx] * SB
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
                        float(resultado.PGER[g_idx] * SB),
                        float(sistema.PGMAX[g_idx] * SB) if g_idx < len(sistema.PGMAX) else 0.0,
                        float(sistema.PGMIN[g_idx] * SB) if g_idx < len(sistema.PGMIN) else 0.0,
                        float(pgwind),
                        float(pcwind)
                    )
                )

        # ----- Detalhes por linha (DLIN_results) -----
        for e_idx in range(len(resultado.FLUXO_LIN)):
            if e_idx < sistema.NLIN:
                de_barra = sistema.indice_para_barra[sistema.line_fr[e_idx]]
                para_barra = sistema.indice_para_barra[sistema.line_to[e_idx]]

                fluxo_mw = resultado.FLUXO_LIN[e_idx] * SB
                limite_mw = sistema.FLIM[e_idx] * SB
                carregamento = abs(fluxo_mw / limite_mw * 100) if limite_mw > 0 else 0.0

                r = sistema.r_line[e_idx]
                perdas_mw = r * (resultado.FLUXO_LIN[e_idx] ** 2) * SB

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