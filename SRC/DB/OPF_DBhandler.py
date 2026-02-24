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
            PGER_result TEXT,

            PGWIND_disponivel_cenario TEXT,             
            PGWIND_result TEXT,
            CURTAILMENT_result TEXT,
                       
            BESS_soc_init_cenario TEXT,
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
            PGER_CONV_total_result REAL,
            PLOSS_result REAL,
            PDEF_result REAL,                       

            PGWIND_disponivel_cenario REAL,
            PGWIND_total_result REAL,
            CURTAILMENT_total_result REAL,
            
            BESS_init_cenario REAL,
            BESS_operation_result REAL,
            BESS_soc_atual_result REAL,   
                       
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
            PGWIND_disponivel_cenario REAL,
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
        """Salva resultado no banco de dados, adaptado para nova estrutura."""
        if cen_id is None:
            raise ValueError("cen_id é obrigatório para identificar a simulação")

        conn = self.connect()
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        SB = sistema.SB
        TOL = 1e-6

        def safe_value(x):
            return 0.0 if abs(x) < TOL else float(x)

        # ----- Preparação dos dados de geradores eólicos -----
        n_eol = sistema.NGER_EOL
        gwd_pgwind = [0.0] * n_eol
        gwd_curtail = [0.0] * n_eol
        gwd_disponivel = [0.0] * n_eol

        for pos in range(n_eol):
            if hasattr(resultado, 'PGWIND') and pos < len(resultado.PGWIND):
                gwd_pgwind[pos] = safe_value(resultado.PGWIND[pos] * SB)
            if hasattr(resultado, 'CURTAILMENT') and pos < len(resultado.CURTAILMENT):
                gwd_curtail[pos] = safe_value(resultado.CURTAILMENT[pos] * SB)
            if hasattr(sistema, 'PGWIND_disponivel') and pos < len(sistema.PGWIND_disponivel):
                gwd_disponivel[pos] = safe_value(sistema.PGWIND_disponivel[pos] * SB)

        # ----- Preparação dos campos JSON -----
        def json_from_array(arr, default='[]'):
            if arr is None or len(arr) == 0:
                return default
            return json.dumps([safe_value(x) for x in arr])

        pg_result_json = json_from_array(resultado.PGER)
        pgwind_result_json = json_from_array(resultado.PGWIND)
        curtailment_result_json = json_from_array(resultado.CURTAILMENT)
        BESS_soc_init_json = json_from_array(resultado.SOC_init)
        BESS_operation_json = json_from_array(resultado.BESS_operation)
        BESS_soc_atual_json = json_from_array(resultado.SOC_atual)
        v_result_json = json_from_array(resultado.V)
        ang_result_json = json_from_array(resultado.ANG)
        fluxlin_json = json_from_array(resultado.FLUXO_LIN)

        # Array combinado para disponibilidade (convencionais zeros + eólicos)
        pgwind_disponivel_total = [0.0] * (sistema.NGER_CONV + n_eol)
        for pos in range(n_eol):
            pgwind_disponivel_total[sistema.NGER_CONV + pos] = gwd_disponivel[pos]
        pgwind_disponivel_json = json.dumps([safe_value(x) for x in pgwind_disponivel_total])

        # ----- Inserção na tabela resultados_opf -----
        cursor.execute('''
        INSERT INTO resultados_opf (
            cen_id, timestamp, data_simulacao, hora_simulacao, sucesso, tempo_execucao, solver_cenario,
            sistema_cenario, fator_vento_cenario,
            PLOAD_cenario, PGER_result,
            PGWIND_disponivel_cenario, PGWIND_result, CURTAILMENT_result,
            BESS_soc_init_cenario, BESS_operation_result, BESS_soc_atual_result,
            V_result, ANG_result, FluxLIN_result
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            cen_id, timestamp, dia, int(hora), int(resultado.sucesso),
            safe_value(getattr(resultado, 'tempo_execucao', 0.0)), solver_name,
            sistema.json_file_path if hasattr(sistema, 'json_file_path') else 'unknown',
            safe_value(perfil_eolica), safe_value(perfil_carga),
            pg_result_json, pgwind_disponivel_json, pgwind_result_json, curtailment_result_json,
            BESS_soc_init_json, BESS_operation_json, BESS_soc_atual_json,
            v_result_json, ang_result_json, fluxlin_json
        ))

        # ----- Detalhes por barra (DBAR_results) -----
        for i in range(sistema.NBAR):
            barra_id = sistema.indice_para_barra[i]
            tipo_barra = next((b["tipo"] for b in sistema.barras if b["ID_Barra"] == barra_id), "PQ")

            # Acumuladores
            geracao_conv = 0.0
            geracao_eol = 0.0
            curtailment = 0.0
            disponivel_eol = 0.0

            # Convencionais
            for g in range(sistema.NGER_CONV):
                if sistema.BARPG_CONV[g] == i and g < len(resultado.PGER):
                    geracao_conv += safe_value(resultado.PGER[g] * SB)

            # Eólicos
            for pos in range(n_eol):
                if sistema.BARPG_EOL[pos] == i:
                    geracao_eol += gwd_pgwind[pos]
                    curtailment += gwd_curtail[pos]
                    disponivel_eol += gwd_disponivel[pos]

            # Perdas na barra
            perdas = 0.0
            for e in range(sistema.NLIN):
                if sistema.line_fr[e] == i or sistema.line_to[e] == i:
                    r = sistema.r_line[e]
                    fluxo = resultado.FLUXO_LIN[e] if e < len(resultado.FLUXO_LIN) else 0.0
                    perdas += r * (fluxo ** 2) * SB
            perdas = safe_value(perdas)

            # Déficit, tensão, ângulo
            deficit = safe_value(resultado.DEFICIT[i] * SB) if i < len(resultado.DEFICIT) else 0.0
            v = safe_value(resultado.V[i]) if i < len(resultado.V) else 0.0
            ang = safe_value(resultado.ANG[i] * 180 / np.pi) if i < len(resultado.ANG) else 0.0

            # BESS
            bess_init = safe_value(resultado.SOC_init[i]) if i < len(resultado.SOC_init) else 0.0
            bess_op = safe_value(resultado.BESS_operation[i]) if i < len(resultado.BESS_operation) else 0.0
            bess_atual = safe_value(resultado.SOC_atual[i]) if i < len(resultado.SOC_atual) else 0.0

            cursor.execute('''
            INSERT INTO DBAR_results (
                cen_id, timestamp, data_simulacao, hora_simulacao,
                BAR_id, BAR_tipo,
                PLOAD_cenario, PGER_CONV_total_result, PLOSS_result, PDEF_result,
                PGWIND_disponivel_cenario, PGWIND_total_result, CURTAILMENT_total_result,
                BESS_init_cenario, BESS_operation_result, BESS_soc_atual_result,
                V_result, ANG_result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                cen_id, timestamp, dia, int(hora), barra_id, tipo_barra,
                safe_value(sistema.PLOAD[i] * SB),
                safe_value(geracao_conv),
                safe_value(perdas),
                safe_value(deficit),
                safe_value(disponivel_eol),
                safe_value(geracao_eol),
                safe_value(curtailment),
                safe_value(bess_init),
                safe_value(bess_op),
                safe_value(bess_atual),
                safe_value(v),
                safe_value(ang)
            ))

        # ----- Detalhes por gerador (DGER_results) -----
        # Convencionais
        for g in range(sistema.NGER_CONV):
            barra_idx = sistema.BARPG_CONV[g]
            barra_id = sistema.indice_para_barra[barra_idx]
            tipo = sistema.GER_TIPOS_CONV[g] if g < len(sistema.GER_TIPOS_CONV) else "CONV"
            custo = safe_value(sistema.CPG_CONV[g]) if g < len(sistema.CPG_CONV) else 0.0

            cursor.execute('''
            INSERT INTO DGER_results (
                cen_id, timestamp, data_simulacao, hora_simulacao,
                GER_id, BAR_id, GER_tipo,
                custo_cenario, PGER_result, PMAX_result, P_MIN_result,
                PGWIND_disponivel_cenario, PGWIND_result, PCWIND_result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                cen_id, timestamp, dia, int(hora),
                g, barra_id, tipo,
                custo,
                safe_value(resultado.PGER[g] * SB) if g < len(resultado.PGER) else 0.0,
                safe_value(sistema.PGMAX_CONV[g] * SB) if g < len(sistema.PGMAX_CONV) else 0.0,
                safe_value(sistema.PGMIN_CONV[g] * SB) if g < len(sistema.PGMIN_CONV) else 0.0,
                0.0, 0.0, 0.0  # eólicos não se aplicam
            ))

        # Eólicos
        for pos in range(n_eol):
            barra_idx = sistema.BARPG_EOL[pos]
            barra_id = sistema.indice_para_barra[barra_idx]
            tipo = "GWD"
            cursor.execute('''
            INSERT INTO DGER_results (
                cen_id, timestamp, data_simulacao, hora_simulacao,
                GER_id, BAR_id, GER_tipo,
                custo_cenario, PGER_result, PMAX_result, P_MIN_result,
                PGWIND_disponivel_cenario, PGWIND_result, PCWIND_result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                cen_id, timestamp, dia, int(hora),
                sistema.NGER_CONV + pos, barra_id, tipo,
                0.0,  # custo_cenario (pode ser ajustado)
                0.0,  # PGER_result
                safe_value(sistema.PGMAX_EOL_EFETIVO[pos] * SB) if pos < len(sistema.PGMAX_EOL_EFETIVO) else 0.0,
                0.0,  # P_MIN
                safe_value(gwd_disponivel[pos]),
                safe_value(gwd_pgwind[pos]),
                safe_value(gwd_curtail[pos])
            ))

        # ----- Detalhes por linha (DLIN_results) -----
        for e in range(len(resultado.FLUXO_LIN)):
            if e < sistema.NLIN:
                de = sistema.indice_para_barra[sistema.line_fr[e]]
                para = sistema.indice_para_barra[sistema.line_to[e]]
                fluxo_mw = safe_value(resultado.FLUXO_LIN[e] * SB)
                limite_mw = safe_value(sistema.FLIM[e] * SB)
                perdas_mw = safe_value(sistema.r_line[e] * (resultado.FLUXO_LIN[e] ** 2) * SB)
                carreg = safe_value(((abs(fluxo_mw) + perdas_mw) / limite_mw) * 100) if limite_mw > 0 else 0.0

                cursor.execute('''
                INSERT INTO DLIN_results (
                    cen_id, timestamp, data_simulacao, hora_simulacao,
                    linha_id, de_barra, para_barra,
                    PLIM_FLUX, FLUX_result, PLOSS_result, LIN_usage_result
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    cen_id, timestamp, dia, int(hora),
                    e, de, para,
                    limite_mw, fluxo_mw, perdas_mw, carreg
                ))

        conn.commit()
        conn.close()

def export_to_csv(self, output_path: str):
        """Exporta todos os resultados para CSV"""
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