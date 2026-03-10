import json
import numpy as np
from typing import Dict, List, Any

class SistemaLoader:
    """Carrega e processa dados do sistema a partir de JSON"""

    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.data = None
        self.barras = []
        self.geradores_data = []
        self.demandas_data = []
        self.linhas = []
        self.baterias_data = []

        self.SB = 100.0  # Potência base
        self.VB = 230.0  # Tensão base
        self.f_base = 60.0  # Frequência base
        self.ZB = 0.0  # Impedância base

        # Estruturas processadas
        self.bus_ids = []
        self.idx_map = {}
        self.indice_para_barra = {}
        self.NBAR = 0
        self.NLIN = 0
        self.slack_idx = 0

        # Arrays de linhas
        self.line_fr = []
        self.line_to = []
        self.x_line = np.array([])
        self.r_line = np.array([])
        self.FLIM = np.array([])

        # --- Geradores convencionais (UTE, UTH) ---
        self.NGER_CONV = 0
        self.BARPG_CONV = []          # índices das barras dos convencionais
        self.GER_TIPOS_CONV = []      # tipos ("UTE", "UTH", etc.)
        self.PGMIN_CONV = np.array([])
        self.PGMAX_CONV = np.array([])
        self.CPG_CONV = np.array([])
        self.RAMP_UP = np.array([])
        self.RAMP_DOWN = np.array([])
        self.PGER_INICIAL_CONV = np.array([])   # geração inicial (pu)

        # --- Geradores eólicos (GWD) ---
        self.NGER_EOL = 0
        self.BARPG_EOL = []            # índices das barras dos eólicos
        self.PGMAX_EOL_ORIGINAL = np.array([])   # capacidade instalada (pu)
        self.PGMAX_EOL_EFETIVO = np.array([])    # após fator de vento (pu)
        self.PGWIND_disponivel = np.array([])    # disponibilidade atual (pu) – igual ao efetivo
        self.CPG_CURTAILMENT = np.array([])      # custo de curtailment por eólico (USD/pu)

        # Mapeamento auxiliar: índice global do gerador no JSON -> posição na lista de eólicos
        self.gwd_idx_to_pos = {}

        # --- Déficit (por barra) ---
        self.CPG_DEFICIT = 5000.0 * self.SB   # custo do déficit (USD/pu), pode ser escalar

        # Carga
        self.PLOAD = np.array([])
        self.QLOAD = np.array([])

        # Processa o sistema
        self.carrega_sistema()
        self.processa_sistema()

    def carrega_sistema(self):
        """Carrega dados do JSON"""
        with open(self.json_file_path, 'r') as f:
            self.data = json.load(f)

        self.barras = self.data["BARRAS"]
        self.geradores_data = self.data["GERADORES"]
        self.demandas_data = self.data["DEMANDAS"]
        self.linhas = self.data["LINHAS"]
        self.baterias_data = self.data.get("BATERIAS", [])

    def processa_sistema(self):
        """Processa todos os dados do sistema"""
        # Parâmetros base
        self.SB = self.data["S_base"]
        self.VB = self.data["V_base"]
        self.f_base = self.data["f_base"]
        self.ZB = (self.VB ** 2) / self.SB

        # Processa em PU
        self.processa_pu()

        # Processa barras
        self.processa_barras()

        # Processa linhas
        self.processa_linhas()

        # Processa geradores (separando convencionais e eólicos)
        self.processa_geradores()

        # Processa carga
        self.processa_carga()

        # Processa déficit e curtailment (não cria geradores artificiais, apenas define custos)
        self.processa_def_gwd()

        # Processa baterias (se houver)
        if len(self.baterias_data) > 0:
            self.processa_baterias()
        else:
            self.BARRAS_COM_BATERIA = []
            self.BATTERIES = []
            self.BATTERY_POWER_LIMIT = np.zeros(self.NBAR)
            self.BATTERY_POWER_OUT = np.zeros(self.NBAR)
            self.BATTERY_CAPACITY = np.zeros(self.NBAR)
            self.BATTERY_MIN_SOC = np.zeros(self.NBAR)
            self.BATTERY_INITIAL_SOC = np.zeros(self.NBAR)
            self.BATTERY_COST = np.zeros(self.NBAR)

    def processa_pu(self):
        """Converte todos os valores para PU"""
        # Barras
        for b in self.barras:
            b["P_carga_pu"] = b.get("P_carga_MW", 0.0) / self.SB
            b["Q_carga_pu"] = b.get("Q_carga_MVAr", 0.0) / self.SB

        # Geradores
        for g in self.geradores_data:
            g["PGERmin_pu"] = g.get("PGERmin_MW", 0.0) / self.SB
            g["PGERmax_pu"] = g.get("PGERmax_MW", 0.0) / self.SB
            g["Qmin_pu"] = g.get("Qmin_MW", 0.0) / self.SB
            g["Qmax_pu"] = g.get("Qmax_MW", 0.0) / self.SB

            # Custos em USD/pu
            g["custo_var_pu"] = g.get("custo_var_USD_MW", 0.0) * self.SB
            g["custo_curtailment_pu"] = g.get("custo_curtailment_USD_MW", 100.0) * self.SB

        # Demandas
        for d in self.demandas_data:
            d["PLOAD_pu"] = d.get("PLOAD", 0.0) / self.SB
            d["QLOAD_pu"] = d.get("QLOAD_MW", 0.0) / self.SB

        # Linhas
        for l in self.linhas:
            if l.get("R_Unidade", "pu") != "pu":
                l["R"] = l["R"] / self.ZB
                l["X"] = l["X"] / self.ZB

            if "Bsh" in l and l.get("Bsh_Unidade", "pu") != "pu":
                l["Bsh"] = l["Bsh"] * self.ZB

            if l.get("LIM_Fluxo_Unidade", "MW") == "MW":
                l["Fmax_pu"] = l["LIM_Fluxo"] / self.SB
            else:
                l["Fmax_pu"] = l["LIM_Fluxo"]

    def processa_barras(self):
        """Processa dados das barras"""
        self.bus_ids = [b["ID_Barra"] for b in self.barras]
        self.idx_map = {id: i for i, id in enumerate(self.bus_ids)}
        self.indice_para_barra = {i: id for i, id in enumerate(self.bus_ids)}
        self.NBAR = len(self.bus_ids)

        # Identifica barra slack
        slack_list = [b for b in self.barras if b["tipo"] == "Slack"]
        if len(slack_list) != 1:
            raise ValueError("Deve haver exatamente 1 barra Slack")
        slack_id = slack_list[0]["ID_Barra"]
        self.slack_idx = self.idx_map[slack_id]

    def processa_linhas(self):
        """Processa dados das linhas"""
        self.NLIN = len(self.linhas)

        self.line_fr = []
        self.line_to = []
        self.x_line = np.zeros(self.NLIN)
        self.r_line = np.zeros(self.NLIN)
        self.FLIM = np.zeros(self.NLIN)

        for e, ln in enumerate(self.linhas):
            fr = ln["ID_Barra_Origem"]
            to = ln["ID_Barra_Destino"]
            self.line_fr.append(self.idx_map[fr])
            self.line_to.append(self.idx_map[to])

            self.x_line[e] = ln.get("X", 0.01)
            self.r_line[e] = ln.get("R", 0.001)
            self.FLIM[e] = ln.get("Fmax_pu", 1.0)

    def processa_geradores(self):
        """
        Separa os geradores em convencionais (UTE, UTH) e eólicos (GWD).
        Preenche os arrays correspondentes.
        """
        # Listas temporárias para convencionais
        barpg_conv = []
        tipos_conv = []
        pgmin_conv = []
        pgmax_conv = []
        cpg_conv = []
        ramp_up_MW_h = []
        ramp_down_MW_h = []
        pg_inicial_conv = []  # geração inicial (pu)

        # Listas temporárias para eólicos
        barpg_eol = []
        pgmax_eol_orig = []
        custo_curtail = []

        # Itera sobre todos os geradores do JSON
        for i, g in enumerate(self.geradores_data):
            id_barra = g["ID_Barra"]
            barra_idx = self.idx_map[id_barra]
            tipo = g.get("Tipo", "CONV")

            if tipo == "GWD":
                # Gerador eólico
                pos = len(barpg_eol)
                barpg_eol.append(barra_idx)
                pgmax_orig = g.get("PGERmax_pu", 0.0)
                pgmax_eol_orig.append(pgmax_orig)
                custo_curtail.append(g.get("custo_curtailment_pu", 1000.0))
                self.gwd_idx_to_pos[i] = pos  # mapeia índice global para posição na lista de eólicos
            else:
                # Gerador convencional
                barpg_conv.append(barra_idx)
                tipos_conv.append(tipo)
                pgmin_conv.append(g.get("PGERmin_pu", 0.0))
                pgmax_conv.append(g.get("PGERmax_pu", 1.0))
                cpg_conv.append(g.get("custo_var_pu", 50.0))
                ramp_up_MW_h.append(g.get("ramp_up_MW_h", 100))
                ramp_down_MW_h.append(g.get("ramp_down_MW_h", 100))
                # Geração inicial: campo opcional, se não existir, assume 0.0
                pg_ini_mw = g.get("PGER_inicial_MW", 0.0)
                pg_inicial_conv.append(pg_ini_mw / self.SB)

        # Converte para arrays numpy
        self.NGER_CONV = len(barpg_conv)
        self.BARPG_CONV = barpg_conv
        self.GER_TIPOS_CONV = tipos_conv
        self.PGMIN_CONV = np.array(pgmin_conv)
        self.PGMAX_CONV = np.array(pgmax_conv)
        self.CPG_CONV = np.array(cpg_conv)
        self.RAMP_UP = np.array(ramp_up_MW_h)
        self.RAMP_DOWN = np.array(ramp_down_MW_h)
        self.PGER_INICIAL_CONV = np.array(pg_inicial_conv)

        self.NGER_EOL = len(barpg_eol)
        self.BARPG_EOL = barpg_eol
        self.PGMAX_EOL_ORIGINAL = np.array(pgmax_eol_orig)
        self.PGMAX_EOL_EFETIVO = self.PGMAX_EOL_ORIGINAL.copy()  # inicialmente igual à original
        self.PGWIND_disponivel = self.PGMAX_EOL_EFETIVO.copy()   # disponibilidade atual (será atualizada)
        self.CPG_CURTAILMENT = np.array(custo_curtail)

        print(f"  ✓ Geradores processados: {self.NGER_CONV} convencionais, {self.NGER_EOL} eólicos")

    def processa_carga(self):
        """Processa dados de carga"""
        self.PLOAD = np.zeros(self.NBAR)
        self.QLOAD = np.zeros(self.NBAR)

        for d in self.demandas_data:
            idx = self.idx_map[d["ID_Barra"]]
            self.PLOAD[idx] += d.get("PLOAD_pu", 0.0)
            self.QLOAD[idx] += d.get("QLOAD_pu", 0.0)

    def processa_def_gwd(self):
        """
        Define as barras que podem ter déficit (PQ sem gerador convencional)
        e o custo do déficit. Não cria geradores artificiais.
        O déficit será modelado como variável por barra no OPF.
        O curtailment está associado aos geradores eólicos.
        """
        # Barras PQ
        barras_PQ = [b for b in self.barras if b["tipo"] == "PQ"]
        # Barras que possuem gerador convencional
        barras_com_gerador_conv = set(self.BARPG_CONV)
        # Barras PQ sem gerador convencional (passíveis de déficit)
        self.barras_PQ_sem_gerador = []
        for b in barras_PQ:
            idx = self.idx_map[b["ID_Barra"]]
            if idx not in barras_com_gerador_conv:
                self.barras_PQ_sem_gerador.append(b)

        # Custo do déficit (pode ser um vetor, mas usamos um escalar por simplicidade)
        # Se quiser por barra, pode ser um array do tamanho NBAR
        self.CPG_DEFICIT = 5000.0 * self.SB  # USD/pu

        print(f"  ✓ Déficit: {len(self.barras_PQ_sem_gerador)} barras PQ sem gerador convencional")

    def processa_baterias(self):
        """Processa dados de baterias"""
        self.BARRAS_COM_BATERIA = []

        BATmax_in_base = np.zeros(self.NBAR)
        BATmax_out_base = np.zeros(self.NBAR)
        BATcapacidade_base = np.zeros(self.NBAR)
        BATarm_inicial_base = np.zeros(self.NBAR)
        BATminSoc_base = np.zeros(self.NBAR)

        self.BATTERY_POWER_LIMIT = np.zeros(self.NBAR)
        self.BATTERY_POWER_OUT = np.zeros(self.NBAR)
        self.BATTERY_CAPACITY = np.zeros(self.NBAR)
        self.BATTERY_MIN_SOC = np.zeros(self.NBAR)
        self.BATTERY_INITIAL_SOC = np.zeros(self.NBAR)
        self.BATTERY_COST = np.zeros(self.NBAR)

        for bat in self.baterias_data:
            id_barra = str(bat["ID_Barra"])
            if id_barra not in self.idx_map:
                print(f"  ⚠️  Bateria em barra {id_barra} não encontrada - ignorando")
                continue

            idx = self.idx_map[id_barra]
            self.BARRAS_COM_BATERIA.append(idx)

            if "Pmax_carga_base_pu" in bat:
                p_max_carga = bat["Pmax_carga_base_pu"]
            else:
                p_max_carga = bat.get("Pmax_carga_MW", 0.0) / self.SB

            if "capacidade_base_pu" in bat:
                capacidade = bat["capacidade_base_pu"]
            else:
                capacidade = bat.get("capacidade_armazenamento_MWh", 0.0) / self.SB

            BATmax_in_base[idx] = p_max_carga
            BATmax_out_base[idx] = bat.get("Pmax_descarga_pu", p_max_carga)
            BATcapacidade_base[idx] = capacidade
            BATarm_inicial_base[idx] = bat.get("SOC_inicial_pu", 0.5) * capacidade
            BATminSoc_base[idx] = bat.get("min_soc_pu", 0.1) * capacidade
            self.BATTERY_COST[idx] = bat.get("custo_operacao_pu", 10.0)

        self.BATTERY_POWER_LIMIT = BATmax_in_base.copy()
        self.BATTERY_POWER_OUT = BATmax_out_base.copy()
        self.BATTERY_CAPACITY = BATcapacidade_base.copy()
        self.BATTERY_MIN_SOC = BATminSoc_base.copy()
        self.BATTERY_INITIAL_SOC = BATarm_inicial_base.copy()
        self.BATTERIES = self.BARRAS_COM_BATERIA.copy()

        print(f"  ✓ Baterias processadas: {len(self.BARRAS_COM_BATERIA)} bateria(s) em {len(set(self.BARRAS_COM_BATERIA))} barra(s)")

    def atualizar_perfil_eolico(self, fator_vento: float):
        """
        Atualiza a capacidade eólica efetiva com base no fator de vento.
        PGMAX_EOL_EFETIVO = PGMAX_EOL_ORIGINAL * fator_vento
        PGWIND_disponivel = PGMAX_EOL_EFETIVO (disponibilidade atual)
        """
        self.PGMAX_EOL_EFETIVO = self.PGMAX_EOL_ORIGINAL * fator_vento
        self.PGWIND_disponivel = self.PGMAX_EOL_EFETIVO.copy()
        print(f"  ✓ Perfil eólico atualizado: fator={fator_vento:.3f}")

    def get_sistema_dict(self) -> Dict:
        """Retorna dicionário com todos os dados do sistema processados"""
        return {
            # Parâmetros base
            'SB': self.SB,
            'VB': self.VB,
            'f_base': self.f_base,
            'ZB': self.ZB,

            # Barras
            'bus_ids': self.bus_ids,
            'idx_map': self.idx_map,
            'indice_para_barra': self.indice_para_barra,
            'NBAR': self.NBAR,
            'slack_idx': self.slack_idx,

            # Linhas
            'NLIN': self.NLIN,
            'line_fr': self.line_fr,
            'line_to': self.line_to,
            'x_line': self.x_line,
            'r_line': self.r_line,
            'FLIM': self.FLIM,

            # Geradores convencionais
            'NGER_CONV': self.NGER_CONV,
            'BARPG_CONV': self.BARPG_CONV,
            'GER_TIPOS_CONV': self.GER_TIPOS_CONV,
            'PGMIN_CONV': self.PGMIN_CONV,
            'PGMAX_CONV': self.PGMAX_CONV,
            'CPG_CONV': self.CPG_CONV,
            'RAMP_UP': self.RAMP_UP,
            'RAMP_DOWN': self.RAMP_DOWN,
            'PGER_INICIAL_CONV': self.PGER_INICIAL_CONV,

            # Geradores eólicos
            'NGER_EOL': self.NGER_EOL,
            'BARPG_EOL': self.BARPG_EOL,
            'PGMAX_EOL_ORIGINAL': self.PGMAX_EOL_ORIGINAL,
            'PGMAX_EOL_EFETIVO': self.PGMAX_EOL_EFETIVO,
            'PGWIND_disponivel': self.PGWIND_disponivel,
            'CPG_CURTAILMENT': self.CPG_CURTAILMENT,
            'gwd_idx_to_pos': self.gwd_idx_to_pos,

            # Déficit
            'barras_PQ_sem_gerador': self.barras_PQ_sem_gerador,
            'CPG_DEFICIT': self.CPG_DEFICIT,

            # Carga
            'PLOAD': self.PLOAD,
            'QLOAD': self.QLOAD,

            # Dados originais
            'barras': self.barras,
            'geradores_data': self.geradores_data,
            'demandas_data': self.demandas_data,
            'linhas': self.linhas,
            'json_file_path': self.json_file_path
        }


def load_system(json_file_path: str):
    """
    Função utilitária para carregar e retornar o objeto de sistema já processado.
    Compatível com o uso em scripts principais.
    """
    return SistemaLoader(json_file_path)