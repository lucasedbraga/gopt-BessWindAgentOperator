from pyomo.environ import ConcreteModel, Block, Var, Constraint, NonNegativeReals, Objective, minimize, ConstraintList
from pyomo.opt import SolverFactory
from OPT.DC_OPF_Model import DC_OPF_Model
import numpy as np
from SOLVER.FOB.economic_dispatch import DC_OPF_EconomicDispatch_Solver
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class MultiDayOPFSnapshotResult:
    dia: int
    hora: int
    sucesso: bool

    PGER: List[float] = field(default_factory=list)
    PGWIND: List[float] = field(default_factory=list)
    CURTAILMENT: List[float] = field(default_factory=list)
    SOC_init: List[float] = field(default_factory=list)
    BESS_operation: List[float] = field(default_factory=list)
    SOC_atual: List[float] = field(default_factory=list)
    DEFICIT: List[float] = field(default_factory=list)

    V: List[float] = field(default_factory=list)
    ANG: List[float] = field(default_factory=list)
    FLUXO_LIN: List[float] = field(default_factory=list)

    CUSTO: List[float] = field(default_factory=list)
    CMO: List[float] = field(default_factory=list)
    PERDAS: List[float] = field(default_factory=list)

    mensagem: str = ""
    timestamp: Optional[datetime] = None
    tempo_execucao: float = 0.0



@dataclass
class MultiDayOPFResult:
    snapshots: List[MultiDayOPFSnapshotResult] = field(default_factory=list)
    sucesso_global: bool = True
    mensagem_global: str = ""

class MultiDayOPFModel:
    def __init__(self, sistema, n_horas=24, n_dias=1, db_handler=None):
        self.sistema = sistema
        self.n_horas = n_horas
        self.n_dias = n_dias
        self.db_handler = db_handler  # Handler opcional para salvar resultados
        self.model = None  # Só será criado em build()
        self._solved = False
        self._raw_results = None

    def build(self):
        """Constrói o modelo multi-dia/hora sem acoplamento temporal"""
        self.model = ConcreteModel()
        self.model.snapshots = Block(range(self.n_dias), rule=self._snapshot_block_rule)
        self._add_temporal_coupling()
        self._add_objective()
        self._solved = False
        self._raw_results = None

    def _snapshot_block_rule(self, b, dia):
        b.hours = Block(range(self.n_horas), rule=lambda bh, h: self._build_hour_block(bh, h, dia))

    def _build_hour_block(self, bh, h, dia):
        # Instancia o modelo DC_OPF para cada hora
        opf = DC_OPF_Model()
        opf.build(self.sistema, considerar_perdas=True)
        # Adiciona função objetivo via economic_dispatch
        solver = DC_OPF_EconomicDispatch_Solver(self.sistema)
        solver.add_objective(opf)
        # Desativa objetivo do submodelo para evitar múltiplos objetivos ativos
        if hasattr(opf.model, 'obj'):
            opf.model.obj.deactivate()
        bh.opf = opf.model
        # Adiciona variáveis de SOC para acoplamento
        bh.SOC = Var(range(len(self.sistema.BARRAS_COM_BATERIA)), within=NonNegativeReals)

    def _add_temporal_coupling(self):
        m = self.model
        # Cria restrições de acoplamento temporal para todas as baterias, horas e dias
        m.soc_coupling = ConstraintList()
        for dia in range(self.n_dias):
            for h in range(self.n_horas-1):
                for b_idx in range(len(self.sistema.BARRAS_COM_BATERIA)):
                    block_curr = m.snapshots[dia].hours[h]
                    block_next = m.snapshots[dia].hours[h+1]
                    m.soc_coupling.add(block_next.SOC[b_idx] == block_curr.SOC[b_idx])
            if dia < self.n_dias-1:
                for b_idx in range(len(self.sistema.BARRAS_COM_BATERIA)):
                    block_last = m.snapshots[dia].hours[self.n_horas-1]
                    block_first_next = m.snapshots[dia+1].hours[0]
                    m.soc_coupling.add(block_first_next.SOC[b_idx] == block_last.SOC[b_idx])

    def _add_objective(self):
        m = self.model
        # Soma os objetivos de cada bloco (hora/dia)
        def total_obj_rule(_m):
            return sum(
                getattr(m.snapshots[dia].hours[h].opf, 'obj', 0)
                for dia in range(self.n_dias)
                for h in range(self.n_horas)
            )
        m.total_obj = Objective(rule=total_obj_rule, sense=minimize)

    def solve(self, solver_name='glpk', cen_id=None):
        """Resolve o modelo multi-dia/hora (acoplado) e opcionalmente salva resultados no banco"""
        from pyomo.opt import SolverFactory
        if self.model is None:
            raise RuntimeError("O modelo deve ser construído com build() antes de resolver.")
        solver = SolverFactory(solver_name)
        self._raw_results = solver.solve(self.model, tee=False)
        self._solved = True

        # Se um handler e cen_id forem fornecidos, salva os resultados
        if self.db_handler is not None and cen_id is not None:
            resultados = self.extract_results()
            for snap in resultados.snapshots:
                dia_str = f"{snap.dia+1}"
                perfil_carga = 1.0
                perfil_vento = 1.0
                self.db_handler.save_hourly_result(
                    resultado=snap,
                    sistema=self.sistema,
                    hora=snap.hora,
                    perfil_carga=perfil_carga,
                    perfil_eolica=perfil_vento,
                    solver_name=solver_name,
                    dia=dia_str,
                    cen_id=cen_id
                )
        return self._raw_results
    
    def atualizar_soc_inicial(self, soc_atual):
        """
        Atualiza o SOC inicial do sistema com base nos valores atuais das baterias.
        
        Args:
            soc_atual (list ou array): Valores de SOC para cada bateria na mesma ordem de BARRAS_COM_BATERIA.
        """
        import numpy as np
        soc_atual = np.array(soc_atual)
        barras_bateria = self.sistema.BARRAS_COM_BATERIA  # lista de índices
        novo_soc = np.zeros_like(self.sistema.BATTERY_MIN_SOC)
        novo_soc[barras_bateria] = soc_atual
        return novo_soc

    def solve_multiday_sequencial(self, solver_name='glpk', fator_carga=None, fator_vento=None,
                                  soc_inicial=None, cen_id=None):
        """
        Resolve cada hora/dia separadamente, conectando via SOC da bateria.
        Para cada simulação, aplica os fatores de carga e vento fornecidos.
        Se um db_handler e cen_id forem fornecidos, salva cada snapshot no banco.
        """
        resultados = []
        n_bat = len(self.sistema.BARRAS_COM_BATERIA)
        self.sistema.SOC_init = soc_inicial if soc_inicial is not None else [0.0]*n_bat
        # Salva os valores originais para restaurar depois
        PLOAD_original = self.sistema.PLOAD.copy()
        PGMAX_EFETIVO_original = self.sistema.PGMAX_EFETIVO.copy() if hasattr(self.sistema, 'PGMAX_EFETIVO') else None

        for dia in range(self.n_dias):
            for h in range(self.n_horas):
                # Atualiza carga para o valor dessa hora/dia
                if fator_carga is not None:
                    self.sistema.PLOAD = PLOAD_original * fator_carga[dia, h]
                # Atualiza vento para o valor dessa hora/dia
                if fator_vento is not None:
                    for g_idx in self.sistema.BAR_GWD:
                        self.sistema.PGMAX_EFETIVO[g_idx] = self.sistema.PGMAX[g_idx] * fator_vento[dia, h]
                        
                opf = DC_OPF_Model()
                opf.build(self.sistema, considerar_perdas=True)
                solver = DC_OPF_EconomicDispatch_Solver(self.sistema)
                solver.add_objective(opf)

                solver_pyomo = SolverFactory(solver_name)
                results_pyomo = solver_pyomo.solve(opf.model, tee=False)

                res = opf.extract_results(results_pyomo)
                soc_atual = res.SOC_atual.copy() if res.SOC_atual else [0.0]*n_bat
                # Atualiza SOC inicial
                self.sistema.SOC_init = soc_atual

                # Cria o snapshot de resultado
                snapshot = MultiDayOPFSnapshotResult(
                    dia=dia,
                    hora=h,
                    sucesso=res.sucesso,

                    PGER=res.PGER,
                    PGWIND=res.PGWIND,
                    CURTAILMENT=res.CURTAILMENT,
                    SOC_init=res.SOC_init,
                    BESS_operation=res.BESS_operation,
                    SOC_atual=res.SOC_atual,
                    DEFICIT=res.DEFICIT,

                    V=res.V,
                    ANG=res.ANG,
                    FLUXO_LIN= res.FLUXO_LIN,
                   
                    CUSTO=res.CUSTO,
                    CMO=res.CMO,
                    PERDAS=res.PERDAS,

                    mensagem=res.mensagem,
                    timestamp=res.timestamp,
                    tempo_execucao=getattr(res, 'tempo_execucao', 0.0),

                )
                resultados.append(snapshot)

                # Salva no banco se o handler estiver configurado
                if self.db_handler is not None:
                    if cen_id is None:
                        raise ValueError("cen_id é obrigatório quando db_handler está configurado")
                    dia_str = f"{dia+1}"
                    self.db_handler.save_hourly_result(
                        resultado=snapshot,
                        sistema=self.sistema,
                        hora=h,
                        perfil_carga=fator_carga[dia, h] if fator_carga is not None else 1.0,
                        perfil_eolica=fator_vento[dia, h] if fator_vento is not None else 1.0,
                        solver_name=solver_name,
                        dia=dia_str,
                        cen_id=cen_id
                    )

        # Restaura os valores originais do sistema
        self.sistema.PLOAD = PLOAD_original
        if PGMAX_EFETIVO_original is not None:
            self.sistema.PGMAX_EFETIVO = PGMAX_EFETIVO_original

        sucesso_global = all(r.sucesso for r in resultados)
        mensagem_global = "OK" if sucesso_global else "Alguns snapshots falharam"
        return MultiDayOPFResult(snapshots=resultados, sucesso_global=sucesso_global, mensagem_global=mensagem_global)

    def extract_results(self) -> MultiDayOPFResult:
        """Extrai resultados estruturados de todos os snapshots (dia/hora)"""
        if not self._solved or self.model is None:
            raise RuntimeError("O modelo deve ser resolvido antes de extrair resultados.")
        resultados = []
        for dia in range(self.n_dias):
            for h in range(self.n_horas):
                block = self.model.snapshots[dia].hours[h]
                opf_model = block.opf
                try:
                    PGER = [float(opf_model.PG[g].value) for g in opf_model.GENERATORS] if hasattr(opf_model, 'PG') else []
                    PGWIND = [float(opf_model.PG[g].value) for g in opf_model.BAR_GWD] if hasattr(opf_model, 'PG') else []
                    CURTAILMENT = [float(opf_model.CURTAILMENT[g].value) for g in opf_model.GWD_GENERATORS] if hasattr(opf_model, 'CURTAILMENT') else []
                    SOC_init = [self.sistema.BATTERY_INITIAL_SOC[idx] if idx < len(self.sistema.BATTERY_INITIAL_SOC) else 0.0 for idx in range(len(self.sistema.BARRAS_COM_BATERIA))]
                    BATTERY_OPERATION = []
                    SOC_atual = [block.SOC[idx].value if hasattr(block, 'SOC') else None for idx in range(len(self.sistema.BARRAS_COM_BATERIA))]
                    DEFICIT = [float(opf_model.DEFICIT[b].value) for b in opf_model.BUSES] if hasattr(opf_model, 'DEFICIT') else []
                    
                    V = [float(opf_model.V[b].value) for b in opf_model.BUSES] if hasattr(opf_model, 'V') else []
                    ANG = [float(opf_model.ANG[b].value) for b in opf_model.BUSES] if hasattr(opf_model, 'ANG') else []
                    FLUXO = [float(opf_model.FLUXO[e].value) for e in opf_model.LINES] if hasattr(opf_model, 'FLUXO') else []
                   
                    CUSTO = []
                    CMO = []
                    PERDAS = []

                    for idx in range(len(self.sistema.BARRAS_COM_BATERIA)):
                        charge = opf_model.CHARGE[idx].value if hasattr(opf_model, 'CHARGE') else 0.0
                        discharge = opf_model.DISCHARGE[idx].value if hasattr(opf_model, 'DISCHARGE') else 0.0
                        pot_inst_bess = charge - discharge
                        BATTERY_OPERATION.append(pot_inst_bess)


                    sucesso = True
                    mensagem = ""
                except Exception as e:
                    PGER = PGWIND = DEFICIT = CURTAILMENT = SOC_init = SOC_atual = BATTERY_OPERATION = []
                    V = ANG = FLUXO_LIN = []
                    CUSTO = CMO = PERDAS = []
                    sucesso = False
                    mensagem = str(e)

                resultados.append(
                    MultiDayOPFSnapshotResult(
                        dia=dia,
                        hora=h,
                        sucesso=sucesso,

                        PGER=PGER,
                        PGWIND=PGWIND,
                        CURTAILMENT=CURTAILMENT,
                        SOC_init=SOC_init,
                        BESS_operation=BATTERY_OPERATION,
                        SOC_atual=SOC_atual,
                        DEFICIT=DEFICIT,
                    
                        V=V,
                        ANG=ANG,
                        FLUXO_LIN=FLUXO_LIN,

                        CUSTO=CUSTO,
                        CMO=CMO,
                        PERDAS=PERDAS,

                        mensagem=mensagem,
                        timestamp=datetime.now(),
                        tempo_execucao=0.0,
                        )
                    )

        sucesso_global = all(r.sucesso for r in resultados)
        mensagem_global = "OK" if sucesso_global else "Alguns snapshots falharam"
        return MultiDayOPFResult(snapshots=resultados, sucesso_global=sucesso_global, mensagem_global=mensagem_global)