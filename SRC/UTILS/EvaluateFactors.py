import numpy as np
import pandas as pd

class EvaluateFactors:
    """
    Gera fatores de carga e vento para simulação multi-período.
    
    Para carga: cada barra tem seu próprio fator, variando aleatoriamente dentro de um intervalo
    definido por um perfil horário base e uma incerteza global.
    
    Para vento: utiliza um arquivo CSV com histórico de fatores de vento (normalizados).
    Para cada período, sorteia um fator base do histórico e aplica uma pequena variação
    individual para cada gerador eólico.
    """
    
    def __init__(self,
                 sistema,
                 n_dias,
                 n_horas, 
                 carga_incerteza=0.2,
                 vento_variacao=0.1,
                 seed=None):
        """
        Args:
            sistema: objeto com atributos NBAR, NGER_EOL, etc.
            n_dias: número de dias
            n_horas: horas por dia (máximo 24)
            carga_incerteza: amplitude da variação uniforme em torno do perfil (ex: 0.2 = ±10%)
            vento_variacao: amplitude da variação relativa para cada gerador (ex: 0.1 = ±5%)
            seed: semente para reprodutibilidade
        """
        self.sistema = sistema
        self.n_dias = n_dias
        self.n_horas = n_horas
        self.T = n_dias * n_horas
        
        # Perfil horário normalizado (pico = 1.0)
        self.carga_perfil_horario = np.array([
            0.65, 0.62, 0.60, 0.60, 0.62, 0.70,  # 0-5h (madrugada)
            0.75, 0.70, 0.80, 0.85, 0.88, 0.90,  # 6-11h (manhã)
            0.92, 0.90, 0.88, 0.90, 0.92, 0.95,  # 12-17h (tarde)
            1.00, 0.98, 0.95, 0.90, 0.75, 0.55   # 18-23h (noite/pico)
        ])
        
        self.carga_incerteza = carga_incerteza
        
        vento_arquivo = "/home/lucasedbraga/repositorios/ufjf/mestrado_luedsbr/SRC/SOLVER/DB/getters/intermittent-renewables-production-france.csv"
        vento_arquivo = r"C:\\Users\\lucas\\repositorios\\gopt-BessWindAgentOperator\\SRC\\DB\\getters\\intermittent-renewables-production-france.csv"
        self.vento_arquivo = vento_arquivo
        self.vento_variacao = vento_variacao
        self.seed = seed
        
        # Carregar dados de vento
        self._load_vento_data()
    
    def _load_vento_data(self):
        """Carrega fatores de vento do CSV."""
        df = pd.read_csv(self.vento_arquivo)
        # Ajuste conforme o formato do arquivo
        df['DateTime'] = pd.to_datetime(df['Date and Hour'].str.slice(stop=-6))
        df = df.sort_values('DateTime')
        df_wind = df[df['Source'] == 'Wind'].copy()
        if len(df_wind) == 0:
            raise ValueError("Nenhum dado de vento encontrado no arquivo.")
        max_prod = df_wind['Production'].max()
        if max_prod > 0:
            df_wind['Factor'] = df_wind['Production'] / max_prod
        else:
            df_wind['Factor'] = 0
        self.vento_fatores_base = df_wind['Factor'].values
    
    def gerar_fatores_carga(self):
        """
        Retorna array de fatores de carga com shape (n_dias, n_horas, NBAR).
        Cada barra tem seu próprio fator, sorteado uniformemente entre
        lim_inf[h] e lim_sup[h] para cada hora h.
        """
        np.random.seed(self.seed)
        n_dias = self.n_dias
        n_horas = self.n_horas
        NBAR = self.sistema.NBAR
        
        # Seleciona apenas as primeiras n_horas do perfil
        perfil_horas = self.carga_perfil_horario[:n_horas]
        
        lim_inf = perfil_horas - self.carga_incerteza/2
        lim_sup = perfil_horas + self.carga_incerteza/2
        
        # Garantir que limites não fiquem negativos
        lim_inf = np.maximum(lim_inf, 0.0)
        
        # Shape final: (dias, horas, barras)
        fatores = np.zeros((n_dias, n_horas, NBAR))
        
        for d in range(n_dias):
            for h in range(n_horas):
                # Sorteia para cada barra independentemente
                fatores[d, h, :] = np.random.uniform(lim_inf[h], lim_sup[h], size=NBAR)
        
        return fatores
    
    def gerar_fatores_vento(self):
        """
        Retorna array de fatores de vento com shape (n_dias, n_horas, NGER_EOL).
        Para cada período, sorteia um fator base do histórico e aplica uma variação
        individual para cada gerador eólico: fator = base * (1 + delta), com delta
        uniforme em [-vento_variacao/2, vento_variacao/2].
        """
        if self.vento_fatores_base is None:
            raise ValueError("Dados de vento não carregados.")
        
        np.random.seed(self.seed)
        n_dias = self.n_dias
        n_horas = self.n_horas
        T = self.T
        NGER_EOL = self.sistema.NGER_EOL
        
        # Amostrar T valores com reposição da base histórica
        indices = np.random.choice(len(self.vento_fatores_base), size=T, replace=True)
        fatores_base = self.vento_fatores_base[indices]  # shape (T,)
        
        # Expandir para (T, NGER_EOL)
        fatores_base = np.tile(fatores_base.reshape(-1, 1), (1, NGER_EOL))
        
        # Aplicar variação individual
        delta = np.random.uniform(-self.vento_variacao/2, self.vento_variacao/2, size=(T, NGER_EOL))
        fatores = fatores_base * (1 + delta)
        
        # Garantir que não fiquem negativos
        fatores = np.maximum(fatores, 0.0)
        
        # Redimensionar para (dias, horas, NGER_EOL)
        fatores = fatores.reshape(n_dias, n_horas, NGER_EOL)
        return fatores
    
    def gerar_tudo(self):
        """
        Retorna tuple (fatores_carga, fatores_vento) com shapes:
        carga: (n_dias, n_horas, NBAR)
        vento: (n_dias, n_horas, NGER_EOL)
        """
        fatores_carga = self.gerar_fatores_carga()
        fatores_vento = self.gerar_fatores_vento()
        return fatores_carga, fatores_vento