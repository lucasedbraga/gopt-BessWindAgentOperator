import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (SMAPE) em porcentagem."""
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    if not mask.any():
        return 0.0
    return 100 * np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])

def hit_rate_mixed(y_true, y_pred, rel_tol=0.05, abs_tol=0.01):
    """
    Retorna array booleano indicando acerto por elemento.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    abs_err = np.abs(y_true - y_pred)
    nonzero = y_true != 0
    correct = np.zeros_like(y_true, dtype=bool)
    if np.any(nonzero):
        rel_err = abs_err[nonzero] / np.abs(y_true[nonzero])
        correct[nonzero] = rel_err <= rel_tol
    zero = ~nonzero
    if np.any(zero):
        correct[zero] = abs_err[zero] <= abs_tol
    n_correct = np.sum(correct)
    percent = (n_correct / len(correct)) * 100.0 if len(correct) > 0 else 0.0
    return n_correct, percent

# ------------------------------
# 1. Carregar dados
# ------------------------------
#arquivo = r"C:\Users\lucas\repositorios\gopt-BessWindAgentOperator\DATA\output\modelos_especialistas_v6\hora_18\previsoes_teste.csv"
arquivo = r"C:\Users\lucas\repositorios\gopt-BessWindAgentOperator\DATA\output_CUR_Oficial\modelos_especialistas_v7\hora_16\previsoes_teste.csv"

df = pd.read_csv(arquivo)

# ------------------------------
# 2. Identificar colunas reais e previstas
# ------------------------------
colunas_reais = [col for col in df.columns if not col.endswith('_previsto') and col != 'correto']

# ------------------------------
# 3. Filtrar apenas as colunas que estão na ordem desejada (opcional)
# ------------------------------
ordem_desejada = [
    # 'BESS_operation_result_BAR3',
    # 'BESS_operation_result_BAR14',
    "CURTAILMENT_total_result_BAR3",
    "CURTAILMENT_total_result_BAR14",
    "CURTAILMENT_total_result_BAR8",
    "CURTAILMENT_total_result_BAR73",
    "CURTAILMENT_total_result_BAR111",
    "PLOAD_medido_BAR2",
    "PLOAD_medido_BAR3",
    "PLOAD_medido_BAR4",
    "PLOAD_medido_BAR6",
    "PLOAD_medido_BAR9",
    "PLOAD_medido_BAR14",
    "LIN_usage_result_4-7",
    "LIN_usage_result_4-9",
    "LIN_usage_result_5-6"
]
colunas_reais = [c for c in colunas_reais if c in ordem_desejada]  # manter apenas as desejadas

resultados = []
for real in colunas_reais:
    prev = real + '_previsto'
    if prev not in df.columns:
        continue
    y_true = df[real].values
    y_pred = df[prev].values

    # Cálculo das métricas
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    smape_val = smape(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 and np.std(y_true) > 0 else np.nan
    n_acertos, pct_acertos = hit_rate_mixed(y_true, y_pred, rel_tol=0.08, abs_tol=0.009)

    resultados.append({
        'Variável': real,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'SMAPE (%)': smape_val,
        'Correlação': corr,
        'Acertos (%)': pct_acertos,
        'Acertos (contagem)': n_acertos,
        'Total amostras': len(y_true)
    })

df_metricas = pd.DataFrame(resultados)

colunas_bess = [col for col in colunas_reais if col.startswith('BESS_operation_result')]

if colunas_bess:
    # Inicializar listas para armazenar os vetores reais e previstos
    y_true_bess = []
    y_pred_bess = []
    
    for real in colunas_bess:
        prev = real + '_previsto'
        if prev in df.columns:
            y_true_bess.extend(df[real].values)
            y_pred_bess.extend(df[prev].values)
        else:
            print(f"Aviso: coluna prevista para {real} não encontrada. Ignorando.")
    
    if y_true_bess:
        y_true_bess = np.array(y_true_bess)
        y_pred_bess = np.array(y_pred_bess)
        
        # Cálculo do EEE (Euclidean Error Estimate)
        eee = np.sqrt(np.sum((y_true_bess - y_pred_bess) ** 2)) * 100
        
        # Cálculo do EMT (Maximum Relative Error) – evita divisão por zero
        abs_real = np.abs(y_true_bess)
        abs_diff = np.abs(y_true_bess - y_pred_bess)
        # Proteção para real == 0: considera erro relativo = 0 se real=0 
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_errors = np.where(abs_real != 0, abs_diff / abs_real, 0)
            
        
        emt = np.max(rel_errors) * 100        
        # Cálculo do EAF (Absolute Error Factor)
        eaf = np.max(abs_diff) * 100        
        # IGE = soma
        ige = eee + emt + eaf
        
        # Exibir resultados
        print("\n=== MÉTRICAS COMPOSTAS PARA BESS_operation_result ===\n")
        print(f"EEE : {eee:.6f}")
        print(f"EMT : {emt:.6f}")
        print(f"EA  : {eaf:.6f}")
        print(f"IGE = EEE + EMT + EAF          : {ige:.6f}")
        
        # Opcional: salvar em um DataFrame separado
        df_metricas_bess = pd.DataFrame({
            'Métrica': ['EEE', 'EMT', 'EAF', 'IGE'],
            'Valor': [eee, emt, eaf, ige]
        })
        print("\nTabela resumo:")
        print(df_metricas_bess.to_string(index=False))
        
    else:
        print("Nenhum dado de BESS encontrado para cálculo das métricas compostas.")
else:
    print("Nenhuma coluna de BESS_operation_result encontrada no DataFrame.")

# ------------------------------
# 4. Ordenar conforme a ordem desejada (já filtramos, apenas reordenar)
# ------------------------------
df_metricas['Variável'] = pd.Categorical(df_metricas['Variável'], categories=ordem_desejada, ordered=True)
df_metricas = df_metricas.sort_values('Variável').reset_index(drop=True)

# ------------------------------
# 5. Exibir tabela ordenada
# ------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.6f}'.format)

print("\n=== MÉTRICAS DE ACURÁCIA DA RNA (tolerância mista: 5% rel. ou 0,01 abs.) ===\n")
print(df_metricas.to_string(index=False))

# Desvio padrão das variáveis reais (somente as que estão na tabela)
desvios = df[colunas_reais].std().sort_values()
print("\nDesvio padrão das variáveis reais:")
print(desvios)

# RMSE normalizado (coeficiente de variação do erro) – evita divisão por zero
df_metricas['RMSE/σ'] = df_metricas.apply(
    lambda row: row['RMSE'] / desvios[row['Variável']] if desvios[row['Variável']] > 0 else np.nan,
    axis=1
)
print("\nRMSE / σ:")
print(df_metricas[['Variável', 'RMSE', 'RMSE/σ']])

# ------------------------------
# 6. Gerar gráficos de barras para cada métrica (opcional: incluir contagem)
# ------------------------------
metricas_plot = ['MAE', 'MSE', 'RMSE', 'R²', 'SMAPE (%)', 'Correlação', 'Acertos (%)']
x = df_metricas['Variável'].astype(str)

for metrica in metricas_plot:
    plt.figure(figsize=(12, 6))
    bars = plt.bar(x, df_metricas[metrica], color='skyblue', edgecolor='black')
    plt.title(f'{metrica} por Variável', fontsize=14)
    plt.xlabel('Variável')
    plt.ylabel(metrica)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adicionar valores sobre as barras
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    # Salva o gráfico sem exibir (opcional: comente a linha abaixo para visualizar)
    plt.savefig(f'grafico_{metrica.replace(" ", "_").replace("²", "2").replace("%", "pct")}.png', dpi=150)
    plt.close()  # fecha a figura para não acumular memória
    print(f"Gráfico de {metrica} salvo como PNG.")

# Gráfico adicional da contagem de acertos
plt.figure(figsize=(12, 6))
bars = plt.bar(x, df_metricas['Acertos (contagem)'], color='lightgreen', edgecolor='black')
plt.title('Contagem de Acertos por Variável', fontsize=14)
plt.xlabel('Variável')
plt.ylabel('Número de acertos')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('grafico_Acertos_contagem.png', dpi=150)
plt.close()
print("Gráfico de contagem de acertos salvo como PNG.")

print("\nTodos os gráficos foram gerados e salvos.")

# ------------------------------
# 7. Gráficos de linha: Real vs Previsto para cada variável
# ------------------------------
print("\nGerando gráficos de linha Real vs Previsto para cada variável...")

for real in colunas_reais:
    prev = real + '_previsto'
    if prev not in df.columns:
        print(f"  Aviso: coluna prevista para {real} não encontrada, ignorando.")
        continue

    y_true = df[real].values
    y_pred = df[prev].values
    indices = range(len(y_true))  # número da amostra

    plt.figure(figsize=(12, 6))
    plt.plot(indices, y_true, label='OPF', color='blue', linewidth=1.5)
    plt.plot(indices, y_pred, label='RNA', color='orange', linestyle='--', linewidth=1.5)
    plt.title(f'CURTAILMENT - BAR {real[-3:]} (18h)', fontsize=14)
    plt.xlabel('Número da amostra')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Salva o gráfico
    nome_arquivo = f'grafico_linha_{real}.png'
    plt.savefig(nome_arquivo, dpi=150)
    plt.close()
    print(f"  Gráfico salvo: {nome_arquivo}")

print("Todos os gráficos de linha foram gerados e salvos.")