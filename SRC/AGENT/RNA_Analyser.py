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

# ------------------------------
# 1. Carregar dados
# ------------------------------
# Use o caminho correto (ajuste se necessário)
arquivo = r"C:\Users\lucas\repositorios\gopt-BessWindAgentOperator\DATA\output\modelos_especialistas_v4\hora_17\previsoes_teste.csv"
df = pd.read_csv(arquivo)


# ------------------------------
# 2. Identificar colunas reais e previstas
# ------------------------------
colunas_reais = [col for col in df.columns if not col.endswith('_previsto') and col != 'correto']
colunas_reais.sort()  # ordenação alfabética inicial (depois reordenaremos)

resultados = []
for real in colunas_reais:
    prev = real + '_previsto'
    if prev not in df.columns:
        continue
    y_true = df[real].values
    y_pred = df[prev].values

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    smape_val = smape(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan

    resultados.append({
        'Variável': real,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'SMAPE (%)': smape_val,
        'Correlação': corr
    })

df_metricas = pd.DataFrame(resultados)


# ------------------------------
# 3. Ordenar na sequência desejada
# ------------------------------
ordem_desejada = [
    'BESS_operation_result_BAR3',
    'BESS_operation_result_BAR14',
    'PLOAD_estimado_BAR2',
    'PLOAD_estimado_BAR4',
    'PLOAD_estimado_BAR6',
    'PLOAD_estimado_BAR9',
    'PLOAD_estimado_BAR10',
    'PLOAD_estimado_BAR11',
    'PLOAD_estimado_BAR12',
    'PLOAD_estimado_BAR13',
    'PLOAD_estimado_BAR14',
    'Correto'
]

# Converter a coluna 'Variável' para categoria com a ordem especificada
df_metricas['Variável'] = pd.Categorical(df_metricas['Variável'], categories=ordem_desejada, ordered=True)
df_metricas = df_metricas.sort_values('Variável').reset_index(drop=True)

# ------------------------------
# 4. Exibir tabela ordenada
# ------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.6f}'.format)

print("\n=== MÉTRICAS DE ACURÁCIA DA RNA (ORDEM ESPECIFICADA) ===\n")
print(df_metricas.to_string(index=False))
# Desvio padrão das variáveis reais
desvios = df[colunas_reais].std().sort_values()
print("Desvio padrão das variáveis reais:")
print(desvios)

# RMSE normalizado (coeficiente de variação do erro)
df_metricas['RMSE/σ'] = df_metricas['RMSE'] / desvios[df_metricas['Variável']].values
print(df_metricas[['Variável', 'RMSE', 'RMSE/σ']])
# ------------------------------
# 5. Gerar gráficos de barras para cada métrica
# ------------------------------
metricas_plot = ['MAE', 'MSE', 'RMSE', 'R²', 'SMAPE (%)', 'Correlação']
x = df_metricas['Variável'].astype(str)  # converter para string para evitar problemas com categoria

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
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'grafico_{metrica.replace(" ", "_").replace("²", "2")}.png', dpi=150)
    plt.show()
    print(f"Gráfico de {metrica} salvo como PNG.")

print("\nTodos os gráficos foram gerados.")