import sqlite3
import csv
import os

# Defina aqui o caminho do seu banco SQLite
caminho_banco = "/home/lucasedbraga/repositorios/ufjf/gopt-BessWindAgentOperator/DATA/output/resultados_snapshot.db"

def exportar_tabela(conexao, nome_tabela, arquivo_saida):
    cursor = conexao.cursor()
    cursor.execute(f"SELECT * FROM {nome_tabela}")
    linhas = cursor.fetchall()
    if not linhas:
        print(f"A tabela '{nome_tabela}' está vazia. Nenhum dado exportado.")
        return
    nomes_colunas = [descricao[0] for descricao in cursor.description]
    with open(arquivo_saida, 'w', newline='', encoding='utf-8') as arquivo_csv:
        escritor = csv.writer(arquivo_csv)
        escritor.writerow(nomes_colunas)
        escritor.writerows(linhas)
    print(f"Tabela '{nome_tabela}' exportada para: {arquivo_saida}")

def listar_tabelas(conexao):
    cursor = conexao.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [linha[0] for linha in cursor.fetchall()]

# Conecta ao banco
conexao = sqlite3.connect(caminho_banco)
tabelas = listar_tabelas(conexao)
if not tabelas:
    print("O banco de dados não contém tabelas.")
else:
    for tabela in tabelas:
        arquivo_csv = os.path.join(os.path.dirname(caminho_banco), f"{tabela}.csv")
        exportar_tabela(conexao, tabela, arquivo_csv)

conexao.close()