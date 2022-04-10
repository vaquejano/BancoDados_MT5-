#Importando Bibliotecas
import MetaTrader5 as mt5
import pandas as pd

#import matplotlib.pyplot as plt

#Setando as colunas das planilhas
pd.set_option('display.max_columns', 400) # número de colunas mostradas
pd.set_option('display.width', 1500)      # max. largura máxima da tabela exibida

#Passo 1: Conectar ao sistema Meta Trader 5
# conecte-se ao MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()
print('******************')
print('*   conectado    *')
print('******************')

