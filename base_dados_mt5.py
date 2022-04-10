#Importando Bibliotecas
import MetaTrader5 as mt5
from datetime import datetime
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion
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


def get_ohlc(ativo, timeframe, n = 10):
    ativo = mt5.copy_rates_from_pos(ativo, timeframe, 0, n)
    ativo = pd.DataFrame(ativo)
    ativo['time'] = pd.to_datetime(ativo['time'], unit = 's')
    ativo.set_index('time', inplace = True)
    return ativo
wdo_m1 = get_ohlc('WDO$', mt5.TIMEFRAME_M1,99999)
dol_1m = get_ohlc('DOL$', mt5.TIMEFRAME_M1,99999)
wdo_m1.to_csv('WDO.CSV')
#dol_1m.to_csv('dol.CSV')
cotacao = pd.read_csv('WDO.CSV', index_col = 0)
print(cotacao)


#Passo 2: Coletar os dados da ativo
def get_ohlc(ativo, timeframe, n = 10):
    ativo = mt5.copy_rates_from_pos(ativo, timeframe, 0, n)
    ativo = pd.DataFrame(ativo)
    ativo['time'] = pd.to_datetime(ativo['time'], unit='s')
    ativo.set_index('time', inplace=True)
    return ativo
#Dados do ativo
wdo_m1 = get_ohlc('WDO$', mt5.TIMEFRAME_M1,99999)
tabela = wdo_m1
#print(tabela)
tabela_aj = tabela
#tabela_aj['time'] = pd.to_datetime(tabela_aj['time'], unit='s')
#tabela_aj.set_index('time', inplace=True)
tabela_aj = tabela_aj.drop(["tick_volume", "spread", "real_volume"], axis=1)
tabela_aj.loc[:,'open/high'] = (tabela_aj['high'] - tabela_aj['open'])/1
tabela_aj.loc[:,'open/low'] = (tabela_aj['low'] - tabela_aj['open'])/1
tabela_aj.loc[:,'open/close'] = (tabela_aj['close'] - tabela_aj['open'])/1
tabela_aj.loc[:,'close/high'] = (tabela_aj['high'] - tabela_aj['close'])/1
tabela_aj.loc[:,'close/low'] = (tabela_aj['low'] - tabela_aj['close'])/1
tabela_aj.loc[:,'close/open'] = (tabela_aj['open'] - tabela_aj['close'])/1
#tabela Aurea
tabela_aj.loc[:,'open/a'] = (tabela_aj['open'] * 0.618)
tabela_aj.loc[:,'high/a'] = (tabela_aj['high'] * 0.618)
tabela_aj.loc[:,'low/a'] = (tabela_aj['low'] * 0.618)
tabela_aj.loc[:,'close/a'] = (tabela_aj['close'] * 0.618)
#tabela_aj.loc[:,'high/aa'] = (tabela_aj['high/a'] * 1.618)
#tabela_aj.loc[:,'low/aa'] = (tabela_aj['low/a'] * 1.618)
#tabela_aj.loc[:,'close/aa'] = (tabela_aj['close/a'] * 1.618)
#ponto Aurea
tabela_aj.loc[:,'pa'] = ((tabela_aj['high'] + tabela_aj['low'])/tabela_aj['high'])
#print(tabela_aj)
'''
#Separando dados de treino e dados de teste
#os dados a serem separados
x = tabela_aj.drop(["close"], axis=1)#.drop(["tick_volume", "spread", "real_volume"], axis=1)#informacao prever
#x = tabela_aj.drop("open", axis = 1)
y = tabela_aj['close']
#dados de treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)
#new = [[5666.500]] #novo dado a ser previsto

#Inteligencia Artificial
#Floresta de decisao
floresta = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=0, n_jobs=-1)
floresta.fit(x_treino, y_treino)

#Regressao linear
modelo_regressaolinear = LinearRegression()
modelo_regressaolinear.fit(x_treino, y_treino)

#Teste de Avaliacao do modelo
#criar as previsoes
p_f = floresta.predict(x_teste)
p_r = modelo_regressaolinear.predict(x_teste)
new = [[5666.500]] #novo dado a ser previsto
new = floresta.predict(new)
# modelo_regressaolinear.predict(new)
mse_p_f = np.sqrt(mean_squared_error(y_teste, p_f))
mse_p_r = np.sqrt(mean_squared_error(y_teste, p_r))
#mse_xn = np.sqrt(mean_squared_error(y_teste, new))
#motro os valores
#print(p_f)
#print(p_r)
print(f'Floresta decisao {mse_p_f}%')
print(f'Regressao Linear {mse_p_r}%')
#print(new)


#print(p)
#print(mse_xn)
#print(x)
#print(y_teste)

#print(xnew)
#10 - 5665.98017316
#100- 5666.03089155
#500- 5665.94006256
#1000-5665.95785838





#Inteligencia Artificial
#Crio a Inteligencia artificial
#modelo_regressaolinear = LinearRegression()
#modelo_arvoredecisao = RandomForestRegressor()

#Treino Inteligencia artificial
#FIT, quem treina a Inteligencia Artificial
#modelo_regressaolinear.fit(x_treino, y_treino)
#modelo_arvoredecisao.fit(x_treino, y_treino)

#Teste de Avaliacao do melhor modelo de AI
#criar as previsoes
#previsao_regrecaolinear = modelo_regressaolinear.predict(x_teste)
#previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

#Avaliacao conjunto dados
#reg_predict = previsao_regrecaolinear #modelo_regressaolinear.predict(x_teste)
#reg_mse = mean_squared_error(y_teste, reg_predict)#6.603658556881671e-25
#reg_mse1 = np.sqrt(reg_mse)
#comparar os modelos
#print(reg_mse)
#print(reg_mse1)
#print('***********')
#print('* Modelos *')
#print('***********')
#print(f'Score Regressao linaer:', (metrics.r2_score(y_teste, previsao_regrecaolinear)), '%')
#print(f'Score Floresta de Decisao:', (metrics.r2_score(y_teste, previsao_arvoredecisao)), '%')
#print(f'Score Avaliacao Conj. Test. Regressao Linear:', (reg_mse), '%')
#print(f' reg:{previsao_regrecaolinear}')
#print(f' arv:{previsao_arvoredecisao}')
#print(f'pred:{reg_predict}')
#print(f' reg:{reg_mse}')

#Separando dados de treino e dados de teste
#os dados a serem separados

'''

'''
#.......>>>>>>>>>>>>>
#Teste de Avaliacao do melhor modelo de AI
#criar as previsoes
previsao_regrecaolinear = modelo_regressaolinear.predict(x_teste)
previsao_ForestRegressor = modelo_arvoredecisao.predict(x_teste)

#Avaliacao conjunto dados
#Regressao linear
reg_predict = previsao_regrecaolinear #modelo_regressaolinear.predict(x_teste)
reg_mse = mean_squared_error(y_teste, reg_predict)
reg_mse = np.sqrt(reg_mse)
#Floresta decisao
reg_predict1 = previsao_arvoredecisao #modelo_ForestRegressor.predict(x_teste)
reg_mse1 = mean_squared_error(y_teste, reg_predict1)
reg_mse1 = np.sqrt(reg_mse1)

#comparar os modelos
#print(f' Regressao Linerar:{metrics.r2_score(y_teste, previsao_regrecaolinear)}%')
#print(f' Floresta Decisao:{metrics.r2_score(y_teste, previsao_ForestRegressor)}%')
#print('*' *20)
print(previsao_regrecaolinear)
print(previsao_ForestRegressor)
print(reg_mse)
print(reg_mse1)
'''
'''
#Visualizacoes das previsoes
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsao Regrecaolinear"] = previsao_regrecaolinear
tabela_auxiliar["Previsao ForestRegressor"] = previsao_ForestRegressor
print(tabela_auxiliar)
'''
'''
#Avaliacao conjunto dados
#reg_predict = previsao_regrecaolinear #modelo_regressaolinear.predict(x_teste)
#reg_mse = mean_squared_error(y_teste, reg_predict)
#reg_mse = np.sqrt(reg_mse)
'''
'''
plt.figure(figsize = (15, 6))
sns.lineplot(data = tabela_auxiliar)
plt.show()

sns.barplot(x=x_treino.columns, y = modelo_ForestRegressor.feature_importances_)
plt.show()
'''
