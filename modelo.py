#%% Definir librerías
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import seaborn as sns
import numpy as np

sns.set_theme()
#%% Cargar los datos
path = 'datos/energy_dataset.csv'
dataf = pd.read_csv(path)

#%% Convertir fecha a formato datetime para posterior análisis
dataf['time'] = pd.to_datetime(dataf['time'], utc=True, infer_datetime_format=True) 

dataf.dropna(inplace=True, axis=1, thresh=2)
dataf.dropna(inplace=True)

#%% Crear columnas de mes y año
dataf['Año'] = dataf.time.dt.year
dataf['Mes'] = dataf.time.dt.month

#%% Sacar media de precio energético y agrupar por tupla año-mes 
datos_arima = dataf.groupby(['Año','Mes'])['price actual'].mean()

#%%
X = datos_arima.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
#%%
plt.plot(test, label='Real', linewidth=2)
plt.plot(predictions, color='red', linewidth=2, linestyle='-.', label='Predicción')
plt.legend()
plt.title('Modelo ARIMA')
plt.ylabel('Precio [$USD]')
plt.xlabel('Tiempo (Año, Mes)')
plt.xticks(np.arange(0, 18, step=2), datos_arima.index[::6], rotation=40)
plt.show()
#%% Correlacion

dataf[['total load actual', 'price actual']].corr()

#%%
plt.matshow(dataf[['total load actual', 'price actual']].corr())
plt.show()