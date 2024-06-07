# --- https://cienciadedatos.net/documentos/py27-forecasting-series-temporales-python-scikitlearn
from flask import Flask, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
# Modelado y Forecasting
# ==============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import skforecast
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster
import shap

app = Flask(__name__)

path='E:/Job/ventas_ANDI_Complete_Hist_Just_MonturaCD.xlsx'
datos=pd.read_excel(path, sheet_name='Sheet1',usecols=['fecha','Tot_Bill'])
datos.dropna(subset=['Tot_Bill'],inplace=True)
datos['fecha'] = pd.to_datetime(datos['fecha'])
datos = datos.groupby(['fecha']).Tot_Bill.sum().reset_index()
#print(datos.count())
datos['fecha'] = pd.to_datetime(datos['fecha'])
datos = datos.set_index(datos['fecha'])
datos = datos.asfreq(freq='D')
groups_means = datos['Tot_Bill'].mean()
datos['Tot_Bill'] = datos['Tot_Bill'].fillna(groups_means)

steps = 90
steps2 = 90
datos_train = datos[:-steps]
datos_test  = datos[-steps2:]
print(f"Fechas train : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)})")
print(f"Fechas test  : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)})")

@app.route('/data')
def send_data():
    return jsonify({'data1': datos_train.to_dict(orient='records'), 'data2': datos_test.to_dict(orient='records')})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')