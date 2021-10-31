# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:22:59 2020

@author: saksh
"""
import numpy as np
np.random.seed(1337)
import tensorflow as tf

import pandas as pd
from matplotlib.pyplot import *

from ml_pap import *
from db_managmeant import get_db

matplotlib.pyplot.style.use('classic')

"""
All data imported and scaled according to s&p500
"""

data_cache =get_db()
"""
Run this block to include VAR residuals in input data
"""
for label in data_cache.columns[1:]:
    resids = var_resids('nifty', label, data_cache = data_cache)
    data_cache[str.format("nifty_var_%s"%(label))] = resids 
data_cache.dropna(inplace = True)

"""
Set model targets
"""
data_cache = data_cache[-3090:-90]
data_cache['nifty_volatility'] = np.log(data_cache['nifty']/data_cache['nifty'].shift(1))**2
data_cache.dropna(inplace = True)
data_cache['targets'] = data_cache['nifty_volatility'].shift(-1)
data_cache.dropna(inplace = True)

"""
Split datasets
Use "dense" for SVR predictions
Returns sclaer used for scaling the output variables
"""
X_train, X_test, y_train, y_test, output_scaler = make_datasets(data_cache, model_name = 'dense')

"""
Run model
Inverse transform test targets and predictions
"""
result = svr_model(X_train, y_train, {'C' : [1,10]})
y_pred = result.predict(X_test)
y_test = y_test.reshape(len(y_test), 1)
y_pred = y_pred.reshape(len(y_pred), 1)
y_pred = output_scaler.inverse_transform(y_pred)
y_test = output_scaler.inverse_transform(y_test)

"""
RMSE of inverse transformed variables
"""
m = tf.metrics.RootMeanSquaredError()
m.update_state(y_test, np.abs(y_pred))
transformed = m.result().numpy()
print("RMSE of transformed variables: %d", transformed)

df_plot = make_save_plot(data_cache.index, y_test, y_pred)