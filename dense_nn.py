# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:30:16 2020

@author: saksh
"""
import numpy as np
np.random.seed(1337)
import tensorflow as tf
import pickle
import pandas as pd
from matplotlib.pyplot import *
from params import test_folder,model_folder,pickle_folder
from ml_pap import *
from db_managmeant import db_pre_process,save_test_variables
matplotlib.pyplot.style.use('classic')

"""
All data imported and scaled according to s&p500
"""
# path0="/Users/natankatz/PycharmProjects/pythonProject/Haggai_code/volatility_prediction_study-master/data/"
# sp500 = pd.read_csv(path0+'^GSPC.csv', header = 0, index_col = 'Date')
# sp500.index = pd.to_datetime(sp500.index, format = '%d-%m-%y')
# sp500 = sp500[1:]
#
# nifty = pd.read_csv(path0+'^NSEI.csv', header = 0, index_col = 'Date')
# nifty.index = pd.to_datetime(nifty.index, format = '%d-%m-%y')
# nifty = nifty.reindex(index = sp500.index, method = 'bfill')
# nifty.fillna(method = 'bfill', inplace=True)
#
# sing_sti = pd.read_csv(path0+'^sti_d.csv', header = 0, index_col = 'Date')
# sing_sti.index = pd.to_datetime(sing_sti.index, format = '%Y-%m-%d')
# sing_sti = sing_sti.reindex(index = sp500.index, method = 'bfill')
# sing_sti.fillna(method = 'bfill', inplace=True)
#
# uk_100 = pd.read_csv(path0+'^ukx_d.csv', header = 0, index_col = 'Date')
# uk_100.index = pd.to_datetime(uk_100.index, format = '%Y-%m-%d')
# uk_100 = uk_100.reindex(index = sp500.index, method = 'bfill')
# uk_100.fillna(method = 'bfill', inplace=True)
#
# hangseng = pd.read_csv(path0+'^hsi_d.csv', header = 0, index_col = 'Date')
# hangseng.index = pd.to_datetime(hangseng.index, format = '%Y-%m-%d')
# hangseng = hangseng.reindex(index = sp500.index, method = 'bfill')
# hangseng.fillna(method = 'bfill', inplace=True)
#
# nikkei = pd.read_csv(path0+'^nkx_d.csv', header = 0, index_col = 'Date')
# nikkei.index = pd.to_datetime(nikkei.index, format = '%Y-%m-%d')
# nikkei = nikkei.reindex(index = sp500.index, method = 'bfill')
# nikkei.fillna(method = 'bfill', inplace=True)
#
# shanghai_comp = pd.read_csv(path0+'^shc_d.csv', header = 0, index_col = 'Date')
# shanghai_comp.index = pd.to_datetime(shanghai_comp.index, format = '%Y-%m-%d')
# shanghai_comp = shanghai_comp.reindex(index = sp500.index, method = 'bfill')
# shanghai_comp.fillna(method = 'bfill', inplace=True)
#
# df = pd.DataFrame(index = sp500.index)
# df['nifty'] = nifty['Close']
# df['sing_sti'] = sing_sti['Close']
# df['hangseng'] = hangseng['Close']
# df['nikkei'] = nikkei['Close']
# df['shanghai_comp'] = shanghai_comp['Close']
# df['sp500'] = sp500['Close']
# df['uk_100'] = uk_100['Close']
#
# data_cache = df.copy()
# data_cache.dropna(inplace = True)

"""
Run this block to include VAR residuals in input data
"""
# data_cache =db_pre_process()
# for label in data_cache.columns[1:]:
#     resids = var_resids('nifty', label, data_cache = data_cache)
#     data_cache[str.format("nifty_var_%s"%(label))] = resids
# data_cache.dropna(inplace = True)
#
# """
# Set model targets
# """
# data_cache = data_cache[-3090:-90]
# data_cache['nifty_volatility'] = np.log(data_cache['nifty']/data_cache['nifty'].shift(1))**2
# data_cache.dropna(inplace = True)
# data_cache['targets'] = data_cache['nifty_volatility'].shift(-1)
# data_cache.dropna(inplace = True)
data_cache=db_pre_process()
"""
Split datasets
Use "dense" for Dense NN predictions
Returns sclaer used for scaling the output variables
"""


X_train, X_test, y_train, y_test, output_scaler = make_datasets(data_cache, model_name = 'dense')


save_test_variables(output_scaler,"dense_model.p","dense_test.npz",X_test,y_test)
#
# with   open(pickle_folder+"dense_model.p", 'wb') as f:
#     pickle.dump(output_scaler, f, pickle.HIGHEST_PROTOCOL)
#
# np.savez(test_folder+"desns_test.npz",name1=X_test,name2=y_test)
"""
Model defenition
Model Compilation: Batching enabled
"""
dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=25, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=25, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
result = nn_model_compile(dense, X_train, y_train)
dense.evaluate(X_test, y_test)
result.model.save(model_folder+"dens_model.h5")
print ("Dense training wen well")
# dense.save(model_folder+"dens11_model.h5")
# dense =[]
# dense =tf.keras.models.load_model(model_folder+"dens11_model.h5")
# y_pred = dense.predict(X_test)
# y_test = y_test.reshape(len(y_test), 1)
# y_pred = y_pred.reshape(len(y_pred), 1)
# y_pred = output_scaler.inverse_transform(y_pred)
# y_test = output_scaler.inverse_transform(y_test)
#
# """
# Run model
# Inverse transform test targets and predictions
# """
# #
# # y_pred = dense.predict(X_test)
# # y_test = y_test.reshape(len(y_test), 1)
# # y_pred = y_pred.reshape(len(y_pred), 1)
# # y_pred = output_scaler.inverse_transform(y_pred)
# # y_test = output_scaler.inverse_transform(y_test)
#
# """
# RMSE of inverse transformed variables
# """
# m = tf.metrics.RootMeanSquaredError()
# m.update_state(y_test, np.abs(y_pred))
# transformed = m.result().numpy()
# print("RMSE of transformed variables: %d", transformed)
#
df_plot = make_save_plot(data_cache.index, y_test, y_pred)
