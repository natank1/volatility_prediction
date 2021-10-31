import pandas as pd
from ml_pap import var_resids
import numpy as np
from params import test_folder,pickle_folder
import pickle
def get_db():
    path0="/Users/natankatz/PycharmProjects/pythonProject/Haggai_code/volatility_prediction_study-master/data/"
    sp500 = pd.read_csv(path0+'^GSPC.csv', header = 0, index_col = 'Date')
    sp500.index = pd.to_datetime(sp500.index, format = '%d-%m-%y')
    sp500 = sp500[1:]

    nifty = pd.read_csv(path0+'^NSEI.csv', header = 0, index_col = 'Date')
    nifty.index = pd.to_datetime(nifty.index, format = '%d-%m-%y')
    nifty = nifty.reindex(index = sp500.index, method = 'bfill')
    nifty.fillna(method = 'bfill', inplace=True)

    sing_sti = pd.read_csv(path0+'^sti_d.csv', header = 0, index_col = 'Date')
    sing_sti.index = pd.to_datetime(sing_sti.index, format = '%Y-%m-%d')
    sing_sti = sing_sti.reindex(index = sp500.index, method = 'bfill')
    sing_sti.fillna(method = 'bfill', inplace=True)

    uk_100 = pd.read_csv(path0+'^ukx_d.csv', header = 0, index_col = 'Date')
    uk_100.index = pd.to_datetime(uk_100.index, format = '%Y-%m-%d')
    uk_100 = uk_100.reindex(index = sp500.index, method = 'bfill')
    uk_100.fillna(method = 'bfill', inplace=True)

    hangseng = pd.read_csv(path0+'^hsi_d.csv', header = 0, index_col = 'Date')
    hangseng.index = pd.to_datetime(hangseng.index, format = '%Y-%m-%d')
    hangseng = hangseng.reindex(index = sp500.index, method = 'bfill')
    hangseng.fillna(method = 'bfill', inplace=True)

    nikkei = pd.read_csv(path0+'^nkx_d.csv', header = 0, index_col = 'Date')
    nikkei.index = pd.to_datetime(nikkei.index, format = '%Y-%m-%d')
    nikkei = nikkei.reindex(index = sp500.index, method = 'bfill')
    nikkei.fillna(method = 'bfill', inplace=True)

    shanghai_comp = pd.read_csv(path0+'^shc_d.csv', header = 0, index_col = 'Date')
    shanghai_comp.index = pd.to_datetime(shanghai_comp.index, format = '%Y-%m-%d')
    shanghai_comp = shanghai_comp.reindex(index = sp500.index, method = 'bfill')
    shanghai_comp.fillna(method = 'bfill', inplace=True)

    df = pd.DataFrame(index = sp500.index)
    df['nifty'] = nifty['Close']
    df['sing_sti'] = sing_sti['Close']
    df['hangseng'] = hangseng['Close']
    df['nikkei'] = nikkei['Close']
    df['shanghai_comp'] = shanghai_comp['Close']
    df['sp500'] = sp500['Close']
    df['uk_100'] = uk_100['Close']

    data_cache = df.copy()
    data_cache.dropna(inplace = True)
    return  data_cache



def db_pre_process():
    data_cache = get_db()

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
    return data_cache

def save_test_variables(output_scaler,pickle_name,np_name,X_test,y_test):
    with   open(pickle_folder+"dense_model.p", 'wb') as f:
        pickle.dump(output_scaler, f, pickle.HIGHEST_PROTOCOL)

    np.savez(test_folder+np_name,name1=X_test,name2=y_test)
    return



def get_test_data(np_file_name, pickle_name):
    with   open(pickle_folder + pickle_name, 'rb') as f:
        output_scaler =pickle.load(f)

    data = np.load(test_folder+np_file_name)
    X_test= data['name1']
    y_test= data['name2']
    return X_test, y_test,output_scaler
