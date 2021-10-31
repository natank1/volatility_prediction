import numpy as np
import tensorflow as tf
from params import model_folder,test_folder
from db_managmeant import get_test_data,get_db
from ml_pap import make_save_plot

if __name__ =='__main__':
    model_name = "dens11_model.h5"
    numy_test_file ="dense_test.npz"
    pickle_name ="dense_model.p"
    dense = tf.keras.models.load_model(model_folder + model_name)

    X_test, y_test, output_scalar= get_test_data(numy_test_file, pickle_name)
    y_pred = dense.predict(X_test)


    y_test = y_test.reshape(len(y_test), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)

    y_pred = output_scalar.inverse_transform(y_pred)
    y_test = output_scalar.inverse_transform(y_test)

    """
    Run model
    Inverse transform test targets and predictions
    """
    #
    # y_pred = dense.predict(X_test)
    # y_test = y_test.reshape(len(y_test), 1)
    # y_pred = y_pred.reshape(len(y_pred), 1)
    # y_pred = output_scaler.inverse_transform(y_pred)
    # y_test = output_scaler.inverse_transform(y_test)

    """
    RMSE of inverse transformed variables
    """
    m = tf.metrics.RootMeanSquaredError()
    m.update_state(y_test, np.abs(y_pred))
    transformed = m.result().numpy()
    print("RMSE of transformed variables: %d", transformed)
    data_cache =get_db()
    df_plot = make_save_plot(data_cache.index, y_test, y_pred)
