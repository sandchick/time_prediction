from data import Data
from svr import SVRPredictor
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras.models import load_model
import sys

sys.stdout = open("../ea_log/nan_error.txt","wt")
#chip_ids = [0,2,4,5,6,7]
chip_ids = [2]
for i, chip_id in enumerate(chip_ids):
    AgingData = Data(chip_id, 15)
    #train_data_x, train_data_y = AgingData.get_train_data()
    #test_data_x, test_data_y = AgingData.get_test_data()
    #test_data_x, test_data_y = AgingData.get_test_data_gause()
    #train_data_x, train_data_y = AgingData.get_train_data_gause()
    #predict_index = int((len(train_data_x) * 0.7)) 
    #train_data_x_predict = train_data_x[0:predict_index]
    #print(f"train data = {len(train_data_x)}")
    #print(f"predict data = {len(train_data_x_predict)}")
    #print(train_data_x[1566])
    #print(train_data_x_predict[1566])
    ##test_data_x, test_data_y = AgingData.get_test_data()
    ##test_data_x, test_data_y = AgingData.get_test_data()
    #train_data_x_array = train_data_x.reshape(-1,1)
    #train_data_y_array = train_data_y.reshape(-1,1)
    #test_data_x_array = test_data_x.reshape(-1,1)
    #test_data_y_array = test_data_y.reshape(-1,1)
    #print(f"train_data_x={np.shape(train_data_x_array)},train_data_y={np.shape(train_data_y_array)}")
    #print(f"test_data_x={np.shape(test_data_x_array)},test_data_y={np.shape(test_data_y_array)}")
    #print(f"train_data={test_data_y_array.shape[0]}") #gause_train_data, mean_train_data= AgingData.get_train_data_gause()
    #gause_test_data, mean_test_data= AgingData.get_test_data_gause()
    #print(f"test_data_x={np.shape(test_data_x_array)},test_data_y={np.shape(test_data_y_array)}")
    #print(mean_test_data)
    #print(mean_train_data)
    #print(RULI)
    #print(f"gause value shape={np.shape(gause_train_data)},mean value shape={np.shape(mean_train_data)}")
    #gause_test_data, mean_test_data= AgingData.get_test_data_gause()
    #print(f"gause test value shape={np.shape(gause_train_data)},mean value shape={np.shape(mean_train_data)}")
    test_data_x, test_data_y = AgingData.get_iterative_data()
    test_data_x_predict = test_data_x[0:1000,:]
    test_data_y_predict = test_data_y[0:1000]
    #print(f"train_data_x={np.shape(test_data_x)},train_data_y={np.shape(test_data_y)}")
    #print(f"train_data_x={np.shape(test_data_x_predict)},train_data_y={np.shape(test_data_y_predict)}")
    print(f"train_data_x={test_data_x},train_data_y={test_data_y}")