from data import Data
from svr import SVRPredictor
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras.models import load_model
import sys

sys.stdout = open("../ea_log/nan_error.txt","wt")
chip_ids = [2,4,5,6,7]
for i, chip_id in enumerate(chip_ids):
    AgingData = Data(chip_id, 15)
    #train_data_x, train_data_y = AgingData.get_train_data()
    #test_data_x, test_data_y = AgingData.get_test_data()
    #train_data_x, train_data_y = AgingData.get_train_data()
    #test_data_x, test_data_y = AgingData.get_test_data()
    #train_data_x_array = train_data_x.reshape(-1,1)
    #train_data_y_array = train_data_y.reshape(-1,1)
    #test_data_x_array = test_data_x.reshape(-1,1)
    #test_data_y_array = test_data_y.reshape(-1,1)
    #print(f"train_data_x={np.shape(train_data_x_array)},train_data_y={np.shape(train_data_y_array)}")
    #print(f"test_data_x={np.shape(test_data_x_array)},test_data_y={np.shape(test_data_y_array)}")
    #print(f"train_data={test_data_y_array.shape[0]}")
    gause_train_data, mean_train_data= AgingData.get_train_data_gause()
    gause_test_data, mean_test_data= AgingData.get_test_data_gause()
    #print(f"gause value shape={np.shape(gause_train_data)},mean value shape={np.shape(mean_train_data)}")
    #gause_test_data, mean_test_data= AgingData.get_test_data_gause()
    #print(f"gause test value shape={np.shape(gause_train_data)},mean value shape={np.shape(mean_train_data)}")