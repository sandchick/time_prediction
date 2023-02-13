from data import Data
from svr import SVRPredictor
import numpy as np
import joblib
#from sklearn import dump
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras.models import load_model
import sys


pre_train = 0 # if pre_train = 1, use the saved model
dopt = 1  # draw option, draw all points if dopt = 0, draw last 20% points if dopt = 1
chip_ids = [2, 4, 5, 6, 7]
train_sheet_num = 15
#Arch = [128, 32, 8, 2, 1]
#pcolors = ['b', 'g', 'm', 'r', 'y', 'p']
pcolors = ['b', 'g', 'm', 'r', 'y']

mtitlesz = 60
stitlesz = 56
labelsz = 52
ticklabelsz = 36
linewidth = 4
ssz = 16

# erroe analysis
sys.stdout = open("../ea_log/log.txt","wt")

# draw paper figures
fig, axes = plt.subplots(1, 5, figsize = (80, 13))

for ax in axes.flatten():
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontsize(ticklabelsz) for label in labels]

# SVR

for i, chip_id in enumerate(chip_ids):
    AgingData = Data(chip_id, train_sheet_num)
    train_data_x, train_data_y = AgingData.get_train_data_gause()
    test_data_x, test_data_y = AgingData.get_test_data_gause()
    #predict_index = int((len(train_data_x) * 0.7)) 
    #train_data_x_predict = train_data_x[0:predict_index]
    #test_data_x, test_data_y = AgingData.get_test_data_from_RO()
    train_data_x_array = train_data_x.reshape(-1,1)
    train_data_y_array = train_data_y.reshape(-1,1)
    test_data_x_array = test_data_x.reshape(-1,1)
    test_data_y_array = test_data_y.reshape(-1,1)
    colors = cm.rainbow(np.linspace(0, 1, 10))
    if pre_train == 1:
        Predictor = joblib.load("../model/predictor"+str(0)+str(i)+".pkl")
    else: 
        Predictor = SVRPredictor(chip_id)
        Predictor.train(train_data_x_array, train_data_y_array)
        #joblib.dump(clf,"../model/predictor"+str(0)+str(i)+".pkl")
    predict_data_y = Predictor.predict(test_data_x_array)
    #Predictor.draw(train_data_x_array, train_data_y_array, test_data_x_array, test_data_y_array, predict_data_y, dopt)
    #axes[i].scatter(np.array(range(test_data_y.shape[0])), test_data_y, s = ssz, color = 'black', label = 'measured value')
    # calculate avg val of test_data_y and draw blue line
    axes[i].plot(np.array(range(test_data_y_array.shape[0])), test_data_y, color = 'green', label = 'practical value', linewidth = linewidth)
    axes[i].plot(np.array(range(len(predict_data_y))), predict_data_y, color = 'red', label = 'evaluation result', linewidth = linewidth)

    axes[i].set_title('Chip ' + str(i), fontsize = stitlesz)
    axes[i].set_xlabel('Sample Points', fontsize = labelsz, labelpad = 20)
axes[0].set_ylabel('RUL', fontsize = labelsz, labelpad = 30)
handles,labels = ax.get_legend_handles_labels()

#handles = [handles[2], handles[0], handles[1]]
#labels = [labels[2], labels[0], labels[1]]

ax.legend(handles,labels,loc=5, bbox_to_anchor=(1.5, 0, 0.4, 1), fontsize = labelsz)
plt.suptitle('SVR Evaluation Results', fontsize = mtitlesz, x = 0.5, y = 1.03)
plt.savefig('../img/svr_test.png', bbox_inches = 'tight')
for i in range(5):
    axes[i].cla()

for ax in axes.flatten():
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontsize(ticklabelsz) for label in labels]

for i, chip_id in enumerate(chip_ids):
    AgingData = Data(chip_id, train_sheet_num)
    train_data_x, train_data_y = AgingData.get_train_data_gause()
    test_data_x, test_data_y = AgingData.get_test_data_gause()
    #test_data_x, test_data_y = AgingData.get_test_data_from_RO()
    train_data_x_array = train_data_x.reshape(-1,1)
    train_data_y_array = train_data_y.reshape(-1,1)
    test_data_x_array = test_data_x.reshape(-1,1)
    test_data_y_array = test_data_y.reshape(-1,1)
    #colors = cm.rainbow(np.linspace(0, 1, test_data_y.shape[1]))
    colors = cm.rainbow(np.linspace(0, 1, 10))
    if pre_train == 1:
        Predictor = joblib.load("../model/predictor"+str(0)+str(i)+".pkl")
    else: 
        Predictor = SVRPredictor(chip_id)
        Predictor.train(train_data_x_array, train_data_y_array)
    predict_data_y = Predictor.predict(test_data_x_array)
    err = Predictor.error_analysis(test_data_y_array, predict_data_y)
    print('last 10% NRMSE of SVR evaluation on chip '+str(i)+':\n')
    print(err[int(0.9*len(err)):], '\n')
    print('AVG on last 10%: ', np.mean(err[int(0.9*len(err)):]))
    print('AVG on last 5%: ', np.mean(err[int(0.95*len(err)):]), '\n\n')
    err = []
    for j in range(test_data_y_array.shape[0]):
        
        dp = predict_data_y[j]
        avg = test_data_y[j]
        err.append((dp-avg)*(dp-avg)/avg/avg)
    err = np.array(err)
    axes[i].plot(np.array(range(10, len(err)))+10, err[10:], color = 'red')
    axes[i].set_title('Chip ' + str(i), fontsize = stitlesz)
    axes[i].set_xlabel('Sample Points', fontsize = labelsz, labelpad = 20)
axes[0].set_ylabel('Relative Error', fontsize = labelsz, labelpad = 30)
plt.suptitle('Relative Error of SVR Evaluation Results', fontsize = mtitlesz, x = 0.5, y = 1.03)
plt.savefig('../img/svr_RE.png', bbox_inches = 'tight')
for i in range(5):
    axes[i].cla()