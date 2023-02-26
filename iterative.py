from data import Data
from svr import SVRPredictor
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras.models import load_model
import sys
from scipy.ndimage import gaussian_filter1d


pre_train = 0 # if pre_train = 1, use the saved model
dopt = 1  # draw option, draw all points if dopt = 0, draw last 20% points if dopt = 1
chip_ids = [2, 4, 5, 6 ]
#chip_ids = [ 4, 5, 6, 8]
train_sheet_num = 15
#Arch = [128, 32, 8, 2, 1]
pcolors = ['b', 'g', 'm', 'r', 'y', 'p']
#pcolors = ['b', 'g', 'm', 'r', 'y']

mtitlesz = 60
stitlesz = 56
labelsz = 52
ticklabelsz = 36
linewidth = 4
ssz = 16
predict_ranges = [0.85, 0.9, 0.95]
threshold_AF = 0.032
# erroe analysis
sys.stdout = open("../ea_log/lognew.txt","wt")

# draw paper figures
fig, axes = plt.subplots(1, len(chip_ids), figsize = (80, 13))

for ax in axes.flatten():
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontsize(ticklabelsz) for label in labels]


for k, predict_range in enumerate(predict_ranges):
    for i, chip_id in enumerate(chip_ids):
        AgingData = Data(chip_id, train_sheet_num)
        train_data_x, train_data_y  = AgingData.get_iterative_data()
        test_data_x, test_data_y = AgingData.get_iterative_data()
        #AFR = np.zeros(len(train_data_y))
        #for i in range(len(train_data_y)):
        #    AFR[i] = abs(train_data_y - threshold_AF)
        #AFR = AFR.tolist()
        #FT = AFR.index(min(AFR))
        #predict_index = int((len(train_data_x) * predict_range)) 
        #train_data_x_predict = train_data_x[0:predict_index,:]
        #train_data_y_predict = train_data_y[0:predict_index]
        #train_data_x_array = train_data_x.reshape(-1,1)
        #train_data_y_array = train_data_y.reshape(-1,1)
        #test_data_x_array = test_data_x.reshape(-1,1)
        #test_data_y_array = test_data_y.reshape(-1,1)
        #train_data_x_predict_array = train_data_x_predict.reshape(-1,1)
        #train_data_y_predict_array = train_data_y_predict.reshape(-1,1)
        colors = cm.rainbow(np.linspace(0, 1, 10))
        Predictor = SVRPredictor(chip_id)
        #print(f"train_data_x={np.shape(train_data_x_predict)},train_data_y={np.shape(train_data_y_predict)}")
        #Predictor.train(train_data_x_predict, train_data_y_predict)
        Predictor.train(train_data_x, train_data_y)
        predict_data_y = Predictor.predict(test_data_x)
        predict_data_y_gause = gaussian_filter1d(predict_data_y,5)
        #analyse_error_data_eva = predict_data_y_gause[predict_index:]
        #analyse_error_data_pra = test_data_y[predict_index:]
        #RE = np.zeros(len(analyse_error_data_eva))
        #for j in range(len(analyse_error_data_eva)):
        #    RE[j] = abs((analyse_error_data_eva[j] - analyse_error_data_pra[j])/test_data_y[0])
        #RE_last[i] = RE[-1]
        #RE_mean_chip[i] = np.mean(RE)
        #if (i == 3):
        #    RE_mean_list.append(RE_mean_chip)
        #    RE_last_list.append(RE_last)
        #print(f"predict index = {predict_range} chip{i}'s RE = {RE}")
        #print(f"RE last = {RE_last}")
        #print(f"RE last list = {RE_last_list}")
        #axes[i].plot(np.array(range(len(RE))), RE, color = 'red')
        axes[i].set_title('Chip ' + str(i), fontsize = stitlesz)
        axes[i].set_xlabel('TIME', fontsize = labelsz, labelpad = 20)
        #axes[i].tick_params(labelsize=20)
        #axes[i].vlines(predict_index, 0, test_data_y[0], color = "red")
        axes[i].plot(np.array(range(len(test_data_y))), test_data_y, color = 'green', label = 'practical value', linewidth = linewidth)
        axes[i].plot(np.array(range(len(predict_data_y))), predict_data_y, color = 'red', label = 'evaluation result', linewidth = linewidth)
        #axes[i].plot(np.array(range(len(predict_data_y_gause))), predict_data_y_gause, color = 'black', label = 'evaluation result after gause filiter', linewidth = linewidth)

        #axes[i].set_title('Chip ' + str(i), fontsize = stitlesz)
        #axes[i].set_xlabel('Sample Points', fontsize = labelsz, labelpad = 20)



    axes[0].set_ylabel('AF', fontsize = labelsz, labelpad = 30)
    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles,labels,loc=5, bbox_to_anchor=(1.5, 0, 0.4, 1), fontsize = labelsz)
    plt.suptitle('SVR Evaluation Results', fontsize = mtitlesz, x = 0.5, y = 1.03)
    #plt.savefig('../img/'+ str(predict_range) + '/svr_test'+ str(predict_range) + '+' + str(threshold_AF) +'.png', bbox_inches = 'tight')
    plt.savefig('../img/iteration/'+ str(predict_range) + '/svr_test_test_code'+ str(predict_range) + '+' + str(threshold_AF) +'.png', bbox_inches = 'tight')
    for i in range(len(chip_ids)):
        axes[i].cla()