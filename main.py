from data import Data
from svr import SVRPredictor
import numpy as np
import joblib
#from sklearn import dump
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

# erroe analysis
sys.stdout = open("../ea_log/log.txt","wt")

# draw paper figures
fig, axes = plt.subplots(2, len(chip_ids), figsize = (80, 26))

for ax in axes.flatten():
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontsize(ticklabelsz) for label in labels]

# SVR
RE_mean_list = []
RE_last_list = []

for k, predict_range in enumerate(predict_ranges):
    RE_mean_chip = np.zeros(len(chip_ids))
    RE_last = np.zeros(len(chip_ids))
    for i, chip_id in enumerate(chip_ids):
        AgingData = Data(chip_id, train_sheet_num)
        train_data_x, train_data_y, threshold_AF = AgingData.get_test_data_gause()
        test_data_x, test_data_y, threshold_AF = AgingData.get_test_data_gause()
        predict_index = int((len(train_data_x) * predict_range)) 
        train_data_x_predict = train_data_x[0:predict_index]
        train_data_y_predict = train_data_y[0:predict_index]
        #test_data_x, test_data_y = AgingData.get_test_data_from_RO()
        train_data_x_array = train_data_x.reshape(-1,1)
        train_data_y_array = train_data_y.reshape(-1,1)
        test_data_x_array = test_data_x.reshape(-1,1)
        test_data_y_array = test_data_y.reshape(-1,1)
        train_data_x_predict_array = train_data_x_predict.reshape(-1,1)
        train_data_y_predict_array = train_data_y_predict.reshape(-1,1)
        colors = cm.rainbow(np.linspace(0, 1, 10))
        #fit code
        #if pre_train == 1:
        #    Predictor = joblib.load("../model/predictor"+str(0)+str(i)+".pkl")
        #else: 
        #    Predictor = SVRPredictor(chip_id)
        #    Predictor.train(train_data_x_array, train_data_y_array)
        #    #joblib.dump(clf,"../model/predictor"+str(0)+str(i)+".pkl")
        #predict_data_y = Predictor.predict(test_data_x_array)
        #fit code

        #predict code

        Predictor = SVRPredictor(chip_id)
        Predictor.train(train_data_x_predict_array, train_data_y_predict_array)
        predict_data_y = Predictor.predict(test_data_x_array)
        predict_data_y_gause = gaussian_filter1d(predict_data_y,5)
        #print(len(RE_mean_chip))
        analyse_error_data_eva = predict_data_y_gause[predict_index:]
        analyse_error_data_pra = test_data_y[predict_index:]
        RE = np.zeros(len(analyse_error_data_eva))
        for j in range(len(analyse_error_data_eva)):
            RE[j] = abs((analyse_error_data_eva[j] - analyse_error_data_pra[j])/test_data_y[0])
        RE_last[i] = RE[-1]
        RE_mean_chip[i] = np.mean(RE)
        if (i == 3):
            RE_mean_list.append(RE_mean_chip)
            RE_last_list.append(RE_last)
        #print(f"RE_mean = {RE_mean_chip}")
        print(f"predict index = {predict_range} chip{i}'s RE = {RE}")
        #print(f"RE mean list = {RE_mean_list}")
        print(f"RE last = {RE_last}")
        print(f"RE last list = {RE_last_list}")
        
        #print(np.shape(RE))
        #err = Predictor.error_analysis(test_data_y_array, predict_data_y)
        #print('last 10% NRMSE of SVR evaluation on chip '+str(i)+':\n')
        #print(err[int(0.9*len(err)):], '\n')
        #print('AVG on last 10%: ', np.mean(err[int(0.9*len(err)):]))
        #print('AVG on last 5%: ', np.mean(err[int(0.95*len(err)):]), '\n\n')
        #err = []
        #for j in range(test_data_y_array.shape[0]):
        #    
    #        
        #    
        #    dp = predict_data_y[j]
        #    avg = test_data_y[j]
        #    err.append((dp-avg)*(dp-avg)/avg/avg)
        #err = np.array(err)
        axes[0,i].plot(np.array(range(len(RE))), RE, color = 'red')
        axes[0,i].set_title('Chip ' + str(i), fontsize = stitlesz)
        axes[0,i].set_xlabel('Sample Points', fontsize = labelsz, labelpad = 20)
        axes[0,i].tick_params(labelsize=20)
    #plt.suptitle('Relative Error of SVR Evaluation Results', fontsize = mtitlesz, x = 0.5, y = 1.03)
    #plt.savefig('../img/' + str(predict_range) + '/svr_RE' + str(predict_range) + '+' + str(threshold_AF) + '.png', bbox_inches = 'tight')
        #predict code

        #analyse_error_data_eva = predict_data_y_gause[predict_index:]
        #analyse_error_data_pra = test_data_y[predict_index:]
        #RE = np.zeros(len(analyse_error_data_eva)) #这个地方有个奇怪的问题
        #for i in range(len(analyse_error_data_eva)):
        #    RE[i] = abs(analyse_error_data_eva[i] - analyse_error_data_pra[i])/test_data_y[0]
        #print(f"RE = {RE}")
        #Predictor.draw(train_data_x_array, train_data_y_array, test_data_x_array, test_data_y_array, predict_data_y, dopt)
        #axes[i].scatter(np.array(range(test_data_y.shape[0])), test_data_y, s = ssz, color = 'black', label = 'measured value')
        # calculate avg val of test_data_y and draw blue line
        axes[1,i].vlines(predict_index, 0, test_data_y[0], color = "red")
        axes[1,i].plot(np.array(range(len(test_data_y))), test_data_y, color = 'green', label = 'practical value', linewidth = linewidth)
        axes[1,i].plot(np.array(range(len(predict_data_y))), predict_data_y, color = 'red', label = 'evaluation result', linewidth = linewidth)
        axes[1,i].plot(np.array(range(len(predict_data_y_gause))), predict_data_y_gause, color = 'black', label = 'evaluation result after gause filiter', linewidth = linewidth)

        axes[1,i].set_title('Chip ' + str(i), fontsize = stitlesz)
        axes[1,i].set_xlabel('Sample Points', fontsize = labelsz, labelpad = 20)



    axes[0,0].set_ylabel('Relative Error', fontsize = labelsz, labelpad = 30)
    axes[1,0].set_ylabel('RUL', fontsize = labelsz, labelpad = 30)
    handles,labels = ax.get_legend_handles_labels()

    #handles = [handles[2], handles[0], handles[1]]
    #labels = [labels[2], labels[0], labels[1]]

    ax.legend(handles,labels,loc=5, bbox_to_anchor=(1.5, 0, 0.4, 1), fontsize = labelsz)
    plt.suptitle('SVR Evaluation Results', fontsize = mtitlesz, x = 0.5, y = 1.03)
    #plt.savefig('../img/'+ str(predict_range) + '/svr_test'+ str(predict_range) + '+' + str(threshold_AF) +'.png', bbox_inches = 'tight')
    plt.savefig('../img/'+ str(predict_range) + '/svr_test_test_code'+ str(predict_range) + '+' + str(threshold_AF) +'.png', bbox_inches = 'tight')
    for i in range(len(chip_ids)):
        axes[0,i].cla()
        axes[1,i].cla()

    for ax in axes.flatten():
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontsize(ticklabelsz) for label in labels]



#for k, predict_range in enumerate(predict_ranges):
#    
#    RE_mean_chip = np.zeros(len(chip_ids))
#    RE_last = np.zeros(len(chip_ids))
#    for i, chip_id in enumerate(chip_ids):
#        AgingData = Data(chip_id, train_sheet_num)
#        train_data_x, train_data_y, threshold_AF = AgingData.get_test_data_gause()
#        test_data_x, test_data_y, threshold_AF = AgingData.get_test_data_gause()
#        #test_data_x, test_data_y = AgingData.get_test_data_from_RO()
#        predict_index = int((len(train_data_x) * predict_range)) 
#        train_data_x_predict = train_data_x[0:predict_index]
#        train_data_y_predict = train_data_y[0:predict_index]
#        train_data_x_array = train_data_x.reshape(-1,1)
#        train_data_y_array = train_data_y.reshape(-1,1)
#        test_data_x_array = test_data_x.reshape(-1,1)
#        test_data_y_array = test_data_y.reshape(-1,1)
#        train_data_x_predict_array = train_data_x_predict.reshape(-1,1)
#        train_data_y_predict_array = train_data_y_predict.reshape(-1,1)
#        #colors = cm.rainbow(np.linspace(0, 1, test_data_y.shape[1]))
#        colors = cm.rainbow(np.linspace(0, 1, 10))
#        Predictor = SVRPredictor(chip_id)
#        Predictor.train(train_data_x_array, train_data_y_array)
#        predict_data_y = Predictor.predict(test_data_x_array)
#        predict_data_y_gause = gaussian_filter1d(predict_data_y,3)
#        analyse_error_data_eva = predict_data_y_gause[predict_index:]
#        analyse_error_data_pra = test_data_y[predict_index:]
#        RE = np.zeros(len(analyse_error_data_eva)) 
#        #print(len(RE_mean_chip))
#        for j in range(len(analyse_error_data_eva)):
#            RE[j] = abs((analyse_error_data_eva[j] - analyse_error_data_pra[j])/test_data_y[0])
#        RE_last[i] = RE[-1]
#        RE_mean_chip[i] = np.mean(RE)
#        if (i == 3):
#            RE_mean_list.append(RE_mean_chip)
#            RE_last_list.append(RE_last)
#        #print(f"RE_mean = {RE_mean_chip}")
#        print(f"predict index = {predict_range} chip{i}'s RE = {RE}")
#        #print(f"RE mean list = {RE_mean_list}")
#        print(f"RE last = {RE_last}")
#        print(f"RE last list = {RE_last_list}")
#        
#        #print(np.shape(RE))
#        #err = Predictor.error_analysis(test_data_y_array, predict_data_y)
#        #print('last 10% NRMSE of SVR evaluation on chip '+str(i)+':\n')
#        #print(err[int(0.9*len(err)):], '\n')
#        #print('AVG on last 10%: ', np.mean(err[int(0.9*len(err)):]))
#        #print('AVG on last 5%: ', np.mean(err[int(0.95*len(err)):]), '\n\n')
#        #err = []
#        #for j in range(test_data_y_array.shape[0]):
#        #    
#    #        
#        #    
#        #    dp = predict_data_y[j]
#        #    avg = test_data_y[j]
#        #    err.append((dp-avg)*(dp-avg)/avg/avg)
#        #err = np.array(err)
#        axes[i].plot(np.array(range(len(RE))), RE, color = 'red')
#        axes[i].set_title('Chip ' + str(i), fontsize = stitlesz)
#        axes[i].set_xlabel('Sample Points', fontsize = labelsz, labelpad = 20)
#        axes[i].tick_params(labelsize=20)
#    axes[0].set_ylabel('Relative Error', fontsize = labelsz, labelpad = 30)
#    plt.suptitle('Relative Error of SVR Evaluation Results', fontsize = mtitlesz, x = 0.5, y = 1.03)
#    plt.savefig('../img/' + str(predict_range) + '/svr_RE' + str(predict_range) + '+' + str(threshold_AF) + '.png', bbox_inches = 'tight')
#    for i in range(len(chip_ids)):
#        axes[i].cla()
RE_mean_array = np.array(RE_mean_list)
RE_last_array = np.array(RE_last_list)
#print(f"RE for average = {RE_mean_array}")
print(f"RE for last = {RE_last_array}")
chip_ids_draw = ['0', '1', '2', '3']
fig, axes = plt.subplots(1, len(predict_ranges), figsize = (80, 13))
for ax in axes.flatten():
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontsize(ticklabelsz) for label in labels]
for k, predict_range in enumerate(predict_ranges):
    axes[k].bar(chip_ids_draw, RE_last_array[k], color = 'red')
    axes[k].set_title('predict index = ' + str(predict_range), fontsize = stitlesz)
    axes[k].set_xlabel('Chip', fontsize = labelsz, labelpad = 20)
axes[0].set_ylabel('Relative Error ', fontsize = labelsz, labelpad = 30)
plt.suptitle('Relative Error on Different Chip', fontsize = mtitlesz, x = 0.5, y = 1.03)
plt.savefig('../img/paper/svr_RE_different.png', bbox_inches = 'tight')

predict_ranges_draw = ['0.85', '0.9', '0.95']
fig, axes = plt.subplots(1, len(chip_ids), figsize = (80, 13))
for ax in axes.flatten():
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontsize(ticklabelsz) for label in labels]
for i, chip_id in enumerate(chip_ids):
    axes[i].bar(predict_ranges_draw, RE_last_array[:,i], color = 'red')
    axes[i].set_title('Chip = ' + str(i), fontsize = stitlesz)
    axes[i].set_xlabel('Predict index', fontsize = labelsz, labelpad = 20)
axes[0].set_ylabel('Relative Error ', fontsize = labelsz, labelpad = 30)
plt.suptitle('Relative Error on Same Chip', fontsize = mtitlesz, x = 0.5, y = 1.03)
plt.savefig('../img/paper/svr_RE_same.png', bbox_inches = 'tight')
