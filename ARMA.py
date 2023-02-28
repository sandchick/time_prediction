import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gptkalman import Kalman
from statsmodels.tsa.arima.model import ARIMA
import sys

pcolors = ['b', 'g', 'm', 'r', 'y', 'p']
mtitlesz = 60
stitlesz = 56
labelsz = 52
ticklabelsz = 36
linewidth = 4
ssz = 16
fig, axes = plt.subplots(1,1,figsize = (50,26))
labels = axes.get_xticklabels() + axes.get_yticklabels()
[label.set_fontsize(ticklabelsz) for label in labels]

chip_id = 4
#error analysis
sys.stdout = open("../ea_log/logarma.txt","wt")
# 读取数据集
data_range = 0.9
Aging_data = Kalman(chip_id,15)
raw_data = Aging_data.fliter()
data = raw_data[:int(data_range*len(raw_data))]
#data_diff = np.gradient(data)
# 将数据集分为训练集和测试集


train_range = 0.95
x = np.array(range(len(data)))
train_samples = int(len(data) * train_range)
#train_data_diff = data_diff[:int(0.9*(len(data)))]
train_data = data[:int(train_range*(len(data)))]
test_data = data[int(train_range*(len(data))):]
train_x = x[:train_samples]
test_x= x[train_samples:]

# 拟合ARIMA模型
ps = [1,2]
ds = [1,2]
qs = [1,2]
for i, p in enumerate(ps):
    for j, d in enumerate(ds):
        for k, q in enumerate(qs):
            model = ARIMA(train_data, order=(p,d,q))
            results = model.fit()

        # 对测试集进行预测
            predictions = results.predict(start=len(train_data), end=len(data)-1, dynamic=False)
            #pred_cumsum = prediction.cumsum()
            #predictions = pred_cumsum + data[0]

# 绘制预测结果和实际值
            axes.plot(train_x, train_data, color = "blue", label='befor pre actual',linewidth = linewidth)
            axes.plot(test_x, predictions, color = "red", label='predictions',linewidth = linewidth)
            axes.plot(test_x, test_data, color = "green", label='actual',linewidth = linewidth)
            axes.set_xlabel('Time', fontsize = labelsz, labelpad = 30)
            axes.set_ylabel('Aging Factors', fontsize = labelsz, labelpad = 30)
            plt.suptitle('ARMA Predict Results', fontsize = mtitlesz, x = 0.5, y = 1.03)
            handles, labels = axes.get_legend_handles_labels()
            axes.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=labelsz)
            plt.tight_layout()
            plt.savefig('../img/ARMA/kalman' + str(chip_id) + '/' + str(p) + str(d) + str(q) + 'ARMA_pre.png')
            axes.cla()

for i, p in enumerate(ps):
    for j, d in enumerate(ds):
        for k, q in enumerate(qs):
            model = ARIMA(train_data, order=(p,d,q))
            results = model.fit()

        # 对测试集进行预测
            predictions = results.predict(start=len(train_data), end=len(data)-1, dynamic=False)
            #pred_cumsum = prediction.cumsum()
            #predictions = pred_cumsum + data[0]
        
        #error analsis
            err = []
            for l in range(int(len(predictions))):
                dp = predictions[l]
                avg = test_data[l]
                err.append(abs((dp-avg)/avg))
            err = np.array(err)
            print (f"Related error =chip{chip_id} in p = {p},d = {d}, q = {q}, error{err}")
        #draw error
            axes.plot(test_x, err, color = "red", label='predictions error',linewidth = linewidth)
            axes.set_xlabel('Time', fontsize = labelsz, labelpad = 30)
            axes.set_ylabel('Related Error', fontsize = labelsz, labelpad = 30)
            handles,labels = axes.get_legend_handles_labels()
            axes.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=labelsz)
            plt.tight_layout()
            #ax.legend(handles,labels,loc='upper right',bbox_to_anchor=(1.5, 0, 0.4, 1),fontsize = labelsz)
            #ax.legend(loc='upper right',bbox_to_anchor=(1.5, 0, 0.4, 1),fontsize = labelsz)
            plt.suptitle('ARMA Predict Related Error', fontsize = mtitlesz, x = 0.5, y = 1.03)
            plt.savefig('../img/ARMA/kalman' + str(chip_id) + '/' + str(p) + str(d) + str(q) + 'ARMA_error.png')
            axes.cla()