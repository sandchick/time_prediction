import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gptkalman import Kalman
from statsmodels.tsa.arima.model import ARIMA
pcolors = ['b', 'g', 'm', 'r', 'y', 'p']
mtitlesz = 60
stitlesz = 56
labelsz = 52
ticklabelsz = 36
linewidth = 4
ssz = 16
fig, axes = plt.subplots(1,2,figsize = (50,26))
for ax in axes.flatten():
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontsize(ticklabelsz) for label in labels]
# 读取数据集
Aging_data = Kalman(7,15)
data = Aging_data.fliter()
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
            axes[k].plot(train_x, train_data, label='befor pre actual',linewidth = linewidth)
            axes[k].plot(test_x, predictions, label='predictions',linewidth = linewidth)
            axes[k].plot(test_x, test_data, label='actual',linewidth = linewidth)
            axes[k].set_xlabel('Time', fontsize = labelsz, labelpad = 30)
        axes[0].set_ylabel('Aging Factors', fontsize = labelsz, labelpad = 30)
        handles,labels = ax.get_legend_handles_labels()
        ax.legend(handles,labels,loc=5,bbox_to_anchor=(1.5, 0, 0.4, 1),fontsize = labelsz)
        plt.suptitle('ARMA Predict Results', fontsize = mtitlesz, x = 0.5, y = 1.03)
        plt.savefig('../img/ARMA/'+ str(p) + str(d) + str(q) + 'ARMA_pre.png')
        for k in range(2):
            axes[k].cla()
        for ax in axes.flatten():
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontsize(ticklabelsz) for label in labels]