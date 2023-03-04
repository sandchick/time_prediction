import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gptkalman import Kalman
from statsmodels.tsa.arima.model import ARIMA
import sys
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

pcolors = ['b', 'g', 'm', 'r', 'y', 'p']
mtitlesz = 60
stitlesz = 56
labelsz = 52
ticklabelsz = 36
linewidth = 4
ssz = 16
chip_ids = [2,4,5,6,7]
ArmaParameterMap90 = [[2,1,2],
                     [2,1,1],
                     [2,1,2],
                     [2,1,1],
                     [2,1,1]]
ArmaParameterMap95 = [[2,2,2],
                     [2,1,1],
                     [2,2,2],
                     [2,1,1],
                     [2,2,2]]
fig, axes = plt.subplots(1,len(chip_ids),figsize = (80,13))
for ax in axes.flatten():
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontsize(ticklabelsz) for label in labels]

#error analysis
#sys.stdout = open("../ea_log/logarma.txt","wt")
for i,chip_id in enumerate(chip_ids):
# 读取数据集
    data_range = 0.85
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
    p,d,q = ArmaParameterMap95[i]
    model = ARIMA(train_data, order=(p,d,q))
    results = model.fit()

            # 对测试集进行预测
    predictions = results.predict(start=len(train_data), end=len(data)-1, dynamic=False)
    #pred_cumsum = prediction.cumsum()
    #predictions = pred_cumsum + data[0]

    # 绘制预测结果和实际值
    ax_main = axes[i]
    ax_main.plot(train_x, train_data, color = "blue", label='befor pre actual',linewidth = linewidth)
    ax_main.plot(test_x, predictions, color = "red", label='predictions',linewidth = linewidth)
    ax_main.plot(test_x, test_data, color = "green", label='actual',linewidth = linewidth)
    ax_main.set_title('Chip ' + str(chip_id), fontsize = stitlesz)
    ax_main.set_xlabel('Time', fontsize = labelsz, labelpad = 30)

    # 添加缩放子图
    ax_zoom = fig.add_axes([i*0.13+0.15,0.2, 0.05, 0.2])
    ax_zoom.plot(train_x, train_data, color = "blue", label='befor pre actual',linewidth = linewidth)
    ax_zoom.plot(test_x, predictions, color = "red", label='predictions',linewidth = linewidth)
    ax_zoom.plot(test_x, test_data, color = "green", label='actual',linewidth = linewidth)
    ax_zoom.axis([test_x[0],test_x[-1],predictions[0]-0.1,predictions[-1]+0.1])
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
axes[0].set_ylabel('Aging Factors', fontsize = labelsz, labelpad = 30)
plt.suptitle('ARMA Predict Results', fontsize = mtitlesz, x = 0.5, y = 1.0)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=5, bbox_to_anchor=(1.5,0,0.4, 1), fontsize=labelsz)
#plt.tight_layout()
plt.savefig('../paper/ARMA_pre.png')
for i in range(len(chip_ids)):
    axes[i].cla()
for ax in axes.flatten():
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontsize(ticklabelsz) for label in labels]    


for i,chip_id in enumerate(chip_ids):                        
    p,d,q = ArmaParameterMap95[i]                            
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
    avg_err = np.mean(err)
    f = open('../paper/ea/' + str(train_range) + 'chip' + str(chip_id) + str(p) + str(d) + str(q) + 'ARMA_error.txt',"wt")
    print (err,file = f)
    print (avg_err, file = f)
    f.close
#draw error
    axes[i].plot(test_x, err, color = "red", linewidth = linewidth)
    axes[i].set_xlabel('Time', fontsize = labelsz, labelpad = 30)
axes[0].set_ylabel('Related Error', fontsize = labelsz, labelpad = 30)
handles,labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=labelsz)
plt.tight_layout()
#ax.legend(handles,labels,loc='upper right',bbox_to_anchor=(1.5, 0, 0.4, 1),fontsize = labelsz)
#ax.legend(loc='upper right',bbox_to_anchor=(1.5, 0, 0.4, 1),fontsize = labelsz)
plt.suptitle('ARMA Predict Related Error', fontsize = mtitlesz, x = 0.5, y = 1.0)
plt.savefig('../paper/ARMA_error.png')
for i in range(len(chip_ids)):
    axes[i].cla()
for ax in axes.flatten():
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontsize(ticklabelsz) for label in labels]    