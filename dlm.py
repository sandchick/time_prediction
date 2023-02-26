from pydlm import dlm, trend, seasonality
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gptkalman import Kalman
from statsmodels.tsa.arima.model import ARIMA

# 准备数据（假设已经读取到了一个名为“data”的DataFrame中）
Aging_data = Kalman(2,15)
data = Aging_data.fliter()
x = np.array(range(len(data)))
train_samples = int(len(data) * 0.9)
#train_data_diff = data_diff[:int(0.9*(len(data)))]
train_data = data[:int(0.9*(len(data)))]
test_data = data[int(0.9*(len(data))):]
train_x = x[:train_samples]
test_x= x[train_samples:]
# 建立动态线性模型


# 训练模型
predicts_list = np.zeros(len(test_x))
# 进行预测
for i in range(len(test_x)):
    mydlm = dlm(train_data) + trend(1) + seasonality(period=7)
    mydlm.fit()
    predicts, var = mydlm.predict()
    train_data = train_data + predicts
    predicts_list[i] = predicts

## 输出预测结
## 这模型没救了，我傻了
plt.plot(test_x,predicts_list,label ='actual train')
plt.plot(test_x,test_data, label='predictions')
#plt.plot(test_x,test_data, label='actual test')
plt.legend()
plt.show()