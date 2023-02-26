#import numpy as np
#import matplotlib.pyplot as plt
#
## Q为这一轮的心里的预估误差
#Q = 0.00001
## R为下一轮的测量误差
#R = 0.1
## Accumulated_Error为上一轮的估计误差，具体呈现为所有误差的累计
#Accumulated_Error = 1
## 初始旧值
#kalman_adc_old = 0
#
#SCOPE = 50
#
#def kalman(ADC_Value):
#    global kalman_adc_old
#    global Accumulated_Error
#
#    # 新的值相比旧的值差太大时进行跟踪
#    if (abs(ADC_Value-kalman_adc_old)/SCOPE > 0.25):
#        Old_Input = ADC_Value*0.382 + kalman_adc_old*0.618
#    else:
#        Old_Input = kalman_adc_old
#
#    # 上一轮的 总误差=累计误差^2+预估误差^2
#    Old_Error_All = (Accumulated_Error**2 + Q**2)**(1/2)
#
#    # R为这一轮的预估误差
#    # H为利用均方差计算出来的双方的相信度
#    H = Old_Error_All**2/(Old_Error_All**2 + R**2)
#
#    # 旧值 + 1.00001/(1.00001+0.1) * (新值-旧值)
#    kalman_adc = Old_Input + H * (ADC_Value - Old_Input)
#
#    # 计算新的累计误差
#    Accumulated_Error = ((1 - H)*Old_Error_All**2)**(1/2)
#    # 新值变为旧值
#    kalman_adc_old = kalman_adc
#    return kalman_adc
#
# 
#array = np.array([50]*200)
#
#s = np.random.normal(0, 5, 200)
#
#test_array = array + s
#plt.plot(test_array)
#adc=[]
#for i in range(200):
#    adc.append(kalman(test_array[i]))
#    
#plt.plot(adc)   
#plt.plot(array)   
#plt.show()  
from data import Data
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

AgingData = Data(2, 15)
test_x, y = AgingData.get_iterative_data()
# data for train
x = np.array(range(len(y)))
train_samples = int(len(y) * 0.9)
train_x, train_y = x[:train_samples], y[:train_samples]
train_y = y[:train_samples]

# initial and train kalman
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
filtered_state_means, filtered_state_covariances = kf.filter(train_y)

predicted_state_means = filtered_state_means[-1]  # 使用训练集最后一个样本的状态均值作为初始状态均值
filtered_state_covariance = filtered_state_covariances[-1]  # 使用训练集最后一个样本的状态协方差作为初始状态协方差
test_x, test_y = x[train_samples:], y[train_samples:]
predicted_state_means_draw = np.zeros(len(test_x))
predicted_state_covariances = np.zeros(len(test_x))


for i in range(len(y[train_samples:])):
    if i == 0 :
        predicted_state_means,predicted_state_covariances[i]= kf.filter_update(
            predicted_state_means,
            filtered_state_covariance=filtered_state_covariance, 
            observation=train_y[-1], 
            observation_matrix=kf.observation_matrices)
    else:
        predicted_state_means,predicted_state_covariances[i]= kf.filter_update(
            predicted_state_means,
            filtered_state_covariance=filtered_state_covariance, 
            observation=[-1], 
            observation_matrix=kf.observation_matrices)
    predicted_state_means_draw[i] = predicted_state_means
    #filtered_state_means = np.concatenate([filtered_state_means, predicted_state_means])


plt.plot(x, y, label='Original')
plt.plot(train_x, filtered_state_means, label='Kalman Filtered')
plt.plot(test_x, test_y, label='Test Data')
plt.plot(test_x, predicted_state_means_draw, label='Prediction')
plt.legend()
plt.show()

