import numpy as np 
import pandas as pd
from data import Data
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
class Kalman:

    def __init__(self, chip_id, train_sheet_num):
        '''[summary] kalman filter
        Parameters
        ----------
        chip_id : [type] int
            [description] chip id
        train_sheet_num : [type] int
            [description] sheet num of train data file
        '''
    # 生成一个随机的时间序列
        self.chip_id = chip_id
        AgingData = Data(chip_id, train_sheet_num)
        test_x, test_y = AgingData.get_iterative_data()
        self.y = 100 * test_y
        self.x = np.array(range(len(self.y)))

        # 创建一个卡尔曼滤波器
    def fliter(self):
        kf = KalmanFilter(initial_state_mean=self.y[0],
                             n_dim_obs=1,
                             observation_covariance=20,
                             transition_covariance=0.005,
                    transition_matrices=1)
        # 使用卡尔曼滤波器对时间序列进行预测
        filtered_state_means, filtered_state_covariances = kf.filter(self.y)
        return filtered_state_means
        #return self.y 


# 绘制原始时间序列和卡尔曼滤波后的时间序列
#plt.plot(x, y, label='Original')
#plt.plot(x, filtered_state_means, label='Kalman Filtered')
#plt.legend()
#plt.show()