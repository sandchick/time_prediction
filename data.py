import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# Train Data Valid
# If TrainMap[i][j] = 1, Chip_i_Excel[sheet_num = j] is valid
TrainMap = [[1,0,0,1,1,1,1,1,1,1,1,1,1,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,0,0,1,1,0,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
            [1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,0,1,1,0,1,1,1,1,1,1]]
class Data:
    """ Pre-Process Data for model training and testing
    """    

    def __init__(self, chip_id, train_sheet_num):
        """[summary] find train/test files

        Parameters
        ----------
        chip_id : [type] int
            [description] chip id
        train_sheet_num : [type] int
            [description] sheet num of train data file
        """        
        self.chip_id = chip_id
        train_data_file = '../data/Chip' + str(self.chip_id) + 'TrainFilt.xlsx' 
        test_data_file = '../data/Chip' + str(self.chip_id) + 'TestFilt.xlsx' 
        self.train_data = []
        self.threshold_AF = 0.032
        for sheet in range(train_sheet_num):
            data = pd.read_excel(train_data_file, sheet_name = sheet, header = None)
            data_array = np.array(data)
            self.train_data.append(data_array)
        data  = pd.read_excel(test_data_file, header = None)
        self.test_data = np.array(data)

    def get_train_data_gause(self):
        train_data_list = []
        for sheet in range(len(self.train_data)):
            if (TrainMap[self.chip_id][sheet] == 0 or sheet == 0):
                continue
            #test data from RO
            #if sheet == 0:
            #    continue
            #test data from RO
            mean_ini = np.zeros(self.train_data[sheet].shape[0]) 
            for i in range(self.train_data[sheet].shape[0]):
                mean_ini[i] = np.mean(self.train_data[sheet][i][10:])
            gause_value_ini = gaussian_filter1d(mean_ini,3)
            RUL = np.zeros(self.train_data[sheet].shape[0]) 
            AFR = np.zeros(self.train_data[sheet].shape[0]) 
            for i in range(len(mean_ini)):
                AFR[i] = abs(gause_value_ini[i]-self.threshold_AF)
            AFR = AFR.tolist()
            FT = AFR.index(min(AFR))
            #print (f"failure threshold = {FT}")
            mean = np.zeros(FT)
            for i in range(FT):
                mean[i] = np.mean(self.train_data[sheet][i][10:])
                fail_time = self.train_data[sheet][FT][1]
                RUL[i] = fail_time - self.train_data[sheet][i][1]
            gause_value = gaussian_filter1d(mean,3)
            train_data_list_single = list(zip(gause_value,RUL))
            train_data_list.extend(train_data_list_single)
            #if(np.isnan(train_data_list).any()):
            #    print (f"traindata,chip={self.chip_id},sheet={sheet}")
        self.train_gause_array = np.array(train_data_list)
        #print(np.shape(self.train_gause_array),np.shape(train_data_list))
        return self.train_gause_array[:,0], self.train_gause_array[:,1]

    def get_iterative_data(self):
        iterative_width = 50
        train_data_list = []
        mean = np.zeros(self.test_data.shape[0])
        for i in range(self.test_data.shape[0]):
            mean[i] = np.mean(self.test_data[i][10:])
        #gause_value = gaussian_filter1d(mean,3)
        gause_value = mean
        AF_timestamp = list(zip(gause_value,self.test_data[:,1]))
        AF_timestamp_fill = np.zeros((1,2))
        #print(np.shape(AF_timestamp_new))
        #print(f"before = {AF_timestamp}")
        AF_timestamp_list = [[x for x in tup] for tup in AF_timestamp]
        #print(f"before_list = {AF_timestamp_list}")
        time_weights = np.zeros(self.test_data.shape[0] - 1)
        for i in range(self.test_data.shape[0]-1):
            time_weights[i] = int(AF_timestamp[i + 1][1] - AF_timestamp[i][1])
        #print(len(time_weights),self.test_data.shape[0])
        #print(f"time weight = {time_weights}")
        for i in range(len(time_weights)):
            if time_weights[i] == 1:
                continue
            #for time_weight in range(int(time_weights[i])):
                #print(f"time weight = {time_weight}")
            AF_fill = np.zeros((int(time_weights[i])-1,2))
            for time_increase in range (int(time_weights[i])-1):
                AF_fill[time_increase][0] = AF_timestamp[i][0] + ((time_increase + 1) * (AF_timestamp[i+1][0] - AF_timestamp[i][0]))/time_weights[i]
                AF_fill[time_increase][1] = AF_timestamp[i][1] + time_increase + 1
            #print(f"AF_fill = {AF_fill}")
            AF_timestamp_fill = np.concatenate((AF_timestamp_fill,AF_fill),axis=0)
        #print(f"after = {AF_timestamp_fill}")
        AF_timestamp_fill = np.concatenate((AF_timestamp_fill,AF_timestamp_list),axis=0) 
        AF_timestamp_fill = AF_timestamp_fill[np.argsort(AF_timestamp_fill[:,1])]
        AF_timestamp_fill = np.delete(AF_timestamp_fill, 0 , axis=0) 
        #print(f"after = {AF_timestamp_fill}")
        for i in range (AF_timestamp_fill.shape[0]-iterative_width):
            vector = np.zeros(iterative_width + 1)
            vector[0:iterative_width] = [row[0] for row in AF_timestamp_fill[ i : i + iterative_width]]
            vector[iterative_width] = AF_timestamp_fill[i + iterative_width][0]
            train_data_list.append(vector)
        train_data_array = np.array(train_data_list)
        return train_data_array[:,:-1], train_data_array[:,-1]



    def get_test_data_gause(self):
        mean_ini = np.zeros(self.test_data.shape[0]) 
        RUL = np.zeros(self.test_data.shape[0]) 
        for i in range(self.test_data.shape[0]):
            mean_ini[i] = np.mean(self.test_data[i][10:])
        gause_value_ini = gaussian_filter1d(mean_ini,3)
        AFR = np.zeros(self.test_data.shape[0])
        for i in range(self.test_data.shape[0]):
            AFR[i] = abs(gause_value_ini[i]-self.threshold_AF)
        #print(AFR)
        AFR = AFR.tolist()
        #print(AFR)
        FT = AFR.index(min(AFR))
        #print (f"failure threshold = {FT}")
        mean = np.zeros(FT) 
        for i in range(FT):
            mean[i] = np.mean(self.test_data[i][10:])
            fail_time = self.test_data[FT][1]
            RUL[i] = fail_time - self.test_data[i][1]
        #print (f"fail time = {fail_time}")
        #print (f"RUL")
        gause_value = gaussian_filter1d(mean,3)
        test_data_list=list(zip(gause_value,RUL))
       # if(np.isnan(test_data_list).any()):
       #     print (f"testdata,chip={self.chip_id}")
        self.test_gause_array = np.array(test_data_list)
        return self.test_gause_array[:,0], self.test_gause_array[:,1], self.threshold_AF
    
    def get_test_data_from_RO(self):
        mean_ini = np.zeros(self.train_data[0].shape[0]) 
        RUL = np.zeros(self.train_data[0].shape[0]) 
        for i in range(self.train_data[0].shape[0]):
            mean_ini[i] = np.mean(self.train_data[0][i][10:])
        gause_value_ini = gaussian_filter1d(mean_ini,3)
        AFR = np.zeros(self.train_data[0].shape[0])
        for i in range(self.train_data[0].shape[0]):
            AFR[i] = abs(gause_value_ini[i]-0.01)
        #print(AFR)
        AFR = AFR.tolist()
        #print(AFR)
        FT = AFR.index(min(AFR))
        print (f"failure threshold ={self.chip_id} , {FT}")
        mean = np.zeros(FT) 
        for i in range(FT):
            mean[i] = np.mean(self.train_data[0][i][10:])
            fail_time = self.train_data[0][FT][1]
            RUL[i] = fail_time - self.train_data[0][i][1]
        gause_value = gaussian_filter1d(mean,3)
        train_data_list = list(zip(gause_value,RUL))
        self.test_gause_array_RO = np.array(train_data_list)
        return self.test_gause_array_RO[:,0], self.test_gause_array_RO[:,1]

    def get_train_data(self):
        """return data for svr training

        Returns
        -------
        train_array[:, :-1] [type] array
                     [description] training data input
        train_array[:, -1]  [type] array
                     [description] training data output                      
        """        
        train_data_list = []
        for sheet in range(len(self.train_data)):
            if (TrainMap[self.chip_id][sheet] == 0):
                continue
            for i in range(self.train_data[sheet].shape[0]):
                for j in range(self.train_data[sheet].shape[1]-10):
                    fail_time = max(self.train_data[sheet][:,1])
                    vector = np.zeros(2)
                    vector[0] = self.train_data[sheet][i][10+j]
                    vector[1] = fail_time - self.train_data[sheet][i][1]
                    train_data_list.append(vector)
        self.train_array = np.array(train_data_list)


        return self.train_array[:,0], self.train_array[:,1]

    def get_test_data(self):
        """return get data for svr testing

        Returns
        -------
        test_array[:, :-1] [type] array
                     [description] testing data input
        test_array[:, -1]  [type] array
                     [description] testing data output                      
        """
        test_data_list = []
        for i in range(self.test_data.shape[0]):
            for j in range(self.test_data.shape[1]-10):
                fail_time = max(self.test_data[:,1])
                vector = np.zeros(2)
                vector[0] = self.test_data[i][10+j]
                vector[1] = fail_time - self.test_data[i][1]
                test_data_list.append(vector)
        self.test_array = np.array(test_data_list)
        #rul = np.zeros(self.test_data.shape[0])
        #fail_time_test = max(self.test_data[:,1])
        ##print (fail_time_test)
        ##print (self.test_data.shape[0])
        #for i in range(self.test_data.shape[0]):
        #    #print (self.test_data[i][1])
        #    rul[i] = fail_time_test - self.test_data[i][1]
        #    test_data_list.append(self.test_data[i])
        #    #if(np.isnan(test_data_list).any()):
        #    #    print (f"testdata,chip={self.chip_id},y={i}") 
        #self.test_array = np.array(test_data_list)
        #self.test_array = np.insert(self.test_array,1,values=rul,axis=1)
        #print(f"origin test{np.shape(self.test_array)}")
        return self.test_array[:,0], self.test_array[:,1]

    def debug(self):
        """just for debug
        """
        for i in range(len(self.train_data)):
            print(self.train_data[i].shape[0], self.train_data[i].shape[1])        
        print(self.train_array[-1 ,:])
        print(self.test_array)
