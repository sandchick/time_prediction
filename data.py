import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Train Data Valid
# If TrainMap[i][j] = 1, Chip_i_Excel[sheet_num = j] is valid
TrainMap = [[1,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,0,0,1,1,0,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,0,0,1,1,1,1,0,0,0],
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
        for sheet in range(train_sheet_num):
            data = pd.read_excel(train_data_file, sheet_name = sheet, header = None)
            data_array = np.array(data)
            self.train_data.append(data_array)
        data  = pd.read_excel(test_data_file, header = None)
        self.test_data = np.array(data)

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
        return self.test_array[:,0], self.test_array[:,1]

    def debug(self):
        """just for debug
        """
        for i in range(len(self.train_data)):
            print(self.train_data[i].shape[0], self.train_data[i].shape[1])        
        print(self.train_array[-1 ,:])
        print(self.test_array)
