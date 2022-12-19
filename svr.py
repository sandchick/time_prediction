import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

np.set_printoptions(threshold = np.inf)

class SVRPredictor:
    """Aging Predictor using Support Vector Regression
    """    
    
    def __init__(self, chip_id):
        """init model(with user-define parameters)

        Parameters
        ----------
        chip_id : [type] int
            [description] chip id
        """
        self.chip_id = chip_id
        self.model = SVR(kernel = 'rbf', C = 1000)
    
    def train(self, train_data_x, train_data_y):
        """train model

        Parameters
        ----------
        train_data_x : [type] ndarray
            [description] train data input
        train_data_y : [type] ndarray
            [description] train data output
        """
        self.model.fit(train_data_x, train_data_y )

    def predict(self, test_data_x):
        """predict by trained model

        Parameters
        ----------
        test_data_x : [type] ndarray
            [description] test data input

        Returns
        -------
        [type] ndarray
            [description] predict result with test data in
        """        
        return self.model.predict(test_data_x)

    def error_analysis(self, test_data_y, predict_data_y):
        """Analysis error rate

        Parameters
        ----------
        test_data_y : [type] ndarray(n*12)
            [description] accurate output
        predict_data_y : [type] ndarray(n*1)
            [description] predict output

        Returns
        -------
        [type] ndarray(n*1)
            [description] MSE of predict and accurate output
        """
        err0 = [] # MSE
        #err1 = [] # AE
        for i in range(test_data_y.shape[0]):
            if i == 0:
                continue
            sum0 = 0
            #sum1 = 0
            time_pre = predict_data_y[i]
            time_test = test_data_y[i][0]
            sum0 = (time_pre - time_test) * (time_pre - time_test) / time_pre / time_pre 
            err0.append(np.sqrt(sum0/12))
            #err1.append(np.absolute(sum1)/12)
        err0 = np.array(err0)
        #err1 = np.array(err10        #if opt == 1:
        #    return err1
        #else:
        #    return err0
        return err0

    def draw(self, train_data_x, train_data_y, test_data_x, test_data_y, dopt):
        """draw picture to analysis

        Parameters
        ----------
        train_data_x : [type] ndarray(n*1)
            [description] train data input
        train_data_y : [type] ndarray(n*1)
            [description] train data output
        test_data_x : [type] ndarray(n*1)
            [description] test data input
        test_data_y : [type] ndarray(n*12)
            [description] test data output
        """        
        fig = plt.figure(dpi = 600)

        # ax1 = plt.axes(projection='3d')
        # ax1.scatter3D(train_data_x[:, 0], train_data_x[:, 1], train_data_y, cmap = 'b')
        # ax1.plot3D(train_data_x[:, 0], train_data_x[:, 1], predict_data_y, color = 'black')
        # plt.savefig('../img/compare3D.jpg')
        # plt.close()

        # Train Dataset
        predict_data_y = self.predict(train_data_x)
        plt.plot(np.array(range(len(train_data_y))), train_data_y, color = 'red', label = 'train')
        plt.plot(np.array(range(len(predict_data_y))), predict_data_y, color = 'black', label = 'predict')
        plt.xlabel('points')
        plt.ylabel('Remain Useful Life')
        plt.title('SVR Predict Result on Chip ' + str(self.chip_id) + ' Train Dataset')
        plt.legend()
        plt.savefig('../img/self_validation/svr_chip'+str(self.chip_id)+'_train.png')
        plt.close()

        # Test Dataset
        predict_data_y = self.predict(test_data_x)
        colors = cm.rainbow(np.linspace(0, 1, test_data_y.shape[1]))
        for i in range(test_data_y.shape[1]):
            plt.plot(np.array(range(test_data_y.shape[0])), test_data_y[:, i], color = colors[i])
        plt.plot(np.array(range(len(predict_data_y))), predict_data_y, color = 'black', label = 'predict')
        plt.xlabel('points')
        plt.ylabel('Delta delay')
        plt.title('SVR Predict Result on Chip ' + str(self.chip_id) + ' Test Dataset')
        plt.legend()
        plt.savefig('../img/self_validation/svr_chip'+str(self.chip_id)+'_test.png')
        plt.close()

        # Error Analysis
        err = self.error_analysis(test_data_y, predict_data_y)
        # print(err)
        if dopt == 1:
            plt.plot(np.array(range(int(0.8*len(err)), len(err))), err[int(0.8*len(err)):], color = 'red', label = 'MSE')
        else:
            plt.plot(np.array(range(10, len(err))), err[10:], color = 'red', label = 'MSE')
        plt.xlabel('Points')
        plt.ylabel('MSE')
        plt.title('MSE of SVR on Chip ' + str(self.chip_id) + ' Test Dataset')
        plt.legend()
        plt.savefig('../img/self_validation/svr_chip'+str(self.chip_id)+'_mse'+str(dopt)+'.png')
        plt.close()

        ## Error Analysis
        #err = self.error_analysis(test_data_y, predict_data_y, 1)
        ## print(err)
        #if dopt == 1:
        #    plt.plot(np.array(range(int(0.8*len(err)), len(err))), err[int(0.8*len(err)):], color = 'red', label = 'AE')
        #else:
        #    plt.plot(np.array(range(10, len(err))), err[10:], color = 'red', label = 'AE')
        #plt.xlabel('Points')
        #plt.ylabel('AE')
        #plt.title('Absolute Error of SVR on Chip ' + str(self.chip_id) + ' Test Dataset')
        #plt.legend()
        #plt.savefig('../img/self_validation/svr_chip'+str(self.chip_id)+'_ae'+str(dopt)+'.png')
        #plt.close()