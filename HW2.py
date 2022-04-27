'''
NTHU EE Machine Learning HW2
Author:      陳家斌
Student ID:  110033534
'''
import numpy as np
import pandas as pd
import math
import scipy.stats
import argparse


# do not change the name of this function
def BLR(train_data, test_data_feature, O1=5, O2=5):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''


    return y_BLRprediction 


# do not change the name of this function
def MLR(train_data, test_data_feature, O1=5, O2=5):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    cvt_train_data = BasisFunc(train_data, O1, O2)


    return y_MLRprediction 


def CalMSE(data, prediction):

    squared_error = (data - prediction) ** 2
    sum_squared_error = np.sum(squared_error)
    mean__squared_error = sum_squared_error/prediction.shape[0]

    return mean__squared_error

def BasisFunc(data, O_1, O_2):
    x1_max = max(data[:,0])
    x1_min = min(data[:,0])
    x2_max = max(data[:,1])
    x2_min = min(data[:,1])
    s1 = (x1_max-x1_min)/(O_1-1)
    s2 = (x2_max-x2_min)/(O_2-1)
    x1 = data[:,0]
    x2 = data[:,1]
    x3 = data[:,2]
    data_row = data.shape[0]

    output = np.zeros((data_row, O_1*O_2+2))
    # k Goes from 1-25, 
    # expands the data from 2+experience+bias to 25+experience+bias
    for row in range(data_row):
        for i in range(1,O_1+1,1):
            for j in range(1,O_2+1,1):
                k = O_2*(i-1)+j
                mu_i = s1*(i-1)+x1_min
                mu_j = s2*(j-1)+x2_min
                phi_k = math.exp(-math.pow(x1[row]-mu_i, 2)/2*s1*s1-math.pow(x2[row]-mu_j,2)/2*s2*s2)
                output[row][k-1] = phi_k
        output[row][k+1-1] = x3[row]
        output[row][k+2-1] = 1
    
    return output


    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-O1', '--O_1', type=int, default=5)
    parser.add_argument('-O2', '--O_2', type=int, default=5)
    args = parser.parse_args()
    O_1 = args.O_1
    O_2 = args.O_2
    
    data_train = pd.read_csv('Training_set.csv', header=None).to_numpy()    # 300 train data (300, 4)
    data_test = pd.read_csv('Validation_set.csv', header=None).to_numpy()   # 100 test data (100, 4)
    data_test_feature = data_test[:, :3]                                    # Features of the dataset (100, 3)
    data_test_label = data_test[:, 3]                   

    '''
    Prediction using :
    1. Baysian linear regression method (BLR)
    2. Maximum Likelihood method (ML)
    '''
    BasisFunc(data_train, O_1, O_2)
    
    # predict_BLR = BLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    # predict_MLR = MLR(data_train, data_test_feature, O1=O_1, O2=O_2)

    # print('MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(e1=CalMSE(predict_BLR, data_test_label), e2=CalMSE(predict_MLR, data_test_label)))


if __name__ == '__main__':
    main()


