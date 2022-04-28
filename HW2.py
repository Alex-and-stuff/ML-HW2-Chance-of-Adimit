'''
NTHU EE Machine Learning HW2
Author:      陳家斌
Student ID:  110033534
'''
from asyncore import dispatcher_with_send
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
    cvt_train_data = BasisFunc(train_data, O1, O2)

    # Compute the design matrix 
    phi = cvt_train_data
    
    # posterior distribution over w given by p(w|t) = N(w|m_n,S_n)
    alpha = 1
    beta  = 1
    S_n_inv = alpha*np.identity(phi.shape[1])+beta*np.matmul(np.transpose(phi),phi)
    S_n = np.linalg.inv(S_n_inv)
    m_n = beta*np.matmul(np.matmul(S_n,np.transpose(phi)),train_data[:,3])

    # Predict test_data_feature
    cvt_test_data = BasisFunc(test_data_feature, O1, O2)
    y_BLRprediction = []
    for data in range(cvt_test_data.shape[0]):
        y = 0
        for j in range(cvt_test_data.shape[1]):
            y+=m_n[j]*cvt_test_data[data,j]
        y_BLRprediction.append(y)

    return y_BLRprediction 


# do not change the name of this function
def MLR(train_data, test_data_feature, O1=5, O2=5):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    cvt_train_data = BasisFunc(train_data, O1, O2)

    # Compute the design matrix 
    phi = cvt_train_data

    # Generate maximum likelihood weight from solving the gradient
    # w_ml = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi),phi)),np.transpose(phi)),train_data[:,3])

    # Generate maximum likelihood weight with regularization (lambda term)
    lam = .5
    w_ml = np.matmul(np.matmul(np.linalg.inv(lam/2*np.identity(phi.shape[1])+np.matmul(np.transpose(phi),phi)),np.transpose(phi)),train_data[:,3])

    # Predict test_data_feature
    cvt_test_data = BasisFunc(test_data_feature, O1, O2)
    y_MLRprediction = []
    for data in range(cvt_test_data.shape[0]):
        y = 0
        for j in range(cvt_test_data.shape[1]):
            y+=w_ml[j]*cvt_test_data[data,j]
        y_MLRprediction.append(y)
        
    return y_MLRprediction 


def CalMSE(data, prediction):

    squared_error = (data - prediction) ** 2
    sum_squared_error = np.sum(squared_error)
    mean__squared_error = sum_squared_error/prediction.shape[0]

    return mean__squared_error

def BasisFunc(data, O_1, O_2):
    # Calculate parameters
    x1_max = max(data[:,0])
    x1_min = min(data[:,0])
    x2_max = max(data[:,1])
    x2_min = min(data[:,1])
    s1 = (x1_max-x1_min)/(O_1-1)
    s2 = (x2_max-x2_min)/(O_2-1)
    x1 = data[:,0]
    x2 = data[:,1]
    x3 = data[:,2]

    # Normalize data x1 and x2
    # x1 = Normalize(x1, x1_max, x1_min)
    # x2 = Normalize(x2, x2_max, x2_min)
    
    # Obtain # of rows in the train data
    data_row = data.shape[0]

    # Create output matrix
    output = np.zeros((data_row, O_1*O_2+2))

    # k Goes from 1-25, expands the data from 2+experience+bias to 25+experience+bias
    for row in range(data_row):
        for i in range(1,O_1+1,1):
            for j in range(1,O_2+1,1):
                k = O_2*(i-1)+j
                mu_i = s1*(i-1)+x1_min
                mu_j = s2*(j-1)+x2_min
                phi_k = math.exp(-math.pow(x1[row]-mu_i, 2)/2/s1/s1-math.pow(x2[row]-mu_j,2)/2/s2/s2)
                output[row][k-1] = phi_k
        output[row][k+1-1] = x3[row]
        output[row][k+2-1] = 1
    return output

def Normalize(data, max, min):
    data = (data-min)/(max-min)
    return data
    


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

    predict_BLR = BLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    predict_MLR = MLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    # print('MSE of MLR= {e2}.'.format(e2=CalMSE(predict_MLR, data_test_label)))
    # print('MSE of BLR = {e1}.'.format(e1=CalMSE(predict_BLR, data_test_label)))
    print('MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(e1=CalMSE(predict_BLR, data_test_label), e2=CalMSE(predict_MLR, data_test_label)))


if __name__ == '__main__':
    main()


