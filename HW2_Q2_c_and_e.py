import os
import pandas as pd
import numpy as np
import random
import math
from scratch_71 import BagTree
from scratch_66 import RandomForest

'''Main function'''
def sample_variance(predict):
    print(predict)
    predict_new=[]
    File=pd.DataFrame(predict)
    File.to_csv(r'C:/Users/Zhiyan/Desktop/bias_var.csv')
    mean=np.mean(predict,axis=0)
    var=np.zeros((mean.shape))
    print(mean)
    for i in range(predict.shape[0]):
        print(i)
        var=var+(predict[i,:]-mean)*(predict[i,:]-mean)
    var=var/predict.shape[0]
    print(var)
    var=np.mean(var)
    print('variance',var)

def sample_bias(predict,true):
    for i in range(len(true)):
        if true[i] == 'yes':
            true[i] = 1
        else:
            true[i]= -1
    print(true)
    mean = np.mean(predict, axis=0)
    bias = np.zeros((mean.shape))
    bias=bias+(true-mean)*(true-mean)
    bias=np.mean(bias)
    print('bias',bias)

def var_bias(Train,Test,Numeric):
    res_total=[]
    for iter in range(20):
        print(iter)
        #BT = BagTree(tree_no=15, sampling_rate=0.2)
        #BT.train_model(Train, Test, Numeric)
        #res = BT.predict(Test, Numeric)
        RF = RandomForest(tree_no=20, feature_no=2, sampling_rate=0.01)
        res=RF.train_model(Train, Test, Numeric)
        res_total.append(res)
        #error = BT.prediction_accuracy(Test.values[:, -1], res)
    res_total = np.array(res_total)
    for i in range(res_total.shape[0]):
        for j in range(res_total.shape[1]):
            if res_total[i][j] == 'yes':
                res_total[i][j] = 1
            else:
                res_total[i][j] = -1
    res_total=res_total.astype(int)
    sample_variance(res_total)
    sample_bias(res_total,Test['label'].values)

def main():
    Train=pd.read_csv(r'bank/train.csv',header=None)
    Test=pd.read_csv('bank/test.csv',header=None)
    Train.columns = Test.columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing' \
        , 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    Numeric = {'age': True, 'job': False, 'marital': False, 'education': False, 'default': False, 'balance': True,
           'housing': False,'loan': False, 'contact': False, 'day': True, 'month': False, 'duration': True, 'campaign': True,
           'pdays': True, 'previous': True, 'poutcome': False}

    var_bias(Train,Test,Numeric)

if __name__=='__main__':
    main()
