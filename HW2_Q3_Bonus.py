import os
import pandas as pd
import numpy as np
import random
import math
from collections import Counter
from Decision_Tree_bagging import DecisionTree
from Bagged_trees_HW2_Q2_b import BagTree
from AdaBoost_HW2_practice_1_a import AdaBoost
from Random_Forest_HW2_Q2_d import RandomForest

os.chdir(r'C:/Users/Zhiyan/Desktop/')
'''Main'''
#preprocessing the data
Data=pd.read_csv(r'default_of_credit_card_clients.csv',index_col=0)
Data.rename(columns={'Y':'label'},inplace=True)
Numeric={'X1':True,'X2':False,'X3':False,'X4':False,'X5':True,'X6':True,'X7':True,'X8':True,'X9':True,'X10':True,'X11':True, \
         'X12': True,'X13':True,'X14':True,'X15':True,'X16':True,'X17':True,'X18':True,'X19':True,'X20':True,'X21':True,'X22':True,'X23':True,}
print(type(Data.values[0,1]))
print(np.median(Data['X1']))
#shuffule the data and split into train and test dataset
Data=Data.sample(frac=1,random_state=42)
Data.set_index(pd.Index(range(Data.shape[0])),inplace=True)
train=Data.loc[:23999]
test=Data.loc[24000:]
train.set_index(pd.Index(range(train.shape[0])),inplace=True)
test.set_index(pd.Index(range(test.shape[0])),inplace=True)
print('The size of training data: ',train.shape,'The size of testing data: ',test.shape)

''' Decision Tree Model'''
def DT():
    tree=DecisionTree(train,test)
    median=tree.Numeric_processing(train,Numeric)
    tree.Numeric_processing_test(test,Numeric,median)
    tree.Train_model(gain_type='Gini_Index')
    res1=tree.Result_predict(train)
    res2=tree.Result_predict(test)
    print(res1,res2)
    print(tree.Prediction_accuracy(train['label'].values,res1))
    print(tree.Prediction_accuracy(test['label'].values,res2))

''' Adaboost'''
def AdaBoost():

    Median={}
    #turn numeric values into binaries
    for col_name in train.columns:
        if col_name not in ['label','weight'] and Numeric[col_name]:
            Median[col_name]=np.median(train[col_name].values)
    print('Median value for each attribute: ',Median)
    for col_name in train.columns:
        if col_name not in ['label','weight'] and Numeric[col_name]:
            for i in range(train.shape[0]):
                if train[col_name].values[i]<=Median[col_name]:
                    train[col_name].values[i]=0
                else:
                    train[col_name].values[i]=1
    for col_name in test.columns:
        if col_name not in ['label', 'weight'] and Numeric[col_name]:
            for i in range(test.shape[0]):
                if test[col_name].values[i] <= Median[col_name]:
                    test[col_name].values[i] = 0
                else:
                    test[col_name].values[i] = 1
    #transform prediction labels into: 1--->+1/0--->-1
    train['label'][train['label']==1]=1
    train['label'][train['label']==0]=-1
    test['label'][test['label']==1]=1
    test['label'][test['label']==0]=-1
    clf=AdaBoost(100,train.shape[0],test.shape[0])
    clf.train(train,test)
    clf.plot()

'''Bagged Trees'''
def BaggedTrees():
    BT = BagTree(tree_no=100, sampling_rate=0.005)
    BT.train_model(train, test, Numeric)
    res = BT.predict(test, Numeric)
    error = BT.prediction_accuracy(test.values[:, -1], res)
    print(res)
    print(error)

'''Random Forest'''
def Rf():
    RF = RandomForest(tree_no=100, feature_no=4, sampling_rate=0.01)
    RF.train_model(train, test, Numeric)

def main():
    '''run one model each time'''
    DT()
    #AdaBoost()
    #BaggedTrees()
    #Rf()
if __name__=='__main__':
    main()
