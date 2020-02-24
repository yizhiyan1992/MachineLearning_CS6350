import os
import pandas as pd
import numpy as np
import math
from collections import Counter
from Decision_tree_adaboost import DecisionTree

#os.chdir(r'C:/Users/Zhiyan/Desktop/')
class AdaBoost():
    def __init__(self,tree_no):
        self.tree_no=tree_no
        self.trees=[]
        self.medians=[]
        self.coefs=[]

    def train_model(self,train,test,Numeric):
        #initializa weight, and add weight column to the sample matrix
        weight=pd.DataFrame(data=np.array([1/train.shape[0] for _ in range(train.shape[0])]),columns=['weight'])
        train=pd.concat([train,weight],axis=1)
        for index in range(self.tree_no):
            # train the Decision tree model with weighted samples
            tree=DecisionTree(train,test)
            self.trees.append(tree)
            median=tree.Numeric_processing(train,Numeric)
            self.medians.append(median)
            tree.Train_model(gain_type='Gini_Index')
            predict=tree.Result_predict(train)
            print(predict)
            em=self.cal_em(train.values[:,-2],predict,train.values[:,-1])
            print(em)
            coef=self.cal_coefficient(em)
            self.coefs.append(coef)
            weight=self.update_weight(train.values[:,-2],predict,train.values[:,-1],em)
            print(weight)


    def predict(self,test):
        res=[]
        for index in range(test.shape[0]):
            temp=self.predict_single()
            res.append(temp)
        return res

    def predict_single(self,test):
        # intput should be +1 -1
        res=0
        for index in range(self.tree_no):
            res+=self.trees[index].Predict*self.coefs[index]
        return res

    def cal_coefficient(self,em):
        coef=0.5*math.log((1-em)/em,2)
        return coef

    def cal_em(self,train,predict,weight):
        em=0
        for index in range(len(train)):
            if train[index]!=predict[index]:
                em+=weight[index]
        return em

    def update_weight(self,train,predict,weight,em):
        norm=0
        for index in range(len(train)):
            if train[index]!=predict[index]:
                norm+=weight[index]*math.exp(em)
            else:
                norm+=weight[index]*math.exp(-em)
        new_weight=[0 for _ in range(len(weight))]
        for index in range(len(weight)):
            if train[index]!=predict[index]:
                new_weight[index]=weight[index]*math.exp(em)/norm
            else:
                new_weight[index]=weight[index]*math.exp(-em)/norm
        return new_weight

'''Main function'''
Train=pd.read_csv(r'bank/train.csv',header=None)
Test=pd.read_csv('bank/test.csv',header=None)
Train.columns = Test.columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing' \
    , 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
Numeric = {'age': True, 'job': False, 'marital': False, 'education': False, 'default': False, 'balance': True,
           'housing': False,'loan': False, 'contact': False, 'day': True, 'month': False, 'duration': True, 'campaign': True,
           'pdays': True, 'previous': True, 'poutcome': False}

#transform prediction into: yes--->+1/no--->-1
Train['label'][Train['label']=='yes']=1
Train['label'][Train['label']=='no']=-1
Test['label'][Test['label']=='yes']=1
Test['label'][Test['label']=='no']=-1

GBDT=AdaBoost(tree_no=1)
GBDT.train_model(Train,Test,Numeric)
