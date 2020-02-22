import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import math
from Decision_Tree_bagging import DecisionTree

class RandomForest():
    def __init__(self,tree_no,sampling_rate):
        self.tree_no=tree_no
        self.sampling_rate=sampling_rate
        self.trees=[]
        self.medians=[]

    def train_model(self,train,test,Numeric):
        for index in range(self.tree_no):
            print('Current tree number: ',index)
            sub_train=self.bagging(train)
            tree=DecisionTree(sub_train,test)
            self.trees.append(tree)
            median=tree.Numeric_processing(sub_train,Numeric)
            self.medians.append(median)
            tree.Train_model(gain_type='Gini_Index')
        print(self.medians)

    def predict(self,test):
        res_test=[]
        for index in range(self.tree_no):
            median=self.medians[index]
            tree=self.trees[index]
            tree.Numeric_processing_test(test,Numeric,median)
            res_test.append(tree.Result_predict(test))
        res_test=pd.DataFrame(data=np.array(res_test))
        #res_test.to_csv(r'C:/Users/Zhiyan/Desktop/test222.csv')
        #rint(res_test)
        aggregate=[]
        res_test=res_test.values
        for col in range(res_test.shape[1]):
            counter=Counter(res_test[:,col])
            aggregate.append(counter.most_common(1)[0][0])
        return aggregate

    def prediction_accuracy(self,true_label,predict_label):
        total=len(true_label)
        count=0
        for index in range(total):
            if true_label[index]==predict_label[index]:
                count+=1
        return 1-count/total

    def bagging(self,train):
        size = int(self.sampling_rate * train.shape[0])
        # sampling with replacement
        rand = np.random.randint(0, train.shape[0], size)
        sub_set = []
        train = train.values
        for index in rand:
            sub_set.append(train[index, :])
        sub_set = np.array(sub_set)
        sub_set = pd.DataFrame(data=sub_set, columns=Train.columns)
        return sub_set

'''Main function'''
Train=pd.read_csv(r'bank/train.csv',header=None)
Test=pd.read_csv('bank/test.csv',header=None)
Train.columns = Test.columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing' \
    , 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
Numeric = {'age': True, 'job': False, 'marital': False, 'education': False, 'default': False, 'balance': True,
           'housing': False,'loan': False, 'contact': False, 'day': True, 'month': False, 'duration': True, 'campaign': True,
           'pdays': True, 'previous': True, 'poutcome': False}
RF=RandomForest(tree_no=10,sampling_rate=1)
RF.train_model(Train,Test,Numeric)
res=RF.predict(Test)
error=RF.prediction_accuracy(Test.values[:,-1],res)
print(res)
print(error)
