import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import math
from scratch_65 import DecisionTree
import os
import copy
#os.chdir('/Users/Zhiyan1992/Desktop/')

class RandomForest():
    def __init__(self,tree_no,feature_no,sampling_rate):
        self.tree_no=tree_no
        self.feature_no=feature_no
        self.sampling_rate=sampling_rate
        self.feature_mapping=[]
        self.trees=[]
        self.medians=[]
        self.predict_train_res=[]
        self.predict_test_res=[]
        self.score_train=[]
        self.score_test=[]

    def train_model(self,train,test,Numeric):
        for index in range(self.tree_no):
            print('cur iteration: ',index)
            sub_train=copy.deepcopy(train)
            sub_test = copy.deepcopy(test)
            sub_train_2=copy.deepcopy(sub_train)
            #sub_train2 is for prediction accuracy
            sub_train=self.bagging(sub_train)
            tree=DecisionTree(sub_train,sub_test)
            self.trees.append(tree)
            median=tree.Numeric_processing(sub_train,Numeric)
            tree.Numeric_processing_test(sub_test, Numeric, median)
            self.medians.append(median)
            tree.Train_model(gain_type='Gini_Index',feature_no=self.feature_no)
            res=tree.Result_predict(sub_test)
            res_train=tree.Result_predict(sub_train_2)
            self.predict_train_res.append(res_train)
            self.predict_test_res.append(res)
            agg_res=self.predict(self.predict_test_res)
            agg_res_train = self.predict(self.predict_train_res)
            score_train=self.prediction_accuracy(sub_train_2['label'],agg_res_train)
            score=self.prediction_accuracy(sub_test['label'],agg_res)
            self.score_train.append(score_train)
            self.score_test.append(score)
            print('score on train',score_train)
            print('score on test',score)
        print(self.score_train,self.score_test)
        TRAIN=np.array(self.score_train)
        TEST=np.array(self.score_test)
        Data=pd.DataFrame(np.array([TRAIN,TEST]))
        Data.to_csv(r'C:/Users/Zhiyan/Desktop/res.csv')


    def predict(self,res):
        aggregate=[]
        predict=np.array(res)
        for index in range(predict.shape[1]):
            counter=Counter(predict[:,index])
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
        col=train.columns
        train = train.values
        for index in rand:
            sub_set.append(train[index, :])
        sub_set = np.array(sub_set)
        sub_set = pd.DataFrame(data=sub_set, columns=col)
        return sub_set


'''Main function'''
def main():
    Train=pd.read_csv(r'bank/train.csv',header=None)
    Test=pd.read_csv('bank/test.csv',header=None)
    Train.columns = Test.columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing' \
        , 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    Numeric = {'age': True, 'job': False, 'marital': False, 'education': False, 'default': False, 'balance': True,
               'housing': False,'loan': False, 'contact': False, 'day': True, 'month': False, 'duration': True, 'campaign': True,
               'pdays': True, 'previous': True, 'poutcome': False}
    RF=RandomForest(tree_no=1000,feature_no=6,sampling_rate=0.01)
    RF.train_model(Train,Test,Numeric)
if __name__=='__main__':
    main()
