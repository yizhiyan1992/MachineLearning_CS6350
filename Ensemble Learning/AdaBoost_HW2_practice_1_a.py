import os
import pandas as pd
import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
class Node:
    def __init__(self,attribute,map):
        self.attribute=attribute
        self.map=map
#os.chdir('/Users/Zhiyan1992/Desktop/')
class AdaBoost():
    def __init__(self,tree_no,train_size,test_size):
        self.tree_no=tree_no
        self.trees=[]
        self.coefs=[]
        self.res=[]
        self.weight=np.array([np.zeros(train_size) for _ in range(self.tree_no)])
        self.score=np.array([np.zeros(train_size) for _ in range(self.tree_no)])
        self.accu=[]
        self.test_error=[]
        self.each_accu=[]
        self.each_test_error=[]

    def train(self,train,test):
        # add a new column for training samples as weight for each sample
        weight=pd.DataFrame(data=np.array([1/train.shape[0] for _ in range(train.shape[0])]),columns=['weight'])
        train=pd.concat([train,weight],axis=1)
        #iterate N steps, where N is the number of trees
        for index in range(self.tree_no):
            predict=self.decision_stump(train,test)
            em=self.cal_em(train['label'],predict,train['weight'])
            self.each_accu.append(em)
            alpha=self.cal_alpha(em)
            print('iteration time:',index,em,alpha)
            self.weight[index]=train['weight'].values
            self.update_weight(train['label'].values,predict,train['weight'].values,alpha)
            accu=self.train_accuracy(train['label'].values,index)
            self.accu.append(accu)
            print('current training accuracy: ',accu)
            accu_test=self.predict_test(test)
            self.test_error.append(accu_test)
            print('current testing accuracy: ', accu_test)
        print(self.each_test_error)
        print(self.each_accu)

    def decision_stump(self,train,y):
        test=[]
        error=1
        opt_feature=None
        opt_dict=None
        for col_name in train.columns:
            if col_name not in ['label','weight']:
                attribute={}
                temp={}
                for index in range(train.shape[0]):
                    temp.setdefault(train[col_name].values[index],{})
                    temp[train[col_name].values[index]].setdefault(train['label'].values[index],0)
                    temp[train[col_name].values[index]][train['label'].values[index]]+=train['weight'].values[index]
                for key,val in temp.items():
                    if len(val)==1:
                        attribute[key]=key
                        continue
                    if val[-1]>val[1]:
                        attribute[key]=-1
                    else:
                        attribute[key]=1
                err=0
                for index in range(train.shape[0]):
                    if train['label'].values[index]!=attribute[train[col_name].values[index]]:
                        err+=train['weight'].values[index]
                test.append(err)
                if err<error:
                    error=err
                    opt_feature=col_name
                    opt_dict=attribute
        #print(test,error)
        # add the current optimal decision stump into tree array
        tree=Node(opt_feature,opt_dict)
        self.trees.append(tree)
        predict=[]

        for index in range(train.shape[0]):
            predict.append(tree.map[train[tree.attribute].values[index]])
        count=0
        for index in range(y.shape[0]):
            if tree.map[y[tree.attribute].values[index]]==y['label'].values[index]:
                count+=1
        self.each_test_error.append(1-count/y.shape[0])
        self.res.append(np.array(predict))
        #return predicted result
        return np.array(predict)

    def cal_em(self,train,predict,weight):
        em=0
        for index in range(train.shape[0]):
            if train[index]!=predict[index]:
                em+=weight[index]
        return em

    def cal_alpha(self,em):
        alpha=0.5*math.log((1-em)/em)
        self.coefs.append(alpha)
        return alpha

    def update_weight(self,train,predict,weight,alpha):
        #inplace the weight
        for index in range(len(weight)):
            weight[index]=weight[index]*math.exp(-alpha*train[index]*predict[index])
        #normalize
        weight/=np.sum(weight)

        return weight

    def train_accuracy(self,label,iter):
        res=np.array([0 for _ in range(len(self.res[0]))])
        for index in range(len(self.res)):
            res=res+self.coefs[index]*self.res[index]
        self.score[iter]=res
        for index in range(len(self.res[0])):
            if res[index]>0:
                res[index]=1
            else:
                res[index]=-1
        res=res.astype(dtype=int)

        count = 0
        for index in range(len(self.res[0])):
            if res[index]==label[index]:
                count += 1
        return 1 - count / len(self.res[0])

    def predict_test(self,test):
        res=[0 for _ in range(test.shape[0])]
        for i in range(len(self.trees)):
            tree=self.trees[i]
            for index in range(test.shape[0]):
                #temp=tree.map[test[tree.attribute].values[index]]
                res[index]=res[index]+self.coefs[i]*tree.map[test[tree.attribute].values[index]]
        for index in range(test.shape[0]):
            if res[index]>0:
                res[index]=1
            else:
                res[index]=-1
        count = 0
        print(len(self.res[0]))
        for index in range(test.shape[0]):
            if res[index]==test['label'].values[index]:
                count += 1
        return 1 - count /test.shape[0]

    def plot(self):
        plt.figure(1)
        plt.subplot(211)
        plt.plot(range(len(self.accu)),self.accu,label='Training Set')
        plt.title('Final Error on all trees')
        plt.xlabel('Iteration steps')
        plt.ylabel('Prediction Error')
        plt.plot(range(len(self.test_error)), self.test_error,label='Testing Set')
        plt.legend()
        plt.subplot(212)
        plt.plot(range(len(self.each_accu)),self.each_accu,label='Training Set')
        plt.title('Error on each decision stump')
        plt.xlabel('Iteration steps')
        plt.ylabel('Prediction Error')
        plt.plot(range(len(self.each_test_error)), self.each_test_error,label='Testing Set')
        plt.legend()
        plt.tight_layout()
        #plt.savefig(r'/Users/Zhiyan1992/Desktop/res1.png')
        plt.show()
        File=np.array([np.array(self.each_accu),np.array(self.each_test_error)])
        File=pd.DataFrame(File)
        File.to_csv(r'adaboot_res.csv')

'''Main function'''
def main():
    Train=pd.read_csv(r'bank/train.csv',header=None)
    Test=pd.read_csv('bank/test.csv',header=None)
    Train.columns = Test.columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing' \
        , 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    Numeric = {'age': True, 'job': False, 'marital': False, 'education': False, 'default': False, 'balance': True,
               'housing': False,'loan': False, 'contact': False, 'day': True, 'month': False, 'duration': True, 'campaign': True,
               'pdays': True, 'previous': True, 'poutcome': False}
    Median={}
    #turn numeric values into binaries
    for col_name in Train.columns:
        if col_name not in ['label','weight'] and Numeric[col_name]:
            Median[col_name]=np.median(Train[col_name].values)
    print('Median value for each attribute: ',Median)
    for col_name in Train.columns:
        if col_name not in ['label','weight'] and Numeric[col_name]:
            for i in range(Train.shape[0]):
                if Train[col_name].values[i]<=Median[col_name]:
                    Train[col_name].values[i]=0
                else:
                    Train[col_name].values[i]=1

                if Test[col_name].values[i]<=Median[col_name]:
                    Test[col_name].values[i]=0
                else:
                    Test[col_name].values[i]=1
    #transform prediction into: yes--->+1/no--->-1
    Train['label'][Train['label']=='yes']=1
    Train['label'][Train['label']=='no']=-1
    Test['label'][Test['label']=='yes']=1
    Test['label'][Test['label']=='no']=-1

    clf=AdaBoost(100,Train.shape[0],Test.shape[0])
    clf.train(Train,Test)
    clf.plot()
if __name__=='__main__':
    main()
