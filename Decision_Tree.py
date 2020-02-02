import pandas as pd
import numpy as np
import math
from collections import Counter

class Node:
    def __init__(self,attribute):
        self.attribute=attribute
        self.child=[]

class DecisionTree:
    def __init__(self,train,test):
        self.train=train
        self.train_x=train.copy()
        del self.train_x['label']
        self.train_y=train['label']
        print(self.train_y)
        self.test=test
        self.sample_size=len(self.train_y)

    def InformationGain(self,col_feature,col_label):
        val_root=self.Gain([],col_label,'Majority_Error')
        val_child=self.Gain(col_feature,col_label,'Majority_Error')

        info=val_root-val_child
        return info

    # Including three types: 1) Entropy, 2) Majority Error and 3) Gini index
    def Gain(self,col_feature,col_label,gain):
        Dict = {}
        if len(col_feature)==0: # calculate the gain for root node
            Dict['counter']=Counter(col_label)
        else:
            for index in range(len(col_feature)):
                Dict.setdefault(col_feature[index],[])
                Dict[col_feature[index]].append(col_label[index])
            # for each class, count the type of labels
            for key,val in Dict.items():
                Dict[key]=Counter(val)
        #calculation
        if gain=='Entropy':
            #print(Dict,col_feature,col_label)
            entro = []
            for Class in Dict.keys():
                samples = Dict[Class]
                samples_sum = sum(samples.values())
                cur_val = 0
                for label,val in samples.items():
                    cur_val+=-val/samples_sum*math.log(val/samples_sum,2)
                cur_val=cur_val*(samples_sum/self.sample_size)
                entro.append(cur_val)
            #print(entro)
            return sum(entro)
        elif gain=='Gini_Index':
            entro = []
            for Class in Dict.keys():
                samples = Dict[Class]
                samples_sum = sum(samples.values())
                cur_val = 0
                for label, val in samples.items():
                    cur_val += (val / samples_sum)**2
                cur_val=1-cur_val
                cur_val =cur_val * (samples_sum / self.sample_size)
                entro.append(cur_val)
            return sum(entro)
        elif gain=='Majority_Error':
            entro = []
            for Class in Dict.keys():
                samples = Dict[Class]
                samples_sum = sum(samples.values())
                cur_val = 1
                for label, val in samples.items():
                    cur_val=min(cur_val,val/samples_sum)
                cur_val =cur_val * (samples_sum / self.sample_size)
                entro.append(cur_val)
            #calculation for Majority error is errorneous! Need to fix later!!!!
            return sum(entro)
    def recursion(self,cur_features):
        name=[]
        information_gain=[]
        for key,val in cur_features.items():
            name.append(key)
            val=self.InformationGain(val,self.train_y)
            information_gain.append(val)
        print(name,information_gain)
        return



Train=pd.read_csv(r'C:/Users/Zhiyan/Desktop/MLassignments/New folder/car/train.csv',header=None)
Train=pd.DataFrame([[1,1,2,3,3,3,2,1,1,3,1,2,2,3],[1,1,1,2,3,3,3,2,3,2,2,2,1,2],[1,1,1,1,2,2,2,1,2,2,2,1,2,1],[1,2,1,1,1,2,2,1,1,1,2,2,1,2],[0,0,1,1,1,0,1,0,1,1,1,1,1,0]])
Train=Train.T
Test=[]
print('total training size: ',Train.shape)
# transfer traning data into a dict format
#col_name=['buying','maint','doors','persons','lug_boot','safety','label']
col_name=['O','T','H','W','label']
Train_dict={}
for index,name in enumerate(col_name):
    Train_dict[name]=Train.values[:,index]

col=Train.values[:,0]
label=Train.values[:,-1]
Tree=DecisionTree(Train_dict,Test)

#print(Tree.InformationGain(col,label))
print(Tree.recursion(Tree.train_x))
