import pandas as pd
import numpy as np
import math
from collections import Counter

class Node:
    def __init__(self,name,attribute,depth):
        self.name=name
        self.depth=depth
        self.split_attribute=attribute
        self.child=[]

class DecisionTree:
    def __init__(self,train_matrix,test):
        self.root=None
        self.max_depth=6
        self.train_matrix=train_matrix
        self.test=test
        self.train_sample_size=self.train_matrix.shape[0]
        self.train_feature_size=self.train_matrix.shape[1]

    def Train_model(self,gain_type):
        self.recursion(self.train_matrix,self.root,1,gain_type)
        return

    def InformationGain(self,col_feature,col_label,gain_type):
        val_root=self.Gain([],col_label,gain_type)
        val_child=self.Gain(col_feature,col_label,gain_type)
        info=val_root-val_child
        return info

    # Including three types: 1) Entropy, 2) Majority Error and 3) Gini index
    def Gain(self,col_feature,col_label,gain):
        Dict = {}
        if len(col_feature)==0: # calculate the gain for father node
            Dict['counter']=Counter(col_label)
        else:
            for index in range(len(col_feature)):
                Dict.setdefault(col_feature[index],[])
                Dict[col_feature[index]].append(col_label[index])
            for key,val in Dict.items():
                Dict[key]=Counter(val)

        if gain=='Entropy':
            entro = []
            for Class in Dict.keys():
                samples = Dict[Class]
                samples_sum = sum(samples.values())
                cur_val = 0
                for label,val in samples.items():
                    cur_val+=-val/samples_sum*math.log(val/samples_sum,2)
                cur_val=cur_val*(samples_sum/self.train_sample_size)
                entro.append(cur_val)
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
                cur_val =cur_val * (samples_sum / self.train_sample_size)
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
                if cur_val==1: #when node is pure
                    cur_val=0
                cur_val =cur_val * (samples_sum / self.train_sample_size)
                entro.append(cur_val)
            return sum(entro)


    def recursion(self,cur_samples,node,depth,gain_type):
        name=[]
        information_gain=[]
        # select the feature with max information gain value
        for col in cur_samples.columns:
            if col!='label':
                name.append(col)
                val=self.InformationGain(cur_samples[col].values,cur_samples['label'].values,gain_type)
                information_gain.append(round(val, 3))
        del_feature=name[information_gain.index(max(information_gain))]
        if not node:
            self.root=Node('root',del_feature,1)
            node=self.root

        else:
            node.split_attribute=del_feature
        node.depth=depth

        Classes=set(cur_samples[del_feature])
        for Class in Classes:
            next_node=Node(Class,None,depth+1)
            next_cur_samples=cur_samples[cur_samples[del_feature]==Class]
            next_cur_samples.drop(columns=[del_feature],inplace=True)
            node.child.append(next_node)
            '''
            there are three ways to terminate recursion:
            1) when the recursion depth exceeds the set threshold
            2) No feature can be used
            3) all samples are pure
            '''
            if len(set(next_cur_samples['label']))!=1 and not next_cur_samples.empty and depth+1<=self.max_depth:
                self.recursion(next_cur_samples,next_node,depth+1,gain_type)

        return

'''main function'''
Train=pd.read_csv(r'C:/Users/Zhiyan/Desktop/MLassignments/New folder/car/train.csv',header=None)
Train.columns=['buying','maint','doors','persons','lug_boot','safety','label']
#Train=pd.DataFrame([['s','s','o','r','r','r','o','s','s','r','s','o','o','r'],['H','H','H','M','C','C','C','M','C','M','M','M','H','M'],['H','H','H','H','N','N','N','H','N','N','N','H','N','H'],['W','S','W','W','W','S','S','W','W','W','S','S','W','S'],[0,0,1,1,1,0,1,0,1,1,1,1,1,0]])
#Train=Train.T
#Train.columns=['O','T','H','W','label']
Test=[]
print('total training size: ',Train.shape)

Tree=DecisionTree(Train,Test)
'''
gain_type=['Entropy','Gini_Index','Majority_Error']
'''
Tree.Train_model(gain_type='Majority_Error')


# print tree
from collections import deque
stack=deque([Tree.root])
level=0
while stack:
    print(level)
    for _ in range(len(stack)):
        node=stack.popleft()
        print(node.split_attribute,node.name,node.depth)
        for child in node.child:
            stack.append(child)
    level+=1
