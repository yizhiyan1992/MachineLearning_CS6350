'''
tree depth---> self.max_depth
gain type---> DecisionTree.Train_model(gain_type)
gain_type=['Entropy','Gini_Index','Majority_Error']

'''
import os
import pandas as pd
import numpy as np
import math
from collections import Counter
print(os.getcwd())


class Node:
    def __init__(self,name,attribute,depth):
        self.name=name
        self.depth=depth
        self.split_attribute=attribute
        self.child=[]
        self.leaf=False
        self.prediction=None

class DecisionTree:
    def __init__(self,train_matrix,test):
        self.root=None
        # adjust this parameter to set the depth of the tree (depth is 0-indexed)
        self.max_depth=6
        self.train_matrix=train_matrix
        self.test=test
        self.train_sample_size=self.train_matrix.shape[0]
        self.train_feature_size=self.train_matrix.shape[1]

    def Train_model(self,gain_type):
        self.recursion(self.train_matrix,self.root,0,gain_type)
        return

    def Result_predict(self,samples):
        res=[]
        for index in range(samples.shape[0]):
            sample=samples.loc[index,:]
            res.append(self.Result_predict_each_sample(sample))
        return res
    def Result_predict_each_sample(self,sample):
        node=self.root
        while not node.leaf:
            not_found=True
            for child in node.child:
                if sample[node.split_attribute]==child.name:
                    not_found=False
                    node=child
                    break
            if not_found:
                return node.prediction
        return node.prediction

    def Prediction_accuracy(self,true_label,predict_label):
        total=len(true_label)
        count=0
        for index in range(total):
            if true_label[index]==predict_label[index]:
                count+=1
        return 1-count/total

    def InformationGain(self,col_feature,col_label,gain_type):
        val_root=self.Gain([],col_label,gain_type)
        val_child=self.Gain(col_feature,col_label,gain_type)
        info=val_root-val_child
        return info

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
            self.root=Node('root',del_feature,0)
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
            next_node.prediction = Counter(next_cur_samples['label']).most_common(1)[0][0]
            if len(set(next_cur_samples['label']))!=1 and not next_cur_samples.empty and depth+1<self.max_depth:
                self.recursion(next_cur_samples,next_node,depth+1,gain_type)
            else:
                # form into a leaf node, and output the prediction result by voting the most frequent element.
                next_node.leaf=True
        return

'''main function'''
# Download dataset, assgin column names
Train=pd.read_csv('car/train.csv',header=None)
Train.columns=['buying','maint','doors','persons','lug_boot','safety','label']
Test=pd.read_csv('car/test.csv',header=None)
Test.columns=['buying','maint','doors','persons','lug_boot','safety','label']
print('total training size: ',Train.shape)
print('label distribution from training set: ',Counter(Train['label']))

# Train DT model
Tree=DecisionTree(Train,Test)
'''gain_type=['Entropy','Gini_Index','Majority_Error']'''
Tree.Train_model(gain_type='Gini_Index')

#print ptediction result
res=Tree.Result_predict(Train)
res_test=Tree.Result_predict(Test)
print('Prediction error on training set: ',Tree.Prediction_accuracy(Train['label'].values,res))
print('Prediction error on testing set: ',Tree.Prediction_accuracy(Test['label'].values,res_test))

# print tree BFS
#from collections import deque
#stack=deque([Tree.root])
#level=0
#while stack:
#    print(level)
#    for _ in range(len(stack)):
#        node=stack.popleft()
#        print(node.split_attribute,node.name,node.depth,node.prediction,node.leaf)
#        for child in node.child:
#            stack.append(child)
#    level+=1
