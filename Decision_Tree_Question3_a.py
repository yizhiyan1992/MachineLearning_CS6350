'''
Question3-(a)
Adjust tree depth ---> self.max_depth
Adjust gain type---> Tree.Train_model(gain_type=['Entropy','Gini_Index','Majority_Error'])

There are 4 outputs:
1) Prediction results on Train set
2) Prediction accuracy on Train set
3) Prediction results on Test set
4) Prediction accuracy on Test set
'''
import pandas as pd
import math
from collections import Counter

class Node:
    def __init__(self,name,attribute,depth):
        self.name=name
        self.depth=depth
        self.split_attribute=attribute
        self.child=[]
        self.leaf=False
        self.prediction=None

class DecisionTree:
    def __init__(self,train_matrix):
        self.root=None
        # adjust this parameter to set the depth of the tree (depth is 0-indexed)
        self.max_depth=16
        self.train_matrix=train_matrix
        self.train_sample_size=self.train_matrix.shape[0]
        self.train_feature_size=self.train_matrix.shape[1]

    def Train_model(self,gain_type,numeric):
        self.recursion(self.train_matrix,self.root,0,gain_type,numeric)
        return

    def Result_predict(self,samples):
        res=[]
        for index in range(samples.shape[0]):
            sample=samples.loc[index,:]
            res.append(self.Result_predict_each_sample(sample))
        return res
    def Result_predict_each_sample(self,sample):
        #print(sample)
        node=self.root
        while not node.leaf:
            not_found = True
            numeric=Numeric[node.split_attribute]
            for child in node.child:
                if numeric:
                    not_found=False
                    attribute,low_high,median=child.name.split('_')
                    median=float(median)
                    if low_high=='lower' and sample[node.split_attribute]<median:
                        node=child
                        break
                    elif low_high=='higher' and sample[node.split_attribute]>=median:
                        node=child
                        break
                else:
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
        return count/total

    def InformationGain(self,col_feature,col_label,gain_type,numeric):
        val_root=self.Gain([],col_label,gain_type,numeric)
        val_child=self.Gain(col_feature,col_label,gain_type,numeric)
        info=val_root-val_child
        return info

    def Gain(self,col_feature,col_label,gain,numeric):
        Dict = {}
        if len(col_feature)==0: # calculate the gain for father node
            Dict['counter']=Counter(col_label)
        else:
            # numeric variables
            if numeric:
                # find the median first
                temp=col_feature.copy()
                temp.sort()
                if len(temp)%2==1:
                    median=temp[int(len(temp)/2)]
                else:
                    median=0.5*(temp[int((len(temp)-1)/2)]+temp[int((len(temp)+1)/2)])
                Dict.setdefault('lower',[])
                Dict.setdefault('higher',[])
                for index in range(len(col_feature)):
                    if col_feature[index]>median:
                        Dict['higher'].append(col_label[index])
                    else:
                        Dict['lower'].append(col_label[index])
                for key,val in Dict.items():
                    Dict[key]=Counter(val)
            # categorical variables
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

    def recursion(self,cur_samples,node,depth,gain_type,numeric):
        name=[]
        information_gain=[]
        # select the feature with max information gain value
        for col in cur_samples.columns:
            if col!='label':
                name.append(col)
                val=self.InformationGain(cur_samples[col].values,cur_samples['label'].values,gain_type,numeric[col])
                information_gain.append(round(val, 3))
        del_feature=name[information_gain.index(max(information_gain))]
        if not node:
            self.root=Node('root',del_feature,0)
            node=self.root
        else:
            node.split_attribute=del_feature
        node.depth=depth
        #numeric variables
        if numeric[del_feature]:
            temp = list(cur_samples[del_feature].copy())
            temp.sort()
            if len(temp) % 2 == 1:
                median = temp[int(len(temp) / 2)]
            else:
                median = 0.5 * (temp[int((len(temp) - 1) / 2)] + temp[int((len(temp) + 1) / 2)])

            Classes=['lower','higher']
            for Class in Classes:
                next_node=Node(node.split_attribute+'_'+Class+'_'+str(median),None,depth+1)
                if Class=='higher':
                    next_cur_samples = cur_samples[cur_samples[del_feature]>=median]
                elif Class=='lower':
                    next_cur_samples = cur_samples[cur_samples[del_feature]<=median]
                next_cur_samples.drop(columns=[del_feature], inplace=True)
                node.child.append(next_node)
                next_node.prediction = Counter(next_cur_samples['label']).most_common(1)[0][0]
                if len(set(next_cur_samples['label']))!=1 and not next_cur_samples.empty and depth + 1 < self.max_depth:
                    self.recursion(next_cur_samples, next_node, depth + 1, gain_type, numeric)
                else:
                    # form into a leaf node, and output the prediction result by voting the most frequent element.
                    next_node.leaf = True
        #categorical variables
        else:
            Classes = set(cur_samples[del_feature])
            for Class in Classes:
                next_node=Node(Class,None,depth+1)
                next_cur_samples=cur_samples[cur_samples[del_feature]==Class]
                next_cur_samples.drop(columns=[del_feature],inplace=True)
                node.child.append(next_node)
                next_node.prediction = Counter(next_cur_samples['label']).most_common(1)[0][0]
                if len(set(next_cur_samples['label']))!=1 and not next_cur_samples.empty and depth+1<self.max_depth:
                    self.recursion(next_cur_samples,next_node,depth+1,gain_type,numeric)
                else:
                    # form into a leaf node, and output the prediction result by voting the most frequent element.
                    next_node.leaf=True
        return

'''main function'''
#step 1: loading data and assign feature names to each column
Train=pd.read_csv(r'C:/Users/Zhiyan/Desktop/MLassignments/New folder/bank/train.csv',header=None)
Test=pd.read_csv(r'C:/Users/Zhiyan/Desktop/MLassignments/New folder/bank/test.csv',header=None)
Train.columns=Test.columns=['age','job','marital','education','default','balance','housing'\
    ,'loan','contact','day','month','duration','campaign','pdays','previous','poutcome','label']
Numeric={'age':True,'job':False,'marital':False,'education':False,'default':False,'balance':True,'housing':False, \
         'loan':False,'contact':False,'day':True,'month':False,'duration':True,'campaign':True,'pdays':True,'previous':True,'poutcome':False}
print('total training size: ',Train.shape)
print('label distribution from training set: ',Counter(Train['label']))

#Step2: Train the Decision Tree model
Tree=DecisionTree(Train)
'''gain_type=['Entropy','Gini_Index','Majority_Error']'''
Tree.Train_model(gain_type='Entropy',numeric=Numeric)

#Step3: Output the prediction result
res=Tree.Result_predict(Train)
res_test=Tree.Result_predict(Test)
print('Prediction results on training set: ',res,sep='\n')
print('Prediction accuracy on training set: ',Tree.Prediction_accuracy(Train['label'].values,res),sep='\n')
print('Prediction results on testing set: ',res_test,sep='\n')
print('Prediction results on testing set: ',Tree.Prediction_accuracy(Test['label'].values,res_test),sep='\n')

'''
# print tree nodes by BFS to check out
from collections import deque
stack=deque([Tree.root])
level=0
while stack:
    print(level)
    for _ in range(len(stack)):
        node=stack.popleft()
        print(node.split_attribute,node.name,node.depth,node.prediction,node.leaf)
        for child in node.child:
            stack.append(child)
    level+=1
'''
