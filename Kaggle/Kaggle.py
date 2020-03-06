import pandas as pd
import numpy as np
import os
os.chdir('/Users/Zhiyan1992/Desktop/')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

def GBDT(train_x,train_y,test_x):
    #clf=GradientBoostingClassifier(n_estimators=500)
    clf=SVC()
    clf.fit(train_x,train_y)
    res=clf.predict(test_x)
    res=pd.DataFrame(res)
    res.to_csv('results.csv',header=False)
    print(clf.score(train_x,train_y))
    print(res)



def main():
    Train=pd.read_csv('income-predict/train_final.csv')
    Test=pd.read_csv('income-predict/test_final.csv',index_col=0)
    print(Test.shape)
    #T
    #Test= pd.get_dummies(Test, columns=['workclass', 'education', 'education.num', 'marital.status', 'occupation','relationship', 'race', 'sex', 'native.country'])
    train_y=Train['income>50K']
    train_x=Train.drop(columns=['income>50K'])
    Concat=pd.concat([train_x,Test])
    Concat = pd.get_dummies(Concat, columns=['workclass', 'education', 'education.num', 'marital.status', 'occupation',
                                          'relationship', 'race', 'sex', 'native.country'])
    train_x=Concat.values[:train_y.shape[0],:]
    test_x=Concat.values[train_y.shape[0]:,:]
    print(train_y.shape,train_x.shape,test_x.shape)

    GBDT(train_x,train_y,test_x)

if __name__=='__main__':
    main()
