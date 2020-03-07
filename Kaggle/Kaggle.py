import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import os
os.chdir('/Users/Zhiyan1992/Desktop/')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def Preprocessing(train_x,test_x):
    for col in train_x.columns:
        if '?' in train_x[col].values:
            temp=Counter(train_x[col].values)
            most_freq=temp.most_common(1)[0][0]
            train_x[col][train_x[col]=='?']=most_freq
            test_x[col][test_x[col] == '?'] = most_freq
    return train_x,test_x

def Correlation_analysis(array):
    corr_coef=array.corr()
    print(corr_coef)
    fig, ax = plt.subplots()
    im= ax.imshow(corr_coef,cmap="YlGn")
    ax.set_xticks(np.arange(corr_coef.shape[0]))
    ax.set_yticks(np.arange(corr_coef.shape[0]))

    # ... and label them with the respective list entries
    ax.set_xticklabels(corr_coef.columns,rotation=45)
    ax.set_yticklabels(corr_coef.columns)
    for i in range(corr_coef.shape[0]):
        for j in range(corr_coef.shape[0]):
            text = ax.text(j, i, round(corr_coef.values[i][j],2),va='center',ha='center',color="k")
    plt.tight_layout()
    plt.savefig('correlation.png')
    plt.show()

def GBDT(train_x,train_y,test_x):
    clf=RandomForestClassifier(random_state=42)
    #clf=GradientBoostingClassifier(random_state=42)
    clf.fit(train_x,train_y)
    res=clf.predict(test_x)
    res=pd.DataFrame(res)
    res.to_csv('results.csv',header=False)
    print(clf.score(train_x,train_y))



def main():
    Train=pd.read_csv('income-predict/train_final.csv')
    Test=pd.read_csv('income-predict/test_final.csv',index_col=0)
    print(Test.shape)
    train_y=Train['income>50K']
    train_x=Train.drop(columns=['income>50K'])
    #train_x,Test=Preprocessing(train_x,Test)
    Concat=pd.concat([train_x,Test])
    Sum=0
    #Correlation_analysis(Train.drop(columns=['workclass', 'education', 'marital.status', 'occupation',
    #                                      'relationship', 'race', 'sex', 'native.country','income>50K']))
    Concat = pd.get_dummies(Concat, columns=['workclass', 'education', 'marital.status', 'occupation',
                                          'relationship', 'race', 'sex', 'native.country'])
    train_x=Concat.values[:train_y.shape[0],:]
    test_x=Concat.values[train_y.shape[0]:,:]
    #print(train_y.shape,train_x.shape,test_x.shape)
    GBDT(train_x,train_y,test_x)

if __name__=='__main__':
    main()
