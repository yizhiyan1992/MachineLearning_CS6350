import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from collections import Counter
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import xgboost

os.chdir('/Users/Zhiyan1992/Desktop/')

def Preprocessing(train_x,test_x):
    #
    for col in train_x.columns:
        if '?' in train_x[col].values:
            temp=Counter(train_x[col].values)
            most_freq=temp.most_common(1)[0][0]
            train_x[col][train_x[col]=='?']=most_freq
            test_x[col][test_x[col] == '?'] = most_freq
    return train_x,test_x

def Feature_Merge(train,test):
    #change the merital status
    train['marital.status'] = train['marital.status'].replace(['Widowed', 'Divorced', 'Separated', 'Never-married'], 'single')
    test['marital.status'] = test['marital.status'].replace(['Widowed', 'Divorced', 'Separated', 'Never-married'], 'single')
    train['marital.status'] = train['marital.status'].replace(['Married-spouse-absent', 'Married-civ-spouse', 'Married-AF-spouse'], 'married')
    test['marital.status'] = test['marital.status'].replace(['Married-spouse-absent', 'Married-civ-spouse', 'Married-AF-spouse'], 'married')

    #change the relationship
    train['relationship']=train['relationship'].replace(['Husband','Wife'],'Couple')
    test['relationship']=test['relationship'].replace(['Husband', 'Wife'], 'Couple')
    train['relationship'] = train['relationship'].replace(['Unmarried', 'Not-in-family','Own-child','Other-relative'], 'Others')
    test['relationship'] = test['relationship'].replace(['Unmarried', 'Not-in-family', 'Own-child', 'Other-relative'], 'Others')
    #check the result
    #print(set(train['marital.status']))
    #print(set(train['relationship']))
    return train,test

def plot_ROCcurve(label,predict_prob):
    fpr,tpr,threshold=metrics.roc_curve(label,predict_prob)
    roc_auc=metrics.auc(fpr,tpr)
    plt.title('ROC')
    plt.plot(fpr,tpr,label='AUC=%0.4f'%roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()
    return

def train_test_split(x,y):
    y=y[:,np.newaxis]
    print(y.shape)
    #use 80% data to train model and 20% to test its performance
    np.random.seed(42)
    shuffle=np.random.permutation(x.shape[0])
    point=int(x.shape[0]*0.8)
    x=x[shuffle,:]
    y=y[shuffle,:]
    print(x.shape,y.shape)
    train_x,test_x=x[:point,:],x[point:,:]
    train_y, test_y = y[:point,:],y[point:,:]
    return train_x,train_y,test_x,test_y

def RandomForest(train_x,train_y,test_x,test_y,final):
    clf=RandomForestClassifier(random_state=42)
    clf.fit(train_x,train_y)
    plot_ROCcurve(test_y,clf.predict_proba(test_x)[:,1])
    #res=clf.predict_proba(test_x)
    #res=pd.DataFrame(res)
    #res.to_csv('results_April.csv',header=False)
    #score=cross_val_score(clf,train_x,train_y,cv=5)
    #print('training set:',clf.score(train_x,train_y),'validation set (5-fold)',np.mean(score),sep='\n')

def XGB(train_x,train_y,test_x):
    clf=xgboost.XGBClassifier(seed=42,n_estimators=500)
    clf.fit(train_x,train_y)
    res = clf.predict(test_x)
    res=clf.predict_proba(test_x)
    res = pd.DataFrame(res)
    res.to_csv('results_April.csv', header=False)
    #score = cross_val_score(clf, train_x, train_y, cv=5)
    #print('training set:',clf.score(train_x,train_y),'validation set (5-fold)',np.mean(score),sep='\n')

def main():
    Train=pd.read_csv('income-predict/train_final.csv')
    Final=pd.read_csv('income-predict/test_final.csv',index_col=0)
    Train=Train.replace('?',np.NaN)
    Train=Train.dropna()

    train_y=Train['income>50K']
    train_x=Train.drop(columns=['income>50K'])
    #train_x,Test=Preprocessing(train_x,Test)

    train_x,Test=Feature_Merge(train_x,Final)
    #print(train_x['marital.status'])

    Concat=pd.concat([train_x,Final])
    Concat=Concat.drop(columns=['native.country'])

    Concat = pd.get_dummies(Concat, columns=['workclass', 'education', 'marital.status', 'occupation',
                                             'relationship','race', 'sex'])

    train_x=Concat.values[:train_y.shape[0],:]
    Final=Concat.values[train_y.shape[0]:,:]


    train_x,train_y,test_x,test_y=train_test_split(train_x,train_y)

    RandomForest(train_x,train_y,test_x,test_y,Final)
    #XGB(train_x,train_y,test_x)
if __name__=='__main__':
    main()
