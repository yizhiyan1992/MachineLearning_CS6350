import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from collections import Counter
import os
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
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

def plot_ROCcurve(label,predict_prob,name):
    fpr,tpr,threshold=metrics.roc_curve(label,predict_prob)
    print('this is threshold',threshold)
    roc_auc=metrics.auc(fpr,tpr)
    plt.title('ROC')
    plt.plot(fpr,tpr,label='Model: '+name+' AUC=%0.4f'%roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
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

class choose_model():

    def Perceptron(self,train_x,train_y,test_x,test_y):
        clf = Perceptron()
        clf.fit(train_x, train_y)
        probe = clf.decision_function(test_x)
        plot_ROCcurve(test_y, probe, 'Perceptron')
        return

    def ANN(self,train_x,train_y,test_x,test_y):
        clf=MLPClassifier()
        clf.fit(train_x,train_y)
        probe=clf.predict_proba(test_x)[:,1]
        plot_ROCcurve(test_y,probe,'ANN')
        return

    def RandomForest(self,train_x,train_y,test_x,test_y):
        clf=RandomForestClassifier(random_state=42)
        clf.fit(train_x,train_y)
        probe = clf.predict_proba(test_x)[:, 1]
        plot_ROCcurve(test_y, probe, 'Random Forest')
        return

    def XGB(self,train_x, train_y, test_x,test_y):
        clf = xgboost.XGBClassifier(seed=42,n_estimators=50)
        clf.fit(train_x, train_y)
        probe=clf.predict_proba(test_x)[:,1]
        plot_ROCcurve(test_y,probe,'XGBoost')
        return

    def NaiveBayes(self,train_x, train_y, test_x,test_y):
        print('start')
        clf = GaussianNB()
        clf.fit(train_x, train_y)
        print('finish')
        probe = clf.predict_proba(test_x)[:, 1]
        plot_ROCcurve(test_y, probe, 'Gaussian Naive Bayes')

        return


def calibration(x,y):
    model=xgboost.XGBClassifier(seed=42)
    hyperparams={'n_estimators':[100,500,750,1000],'learning_rate':[0.01,0.1,0.5,1]}
    clf=GridSearchCV(model,hyperparams,cv=4,scoring='roc_auc')
    clf.fit(x,y)
    print(clf.cv_results_['mean_test_score'])
    print(clf.best_params_)
    return

def final_prediction(x_train,y_train,final):
    hyperparams = {'seed':42,'n_estimators':750, 'learning_rate': 0.1}
    clf = xgboost.XGBClassifier(**hyperparams)
    clf.fit(x_train,y_train)
    res=clf.predict_proba(final)[:,1]
    res=pd.DataFrame(res)
    res.to_csv('Final_prediction.csv')
    return



def main():
    Train=pd.read_csv('income-predict/train_final.csv')
    Final=pd.read_csv('income-predict/test_final.csv',index_col=0)
    Train=Train.replace('?',np.NaN)
    Train=Train.dropna()

    Train_y=Train['income>50K']
    Train_x=Train.drop(columns=['income>50K'])
    #train_x,Test=Preprocessing(train_x,Test)

    Train_x,Test=Feature_Merge(Train_x,Final)
    Concat=pd.concat([Train_x,Final])
    Concat=Concat.drop(columns=['native.country'])
    Concat = pd.get_dummies(Concat, columns=['workclass', 'education', 'marital.status', 'occupation',
                                             'relationship','race', 'sex'])

    Train_x=Concat.values[:Train_y.shape[0],:]
    Final=Concat.values[Train_y.shape[0]:,:]
    train_x,train_y,test_x,test_y=train_test_split(Train_x,Train_y)

    #select model
    '''
    model_test=choose_model()
    model_test.Perceptron(train_x, train_y, test_x, test_y)
    model_test.ANN(train_x,train_y,test_x,test_y)
    model_test.RandomForest(train_x, train_y, test_x, test_y)
    model_test.XGB(train_x, train_y, test_x, test_y)
    model_test.NaiveBayes(train_x, train_y, test_x, test_y)
    plt.show()
    #t the best model is xgboost


    #parameter tuning
    calibration(Train_x, Train_y)
    '''

    #Final prediction
    final_prediction(Train_x,Train_y,Final)


if __name__=='__main__':
    main()
