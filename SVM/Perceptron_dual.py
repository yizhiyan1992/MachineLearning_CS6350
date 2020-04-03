import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('/Users/Zhiyan1992/Desktop/')

class Perceptron():
    def __init__(self):
        self.parameters={}

    def preprocessing_Y(self,Y):
        # map Y into {-1,1}
        Y_origin=list(set(list(Y.reshape(-1))))
        Y=np.where(Y==Y_origin[0],-1,1)
        return Y

    def preprocessing_X(self,X):
        #add one dimension , param b
        b=np.ones((X.shape[0],1))
        X=np.hstack((X,b))
        return X

    def initialize_parameters(self,m):
        self.parameters['alpha']=np.zeros((m,1))
        self.parameters['b']=0
        return

    def Gaussian_kernel(self,xi,xj,gamma):
        res=np.exp(-np.sum(np.square(xi-xj),axis=1)/gamma)
        res=res[:,np.newaxis]
        return res

    def train_model(self, X, Y, epoch=50, learning_rate=1,gamma=1):
        # m=sample size; n=number of features
        np.random.seed(42)
        m = X.shape[0]
        n = X.shape[1]
        Y = self.preprocessing_Y(Y)
        self.initialize_parameters(m)

        for i in range(epoch):
            for j in range(m):
                product=self.Gaussian_kernel(X,X[j,:],gamma)
                if Y[j]*(np.sum(learning_rate*self.parameters['alpha']*Y*product)+self.parameters['b'])<=0:
                    self.parameters['alpha'][j,0]+=1
                    self.parameters['b']+=learning_rate*Y[j]
        w=np.sum(self.parameters['alpha']*Y*X,axis=0)
        self.parameters['W']=np.array([[w[0],w[1]]])
        self.X=X
        self.Y=Y
        self.learning_rate=learning_rate

    def predict(self,X):
        X_train=self.X
        Y_train=self.Y
        learning_rate=self.learning_rate
        predict=np.zeros((X.shape[0],1))
        for i in range(X.shape[0]) :
            product=self.Gaussian_kernel(X_train,X[i,:],1)
            predict[i,0]=(np.sum(learning_rate*self.parameters['alpha']*Y_train*product)+self.parameters['b'])
        predict=np.where(predict>=0,1,-1)
        return predict

    def score(self,Y,Y_predict):

        Y=self.preprocessing_Y(Y)

        score=np.sum(Y==Y_predict)/Y.shape[0]
        return score

def main():
    train=pd.read_csv('bank-note/train.csv',header=None)
    test=pd.read_csv('bank-note/test.csv',header=None)
    train_x,train_y=train.values[:,:-1],train.values[:,-1]
    train_y=train_y[:,np.newaxis]
    test_x, test_y = test.values[:, :-1], test.values[:, -1]
    test_y = test_y[:, np.newaxis]
    clf=Perceptron()
    clf.train_model(train_x,train_y,gamma=90)
    train_y_predict=clf.predict(train_x)
    score_train=clf.score(train_y,train_y_predict)
    print(score_train)
    test_y_predict=clf.predict(test_x)
    score_test=clf.score(test_y,test_y_predict)
    print(1-score_train,1-score_test)

if __name__=="__main__":
    main()
