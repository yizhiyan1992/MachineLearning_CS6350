import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from collections import Counter
os.chdir('/Users/Zhiyan1992/Desktop/')

class SVM():
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
        self.parameters['alpha']=np.random.rand(m,1)
        return

    def Gaussian_kernel(self,xi,xj,gamma):
        res=np.exp(-np.sum(np.square(xi-xj),axis=1)/gamma)
        res=res[:,np.newaxis]
        return res

    def rbf_kernel(self,x1, x2, sigma):
        X12norm = np.sum(x1 ** 2, 1, keepdims=True) - 2 * x1 @ x2.T + np.sum(x2 ** 2, 1, keepdims=True).T
        return np.exp(-X12norm / (2 * sigma ** 2))

    def optimize(self,X,Y,C,m,sigma):
        d = X.shape[1]  # dim of samples
        a = cp.Variable(shape=(m, 1), pos=True)  # lagrange multiplier

        #G = np.matmul(Y * X, (Y * X).T)  # Gram matrix
        G=self.rbf_kernel(Y*X,Y*X,sigma)
        objective = cp.Maximize(cp.sum(a) - (1 / 2) * cp.quad_form(a, G))
        constraints = [a <= C, cp.sum(cp.multiply(a, Y)) == 0]  # box constraint
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        res = a.value

        w=np.sum(res*Y*X,axis=0,keepdims=True)
        self.parameters['W']=w
        for i in range(res.shape[0]):
            if res[i,0]>0 and res[i,0]<C:
                #b=Y[i,0]-np.dot(X[i,:],w.T)
                b=Y[i,0]-np.sum(res*Y*np.dot(X,X[i,:].T))
                self.parameters['b']=b
                break
        return

    def train_model(self, X, Y, C=100/ 873, epoch=2, learning_rate=0.0001,sigma=0.1):
        # m=sample size; n=number of features
        np.random.seed(42)
        m = X.shape[0]
        n = X.shape[1]
        Y = self.preprocessing_Y(Y)
        #X = self.preprocessing_X(X)
        self.initialize_parameters(m)
        self.optimize(X,Y,C,m,sigma)
        self.X=X
        self.Y=Y
        self.sigma=sigma

    def predict(self,X):
        X_train = self.X
        Y_train = self.Y
        sigma=self.sigma
        predict = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            product = self.Gaussian_kernel(X_train, X[i, :], sigma)
            predict[i, 0] = (np.sum(self.parameters['alpha'] * Y_train * product) )
        predict = np.where(predict >= 0, 1, -1)
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
    plt.scatter(train_x[:,0],train_x[:,1],c=list(train_y.reshape(-1)))

    clf=SVM()
    clf.train_model(train_x,train_y,C=500/ 873, epoch=2, learning_rate=0.0001,sigma=0.1)
    train_y_predict=clf.predict(train_x)
    score_train=clf.score(train_y,train_y_predict)
    print(score_train)
    test_y_predict=clf.predict(test_x)
    score_test=clf.score(test_y,test_y_predict)
    print(1-score_train,1-score_test)

if __name__=="__main__":
    main()
