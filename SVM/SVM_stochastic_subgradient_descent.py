import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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
    def initialize_parameters(self,n):
        self.parameters['W']=np.zeros((1,n+1))
        return

    def calculate_loss_val(self,X,Y,C,m):
        regularization_term=0.5*(np.sum(np.square(self.parameters['W'])))
        loss_term=C*m*np.sum(np.maximum(0,1-Y*np.dot(X,self.parameters['W'].T)))  #hinge loss
        total_loss=regularization_term+loss_term
        return total_loss

    def calculate_total_loss_val(self,X,Y,C,m):
        regularization_term=0.5*(np.sum(np.square(self.parameters['W'])))
        loss_term=C*np.sum(np.maximum(0,1-Y*np.dot(X,self.parameters['W'].T)))  #hinge loss
        total_loss=regularization_term+loss_term
        return total_loss

    def update_params(self,learning_rate,C,m,X,Y):
        if Y*np.dot(X,self.parameters['W'].T)<=1:
            self.parameters['W']=self.parameters['W']-learning_rate*(self.parameters['W']-C*m*Y*X)
        else:
            self.parameters['W'] =(1-learning_rate)*self.parameters['W']
        return


    def train_model(self,X,Y,C=100/873,epoch=1,learning_rate=0.001):
        # m=sample size; n=number of features
        np.random.seed(42)
        m=X.shape[0]
        n=X.shape[1]
        Y=self.preprocessing_Y(Y)
        X=self.preprocessing_X(X)
        self.initialize_parameters(n)
        loss_on_all_samples=[]
        loss_val=[]
        rate=[]
        # train the model by stochastic gradien descent
        for i in range(epoch):
            temp_x=X
            temp_y=Y
            shuffle=np.random.permutation(m)
            temp_x=temp_x[shuffle,:]
            temp_y=temp_y[shuffle,:]
            for j in range(m):
                loss=self.calculate_loss_val(temp_x[j,:],temp_y[j,:],C,m)
                loss_val.append(loss)
                loss_all_sample=self.calculate_total_loss_val(temp_x,temp_y,C,m)
                loss_on_all_samples.append(loss_all_sample)
                #cur_learning_rate=learning_rate/(1+learning_rate/2*(i*m+j))
                cur_learning_rate=learning_rate/(1+(i*m+j))
                rate.append(cur_learning_rate)
                self.update_params(cur_learning_rate,C,m,temp_x[j,:],temp_y[j,:])
                #print(loss_all_sample)

        #print(self.parameters)
        #plt.plot(range(len(loss_val)),loss_val)
        #plt.plot(range(len(loss_on_all_samples)),loss_on_all_samples)
        plt.title('Loss value on total training samples')
        #plt.title('Loss value on a each single training sample')
        plt.xlabel('Number of steps')
        plt.ylabel('Loss value')
        plt.plot(range(len(rate)), rate)
        plt.show()
        return

    def predict(self,X):
        X=self.preprocessing_X(X)
        predict=np.dot(X,self.parameters['W'].T)
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
    print(train.shape,test.shape)
    print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)

    #####
    clf=SVM()
    clf.train_model(train_x,train_y)
    train_y_predict=clf.predict(train_x)
    score_train=clf.score(train_y,train_y_predict)
    test_y_predict=clf.predict(test_x)
    score_test=clf.score(test_y,test_y_predict)
    print(1-score_train,1-score_test)

if __name__=="__main__":
    main()
