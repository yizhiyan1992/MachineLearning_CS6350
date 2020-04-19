import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
os.chdir('/Users/Zhiyan1992/Desktop/')

def load_data():
    train=pd.read_csv('bank-note/train.csv',header=None)
    test=pd.read_csv('bank-note/test.csv',header=None)
    print('data size train {}, test {}:'.format(train.shape,test.shape))
    train_x,train_y=train.values[:,:-1],train.values[:,-1]
    test_x, test_y = test.values[:, :-1], test.values[:, -1]
    return train_x,train_y,test_x,test_y

class logistic_regression():
    def __init__(self):
        self.w=None
        self.b=None

    def process_y(self,y):
        # Y value--->{-1,1}
        y[y==0]=-1
        y=y[:,np.newaxis]
        return y

    def loss_value(self,train_x,train_y,var,type):
        # only one sample each time
        if type=='MAP':
            loss=np.log(1+np.exp(-train_y*(np.dot(train_x,self.w.T)+self.b)))+1/(2*var)*np.dot(self.w,self.w.T)
        elif type == 'ML':
            loss = np.log(1 + np.exp(-train_y * (np.dot(train_x, self.w.T) + self.b)))
        return loss[0][0]

    def total_loss(self,train_x,train_y,var,type):
        if type == 'MAP':
            loss = np.sum(np.log(1 + np.exp(-train_y * (np.dot(train_x, self.w.T) + self.b))),axis=0,keepdims=True) + 1 / (2 * var) * np.dot(self.w,self.w.T)
        elif type=='ML':
            loss = np.sum(np.log(1 + np.exp(-train_y * (np.dot(train_x, self.w.T) + self.b))), axis=0,keepdims=True)
        return loss[0][0]

    def gradient_descent(self,train_x,train_y,var,type):
        if type == 'MAP':
            dw=np.exp(-train_y * (np.dot(train_x, self.w.T)+self.b)) / (1 + np.exp(-train_y * (np.dot(train_x, self.w.T)+self.b))) * (-train_y * train_x) + self.w/var
            db=np.exp(-train_y * (np.dot(train_x, self.w.T)+self.b)) / (1 + np.exp(-train_y * (np.dot(train_x, self.w.T)+self.b))) * (-train_y)
        elif type == 'ML':
            dw=np.exp(-train_y * (np.dot(train_x, self.w.T)+self.b)) / (1 + np.exp(-train_y * (np.dot(train_x, self.w.T)+self.b))) * (-train_y * train_x)
            db=np.exp(-train_y * (np.dot(train_x, self.w.T)+self.b)) / (1 + np.exp(-train_y * (np.dot(train_x, self.w.T)+self.b))) * (-train_y)
        return dw,db

    def train(self,train_x,train_y,epoch=100,learning_rate=0.001,var=1,type='MAP'):
        #stochastic gradient descent
        #n=no of feature, m=no of training samples
        train_y=self.process_y(train_y)
        n=train_x.shape[1]
        m=train_x.shape[0]
        self.w=np.array([[0 for i in range(n)]])
        self.b=0

        loss_val=[]
        for epo in range(epoch):
            train_x_shuffle=train_x.copy()
            train_y_shuffle=train_y.copy()
            shuffle=np.random.permutation(np.arange(m))
            train_x_shuffle=train_x_shuffle[shuffle,:]
            train_y_shuffle=train_y_shuffle[shuffle,:]


            for i in range(m):
                #loss=self.loss_value(train_x_shuffle[i,:],train_y_shuffle[i],var,type)
                loss=self.total_loss(train_x_shuffle,train_y_shuffle,var,type)
                loss_val.append(loss)
                dw,db=self.gradient_descent(train_x_shuffle[i,:],train_y_shuffle[i],var,type)
                new_learning_rate=learning_rate/(1+learning_rate*(m*epo+i))
                self.w=self.w-new_learning_rate*dw
                self.b=self.b-new_learning_rate*db

        plt.plot(range(len(loss_val)),loss_val)
        plt.show()

    def predict_score(self,x,y):
        y=self.process_y(y)
        z=np.sum(self.w*x+self.b,axis=1)
        logits=1/(1+np.exp(-z))
        logits=np.where(logits<=0.5,-1,1)

        y=y.reshape(logits.shape)
        score=np.mean(logits==y)
        return score

def main():
    train_x, train_y, test_x, test_y=load_data()
    var_lis=[0.01,0.1,0.5,1,3,5,10,100]
    for i in var_lis:
        clf=logistic_regression()
        clf.train(train_x,train_y,var=i,type='ML')
        train_score=clf.predict_score(train_x,train_y)
        test_score=clf.predict_score(test_x,test_y)
        print('train score:{} test score: {}'.format(train_score,test_score))


if __name__=='__main__':
    main()
