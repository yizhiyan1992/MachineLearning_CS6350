'''
This Neural Network supports binary classification problems.
The prediction should be changed into {0,1} class
A test dataset (bank-note from UCI) is uploaded to test this model.
Three params can be adjusted:
1) iteration: No of iterations
2) learning_rate: steps
3) layer_list: a list based input, where each number in the list indicates the number of neurons of that layer (e.g. [5] is a one hidden layer NN with 5 neurons)
'''

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
os.chdir('/Users/Zhiyan1992/Desktop/')

class L_layer_Neural_Network():
    def __init__(self):
        self.layer_list=[5]
        self.parameters={}
        self.caches={}

    def train_model(self,X,Y,iteration=100,learning_rate=0.1,layer_list=None):
        #stochastic gradient descent, m =sample number, n=feature number
        m=X.shape[1]
        n=X.shape[0]
        print(X.shape,Y.shape)
        if layer_list:
            self.layer_list=layer_list
        self.X=X
        self.Y=Y
        self.initialize_parameter()
        loss=[]
        for i in range(iteration):
            #shuffle
            temp_x=X.copy()
            temp_y=Y.copy()
            shuffle=np.random.permutation(range(m))
            temp_x=temp_x[:,shuffle]
            temp_y=temp_y[:,shuffle]
            for j in range(m):
                y_predict=self.forward_propagation(temp_x[:,j])
                loss_val=self.compute_loss_val(temp_y[:,j],y_predict)
                self.compute_derivative(self.caches,y_predict,self.parameters,temp_y[:,j])
                self.update_parameters(self.parameters,self.deri,learning_rate)
                loss.append(loss_val)
        plt.plot(range(len(loss)),loss)
        plt.show()
        return

    def initialize_parameter(self):
        '''
        :param no_feature: the number of features for training samples
        :param layer_list: a list to define how many neurons for each hidden layer
        :param y_shape: output dim
        :return: initial parameters of W and b (a dict)
        '''
        no_feature=self.X.shape[0]
        y_shape=self.Y.shape[0]
        np.random.seed(42)
        for i in range(len(self.layer_list)):
            if i==0:
                W=np.random.randn(self.layer_list[i],no_feature)*0.01
                b=np.zeros((self.layer_list[i],1))
            else:
                W = np.random.randn(self.layer_list[i], self.layer_list[i-1]) * 0.01
                b = np.zeros((self.layer_list[i], 1))
            self.parameters['W'+str(i+1)]=W
            self.parameters['b' + str(i + 1)] = b
        #last layer
        W = np.random.randn(y_shape, self.layer_list[-1]) * 0.01
        b = np.zeros((y_shape, 1))
        self.parameters['W' + str(len(self.layer_list) + 1)] = W
        self.parameters['b' + str(len(self.layer_list) + 1)] = b
        return

    def forward_propagation(self,X):
        '''
        :param X: input m training examples
        :param parameters: W and b
        :return: predicted result y_hat, and caches to store A for each layer (A[l],caches)
        formula:
        Z[l]=W[l]A[l-1]+b[l]
        A[l]=activation(Z[l])
        here, we use sigmoid func as activation function
        '''
        self.caches={}
        A=X
        A=A[:,np.newaxis]
        self.caches['Z' + str(0)] = X[:,np.newaxis]
        self.caches['A' + str(0)] = X[:,np.newaxis]
        for i in range(len(self.parameters)//2):
            Z=np.dot(self.parameters['W'+str(i+1)],A)
            A=1/(1+np.exp(-Z))
            self.caches['Z'+str(i+1)]=Z
            self.caches['A'+str(i+1)]=A
        return A

    def compute_loss_val(self,Y,Y_prediction):
        '''
        :param Y:
        :param Y_prediction:
        :return: J(Y,Y_hat)
        use log-loss function to calculate total cost (note that y={0,1}, if not, it needs to be transformed into 0-1 vals)
        '''

        loss=-(np.dot(Y,np.log(Y_prediction).T)+np.dot((1-Y),np.log(1-Y_prediction).T))
        self.loss=loss
        return loss

    def compute_derivative(self,caches,prediction,parameters,Y):

        dZ=(prediction-Y)
        self.deri={}
        for i in reversed(range(len(parameters)//2)):
            dW=np.dot(dZ,caches['A'+str(i)].T)
            db=np.sum(dZ,axis=1,keepdims=True)
            self.deri['dW' + str(i + 1)] = dW
            self.deri['db' + str(i + 1)] = db
            dA_prev=np.dot(parameters['W'+str(i+1)].T,dZ)
            dZ_prev=dA_prev*(np.exp(-caches['Z'+str(i)])/((1+np.exp(-caches['Z'+str(i)]))*(1+np.exp(-caches['Z'+str(i)]))))
            dZ=dZ_prev
        return

    def update_parameters(self,parameters,deri,learning_rate):
        for i in range(len(parameters)//2):
            parameters['W'+str(i+1)]=parameters['W'+str(i+1)]-learning_rate*deri['dW'+str(i+1)]
            parameters['b' + str(i + 1)] = parameters['b' + str(i + 1)] - learning_rate *deri['db' + str(i + 1)]
        return

    def predict(self,X):
        A = X
        for i in range(len(self.parameters) // 2):
            Z = np.dot(self.parameters['W' + str(i + 1)], A)
            A = 1 / (1 + np.exp(-Z))
        res=np.where(A>=0.5,1,0)
        return res

    def score(self,Y,Y_predict):
        m=Y.shape[1]
        return np.sum(Y==Y_predict)/m

def main():
    '''
    preprocessing data:
    data shape should be [N,M], where N is the number of features, and M is the sample size
    '''
    train=pd.read_csv('bank-note/train.csv',header=None)
    test=pd.read_csv('bank-note/test.csv',header=None)
    X,Y=train.values[:,:-1],train.values[:,-1]
    X_test,Y_test=test.values[:,:-1],test.values[:,-1]
    Y=Y[:,np.newaxis]
    Y_test = Y_test[:, np.newaxis]
    X,Y=X.T,Y.T
    X_test,Y_test=X_test.T,Y_test.T
    # train model and predict samples
    ANN=L_layer_Neural_Network()
    ANN.train_model(X,Y,iteration=10,learning_rate=0.1,layer_list=[10,10])
    res=ANN.predict(X)
    print('prediction accuracy on training set: ',ANN.score(Y,res))
    res_test=ANN.predict(X_test)
    print('prediction accuracy on test set: ',ANN.score(Y_test,res_test))

if __name__=="__main__":
    main()
