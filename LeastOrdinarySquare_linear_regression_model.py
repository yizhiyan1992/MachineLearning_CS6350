'''
Linear regression model
learning rate---> LinearRegression.learning_rate
steps---> LinearRegression.iteration_times
gradient descent method---> LinearRegression.Gradient_descent_method /batch and stochastic

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self,learning_rate,iteration_times,GD_method):
        self.learning_rate=learning_rate
        self.iteration=iteration_times
        self.Gradient_descent_method=GD_method

    def LinearFunction(self,train_x,train_y):
        #add one dimension to vector w, and set w_0=b
        self.w=np.array([0 for _ in range(train_x.shape[1]+1)])
        #concatenate a constant x_0=1
        constant=np.array([[1] for _ in range(train_x.shape[0])])
        print(constant.shape,train_x.shape)
        train_x=np.concatenate((constant,train_x),axis=1)
        print(train_x.shape)
        self.LeastOrdinarySquare(train_x,train_y)

    def BatchGradientDescent(self,train_x,train_y):
        dw=[0 for _ in range(len(self.w))]
        for sample in range(train_x.shape[0]):
            temp=-(train_y[sample]-np.dot(self.w,train_x[sample]))*train_x[sample]
            dw+=temp
        self.w=self.w-self.learning_rate*dw

    def StochasticGradientDescent(self,train_x,train_y,rand):
        dw=-(train_y[rand]-np.dot(self.w,train_x[rand,:]))*train_x[rand,:]
        self.w = self.w - self.learning_rate * dw

    def LeastOrdinarySquare(self,train_x,train_y):
        loss=[]
        # randomly select a sample
        np.random.seed(42)
        rand = np.random.randint(0, train_x.shape[0],self.iteration)
        print(train_x.shape,train_y.shape,self.w.shape)
        for index in range(self.iteration):
            res=np.sum(0.5*(train_y-np.dot(train_x,self.w))**2)
            if self.Gradient_descent_method=='stochastic':
                self.StochasticGradientDescent(train_x,train_y,rand[index])
            else:
                self.BatchGradientDescent(train_x,train_y)
            loss.append(res)
        #print(loss)

        #plot loss function:
        plt.plot(range(len(loss)),loss)
        plt.xlabel('Iteration times')
        plt.ylabel('Loss function')
        #plt.title('Batch Gradient Descent')
        plt.title('Stochastic Gradient Descent')
        #plt.savefig(r'C:/Users/Zhiyan/Desktop/Batch.png')
        plt.savefig(r'C:/Users/Zhiyan/Desktop/stochastic.png')
        plt.show()

    def Loss_on_test(self,test_x,test_y):
        # concatenate a constant x_0=1
        constant = np.array([[1] for _ in range(test_x.shape[0])])
        test_x = np.concatenate((constant, test_x), axis=1)
        print(self.w)
        return np.sum(0.5*(test_y-np.dot(test_x,self.w))**2)

'''------main function------'''
train=pd.read_csv(r'concrete/train.csv',header=None)
test=pd.read_csv(r'concrete/test.csv',header=None)
print('the sample size for training set: ',train.shape)
train_x=train.values[:,:-1]
train_y=train.values[:,-1]
test_x=test.values[:,:-1]
test_y=test.values[:,-1]
'''
GD_method=['batch','stochastic']
'''
linear_model=LinearRegression(learning_rate=0.001,iteration_times=50000,GD_method='stochastic')
linear_model.LinearFunction(train_x,train_y)
print(linear_model.Loss_on_test(test_x,test_y))
