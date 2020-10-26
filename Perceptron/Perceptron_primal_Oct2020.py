import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self):
        self.weight=None
        self.bias=None

    def train(self,X,y,epoch=50,learning_rate=0.0001):
        self.N,self.D=X.shape
        self.weight=np.zeros((self.D,1))
        self.bias=0
        print(self.score(X, y))
        for _ in range(epoch):
            for i in range(self.N):
                if y[i]*(np.dot(X[i,:],self.weight)+self.bias)<=0:
                #wrong class, update weight and bias
                    self.weight+=learning_rate*y[i]*X[np.newaxis,i,:].T
                    self.bias+=learning_rate*y[i]
            print(self.score(X,y))

    def predict(self,X):
        pre=np.dot(X,self.weight)+self.bias
        pre=np.where(pre>0,1,-1)
        return pre[:,0]

    def score(self,X,y):
        y_pre=self.predict(X)
        score=np.mean(y_pre==y)
        return score

def main():
    np.random.seed(42)
    X_p1=np.random.normal(loc=3,scale=1,size=50)
    X_p2=np.random.normal(loc=3,scale=1,size=50)
    y_p=np.ones(50)
    X_n1=np.random.normal(loc=-2,scale=1,size=50)
    X_n2=np.random.normal(loc=-2,scale=1,size=50)
    y_n=np.ones(50)*-1
    X_1=np.concatenate((X_p1,X_n1))
    X_2=np.concatenate((X_p2,X_n2))
    X=np.stack((X_1,X_2),axis=1)
    print(X.shape)
    y=np.concatenate((y_p,y_n))
    plt.scatter(X_1,X_2,c=y)
    plt.show()

    #initialize model, train, predict, and test
    model=Perceptron()
    model.train(X,y)
    y_pre=model.predict(X)
    score=model.score(X,y)
    print(score)
