import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

class logistic_regression():
    def __init__(self):
        self.params=None
        self.bias=None

        return

    def train(self,X,y,iteration=100,learning_rate=0.01):
        # X.shape= (N*D), where N is the data size and D is the feature size
        # y.shape= (N,)
        N=X.shape[0]
        D=X.shape[1]
        self.params=np.array([[0] for _ in range(D)])
        self.bias=0

        Loss=[]
        for i in range(iteration):
            t = np.dot(X, self.params) + self.bias  # t.shape should be [n,]
            y_pre = 1 / (1 + np.exp(-t))
            y_pre = y_pre[:, 0]
            loss=-np.mean(y*np.log(y_pre)+(1-y)*np.log(1-y_pre))
            Loss.append(loss)
            diff=y_pre-y
            diff=diff[:,np.newaxis]
            d_params=np.mean(diff*X,axis=0,keepdims=True)
            d_bias = np.mean(diff, axis=0,keepdims=True)
            self.params=self.params-learning_rate*d_params.T
            self.bias=self.bias-learning_rate*d_bias.T
        return

    def predict(self,X):
        prob=1/(1+np.exp(-np.dot(X,self.params)+self.bias))
        prob=prob[:,0]
        pred=np.where(prob>0.5,1,0)
        return pred

    def score(self,X,y):
        y_pre=self.predict(X)
        score=np.mean(y==y_pre)
        return score

model=logistic_regression()
data=datasets.load_iris()
x=data['data'][:100]
y=data['target'][:100]
print(x.shape,y.shape)
model.train(x,y,iteration=30)
model.predict(x)
print(model.score(x,y))
#plt.scatter(x[:,0],x[:,1],c=model.prob,cmap='seismic')
#plt.show()
