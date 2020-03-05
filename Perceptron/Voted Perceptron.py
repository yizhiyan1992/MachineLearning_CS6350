'''
Voted Perceptron
add a counter for each updated weight
'''
import pandas as pd
import numpy as np
import os
os.chdir('C:/Users/Zhiyan/Desktop/')
class Perceptron():
    def __init__(self):
        self.w=None
        self.b=0
        self.epoch=1
        self.learning_rate=1
        self.counter=[]

    def Output_params(self):
        return self.w,self.b

    def Train_model(self,x,y,epoch=None,learning_rate=None):
        #intialize w, epoch, and learning rate
        if epoch:
            self.epoch=epoch
        if learning_rate:
            self.learning_rate=learning_rate

        self.w=np.array([0 for _ in range(x.shape[1])])
        self.counter.append([self.w,self.b,1])
        # go through every epoch
        for epo in range(self.epoch):
        #    # go through all samples in one epoch
            for sam in range(x.shape[0]):
                 # if prediction is wrong, then update w and b
                if y[sam]*(np.dot(self.w,np.transpose(x[sam,:]))+self.b)<=0:
                    self.w=self.w+self.learning_rate*y[sam]*np.transpose(x[sam,:])
                    self.b=self.b+self.learning_rate*y[sam]
                    self.counter.append([self.w,self.b,1])
                else:
                    self.counter[-1][2]+=1
        return

    def Predict(self,x):
        res=[]
        for sam in range(x.shape[0]):
            temp=0
            #  res=sgn(sum(counter*sgn(wx+b)))
            for i in range(len(self.counter)):
                if np.dot(self.counter[i][0], np.transpose(x[sam, :])) + self.counter[i][1] > 0:
                    temp+=1*self.counter[i][2]
                else:
                    temp-=1*self.counter[i][2]
            if temp>0:
                res.append(1)
            else:
                res.append(-1)
        return np.array(res)

    def Accuracy(self,y1,y2):
        count=0
        for sam in range(len(y1)):
            if y1[sam]!=y2[sam]:
                count+=1
        return count/len(y1)

def main():
    Data=pd.read_csv('bank-note/train.csv',header=None)
    Test=pd.read_csv('bank-note/test.csv',header=None)
    #change the label into +1 and -1
    Data[4][Data[4]==0]=-1
    Test[4][Test[4] == 0] = -1
    train_x=Data.values[:,:-1]
    train_y=Data.values[:,-1]
    test_x=Test.values[:,:-1]
    test_y=Test.values[:,-1]

    perceptron=Perceptron()
    perceptron.Train_model(train_x,train_y,epoch=10)
    res=perceptron.Predict(test_x)
    print('The prediction error is:',perceptron.Accuracy(test_y,res))

if __name__=='__main__':
    main()
