import pandas as pd
import numpy as np
import tensorflow as tf
import os
os.chdir('/Users/Zhiyan1992/Desktop/')

class neural_network():
    def __init__(self):
        self.trained_model=None

    def train(self,X,Y,num_layer=9,num_neurons=50,activation='relu',epoch=150):
        m=X.shape[1]
        n=X.shape[0]
        model=tf.keras.models.Sequential()
        for i in range(num_layer-1):
            model.add(tf.keras.layers.Dense(num_neurons,activation=activation,kernel_initializer=tf.keras.initializers.he_normal(seed=1)))
        model.add(tf.keras.layers.Dense(units=1, activation=activation,kernel_initializer=tf.keras.initializers.he_normal(seed=1)))
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        model.fit(X,Y,batch_size=50,epochs=epoch)
        self.trained_model=model
        return

    def predict_score(self,X,Y):
        loss,accuracy=self.trained_model.evaluate(X,Y)
        return accuracy

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

    # train model and predict samples
    ANN=neural_network()
    ANN.train(X,Y)
    res=ANN.predict_score(X,Y)
    print('prediction accuracy on training set: ',res)
    res_test=ANN.predict_score(X_test,Y_test)
    print('prediction accuracy on test set: ',res_test)

if __name__=="__main__":
    main()
