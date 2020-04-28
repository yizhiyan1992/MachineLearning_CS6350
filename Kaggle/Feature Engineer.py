import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
import os

os.chdir('/Users/Zhiyan1992/Desktop/')


def load_data():
    train = pd.read_csv('income-predict/train_final.csv')
    test = pd.read_csv('income-predict/test_final.csv')
    return train, test


def missing_labels(train_x, test_x):
    for col in train_x.columns:
        if '?' in train_x[col].values:
            temp = Counter(train_x[col].values)
            most_freq = temp.most_common(1)[0][0]
            train_x[col][train_x[col] == '?'] = most_freq
            test_x[col][test_x[col] == '?'] = most_freq
    return train_x, test_x


def feature_observation(train):
    matplotlib.rcParams.update({'font.size': 8})
    col_name = train.columns
    print(col_name)
    sns.set_palette('deep', desat=0.6)
    sns.set()
    ax = plt.axes()
    '''
    plt.subplot(4, 4, 1)
    plt.title('age')
    plt.hist(x=train[col_name[0]], bins=len(set(train[col_name[0]].values)), alpha=0.8)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(4, 4, 2)
    plt.title('workclass')
    pic = sns.countplot(x='workclass', data=train)
    pic.set_xticklabels(pic.get_xticklabels(), rotation=30, fontsize=6)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(4, 4, 3)
    plt.title('fnlwgt')
    plt.hist(x=train[col_name[0]], bins=80, alpha=0.8)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(4, 4, 4)
    plt.title('education')
    pic = sns.countplot(x='education', data=train)
    pic.set_xticklabels(pic.get_xticklabels(), rotation=30, fontsize=6)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(4, 4, 5)
    ax.set_xticks([])
    plt.title('education.num')
    plt.hist(x=train[col_name[0]], bins=len(set(train[col_name[4]].values)), alpha=0.8)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(4, 4, 6)
    '''
    plt.title('marital status')
    pic = sns.countplot(x='marital.status', data=train)
    pic.set_xticklabels(pic.get_xticklabels(), rotation=30, fontsize=6)
    ##plt.xticks([])
    #plt.yticks([])
    '''
    plt.subplot(4, 4, 7)
    plt.title('occupation')
    pic = sns.countplot(x='occupation', data=train)
    pic.set_xticklabels(pic.get_xticklabels(), rotation=30, fontsize=6)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(4, 4, 8)
    plt.title('relationship')
    pic = sns.countplot(x='relationship', data=train)
    pic.set_xticklabels(pic.get_xticklabels(), rotation=30, fontsize=6)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(4, 4, 9)
    plt.title('race')
    pic = sns.countplot(x='race', data=train)
    pic.set_xticklabels(pic.get_xticklabels(), rotation=30, fontsize=6)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(4, 4, 10)
    plt.title('sex')
    pic = sns.countplot(x='sex', data=train)
    pic.set_xticklabels(pic.get_xticklabels(), rotation=30, fontsize=6)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(4, 4, 11)
    plt.title('capital.gain')
    plt.hist(x=train[col_name[0]], bins=len(set(train[col_name[10]].values)), alpha=0.8)
    plt.xticks([])
    plt.subplot(4, 4, 12)
    plt.title('capital.loss')
    plt.hist(x=train[col_name[0]], bins=len(set(train[col_name[11]].values)), alpha=0.8)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(4, 4, 13)
    plt.title('hours.per.week')
    plt.hist(x=train[col_name[0]], bins=len(set(train[col_name[12]].values)), alpha=0.8)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(4, 4, 14)
    plt.title('native.country')
    pic = sns.countplot(x='native.country', data=train)
    pic.set_xticklabels(pic.get_xticklabels(), rotation=30, fontsize=6)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.savefig('32222.png')
    '''

    plt.show()


def main():
    train_orgin,test_origin=load_data()
    train_processed,test_processed=missing_labels(train_orgin,test_origin)
    feature_observation(train_processed)

if __name__=="__main__":
    main()
