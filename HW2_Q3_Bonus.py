import os
import pandas as pd
import numpy as np
import math
from collections import Counter
os.chdir('C:/Users/Zhiyan/Desktop/')
'''Main'''
#preprocessing the data
Data=pd.read_csv(r'default_of_credit_card_clients.csv')
print(Data.shape)
#shuffule the data and split into train and test dataset
index=np.array([1,2,3])
#print(Data.loc[1,3,5])
