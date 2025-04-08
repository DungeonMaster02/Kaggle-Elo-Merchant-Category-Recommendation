import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ReadDictionary = pd.read_excel('Kaggle-Elo-Merchant-Category-Recommendation\Data\Data Dictionary.xlsx', header=2, sheet_name='train')
ReadSample = pd.read_csv('Kaggle-Elo-Merchant-Category-Recommendation\Data\sample_submission.csv', header=0).head(5)
ReadInfo = pd.read_csv('Kaggle-Elo-Merchant-Category-Recommendation\Data\sample_submission.csv', header=0).info()
ReadTrain = pd.read_csv('Kaggle-Elo-Merchant-Category-Recommendation/Data/train.csv')
ReadTest = pd.read_csv('Kaggle-Elo-Merchant-Category-Recommendation/Data/sample_submission.csv')

#observe data
'''
print(ReadDictionary)
print(ReadSample)
print(ReadInfo)
print(ReadTrain.shape,ReadTest.shape)
'''

#Check data correctness
#unique
'''
print(ReadTrain['card_id'].nunique() == ReadTrain.shape[0])
print(ReadTest['card_id'].nunique() == ReadTest.shape[0])
print(
    ReadTrain['card_id'].nunique() + ReadTest['card_id'].nunique()
    == len(set(ReadTrain['card_id']) | set(ReadTest['card_id']))
)#check if these two files have intersection
'''

#lost
'''
print(ReadTrain.isnull().sum())
print(ReadTest.isnull().sum()) #show the number of missing values
'''

#Outliers
statistics = ReadTrain['target'].describe()#show the target list
print(statistics)

sns.histplot(ReadTrain['target'], kde=True)#kde means whether we need a Histogram
plt.xlabel('target')
plt.ylabel("count")
plt.show()#display the histogram which shows the frequency of different data in target list



