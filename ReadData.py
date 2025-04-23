import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ReadDictionary = pd.read_excel('Kaggle-Elo-Merchant-Category-Recommendation\Data\Data Dictionary.xlsx', header=2, sheet_name='train')
ReadSample = pd.read_csv('Kaggle-Elo-Merchant-Category-Recommendation\Data\sample_submission.csv', header=0).head(5)
ReadInfo = pd.read_csv('Kaggle-Elo-Merchant-Category-Recommendation\Data\sample_submission.csv', header=0).info()
ReadTrain = pd.read_csv('Kaggle-Elo-Merchant-Category-Recommendation/Data/train.csv')
ReadTest = pd.read_csv('Kaggle-Elo-Merchant-Category-Recommendation/Data/test.csv')


#observe data

# print(ReadDictionary)
# print(ReadSample)
# print(ReadInfo)
# print(ReadTrain.shape,ReadTest.shape)

#Check data correctness
#unique

# print(ReadTrain['card_id'].nunique() == ReadTrain.shape[0])
# print(ReadTest['card_id'].nunique() == ReadTest.shape[0])
# print(
#     ReadTrain['card_id'].nunique() + ReadTest['card_id'].nunique()
#     == len(set(ReadTrain['card_id']) | set(ReadTest['card_id']))
# )#check if these two files have intersection

#lost

# print(ReadTrain.isnull().sum())
# print(ReadTest.isnull().sum()) #show the number of missing values


#Outliers
# statistics = ReadTrain['target'].describe()#show the target list
# print(statistics)
# sns.histplot(ReadTrain['target'], kde=True)#kde means whether we need a Histogram
# plt.xlabel('target')
# plt.ylabel("count")
# plt.show()#display the histogram which shows the frequency of different data in target list, finding the unique data which may be the outliers

#analyze outlier(⭐ensure train data's trait is similar to test data)

# list trait
features = ['first_active_month','feature_1','feature_2','feature_3']
# train, test data's total count 
train_count = ReadTrain. shape[0]
test_count = ReadTest. shape[0]
#different count / total count = rate
# print(ReadTrain['first_active_month'].value_counts().sort_index()/train_count)
# (ReadTrain['first_active_month'].value_counts().sort_index()/ train_count).plot()
# plt.show()#show the proportion of annual activations to the total activations

#compare two datasets' graphs(single variable)
# for feature in features:
#     (ReadTrain[feature].value_counts().sort_index()/ train_count). plot()
#     (ReadTest[feature].value_counts().sort_index()/ test_count). plot()
#     plt. legend(['train','test'])
#     plt. xlabel(feature)
#     plt. ylabel('ratio')
#     plt. show()

#compare two datasets' graphs(combine variables)
def combine_feature(df):
    cols = df. columns
    feature1 = df[cols[0]]. astype(str). values. tolist()
    feature2 = df[cols[1]]. astype(str). values. tolist()
    return pd. Series([feature1[i]+ '&'+ feature2[i] for i in range(df. shape[0])])
# choose two features
cols = [features[0], features[1]]
print(cols)
# check result of combination 
train_com = combine_feature(ReadTrain[cols])
print(train_com)
train_dis = train_com. value_counts(). sort_index()/ train_count
print(train_dis)
test_dis = combine_feature(ReadTest[cols]). value_counts(). sort_index()/ test_count
print(test_dis)

# create new index
index_dis = pd. Series(train_dis. index. tolist() + test_dis. index. tolist()). drop_duplicates()
# fill lost data by 0
(index_dis. map(train_dis). fillna(0)). plot()
(index_dis. map(train_dis). fillna(0)). plot()

plt. legend(['train','test'])
plt. xlabel('&'. join(cols))
plt. ylabel('ratio')
plt. show()
n = len(features)
for i in range(n- 1):
    for j in range(i+1, n):
        cols = [features[i], features[j]]
        print(cols)
        train_dis = combine_feature(ReadTrain[cols]). value_counts(). sort_index()/ train_count
        test_dis = combine_feature(ReadTest[cols]). value_counts(). sort_index()/ test_count
        index_dis = pd. Series(train_dis. index. tolist() + test_dis. index. tolist()). drop_duplicates()
        (index_dis. map(train_dis). fillna(0)). plot()
        (index_dis. map(test_dis). fillna(0)). plot()
        plt. legend(['train','test'])
        plt. xlabel('&'. join(cols))
        plt. ylabel('ratio')
        plt. show()
#These graphs show that orange and blue lines are almost overlapping, which means these two dataset's have similar outlier