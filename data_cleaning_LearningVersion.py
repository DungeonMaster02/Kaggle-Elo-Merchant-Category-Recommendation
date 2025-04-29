import pandas as pd
import numpy as np
from ReadData import history_transaction, new_transaction

merchant = pd.read_csv('Kaggle-Elo-Merchant-Category-Recommendation/Data/merchants.csv', header=0)
# print(merchant.head(5))
# print(merchant.info())
df = pd.read_excel('Kaggle-Elo-Merchant-Category-Recommendation/Data/Data_Dictionary.xlsx',header=2,sheet_name= 'merchant')
# print(df)

# #explore data
# #correctness(find that 22 merchants corresponds to multiple information)
# print(merchant. shape, merchant['merchant_id']. nunique())
# print(pd.Series(merchant.columns.tolist()).sort_values().values ==  pd.Series([va[0] for va in df.values]).sort_values().values)
# #lost data
# print(merchant.isnull().sum())

#Pre-process merchant data
#1.Labeling discrete and numeric fields
category_cols = ['merchant_id', 'merchant_group_id', 'merchant_category_id',
       'subsector_id', 'category_1',
       'most_recent_sales_range', 'most_recent_purchases_range',
       'category_4', 'city_id', 'state_id', 'category_2']
numeric_cols = ['numerical_1', 'numerical_2',
     'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
       'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6',
       'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12']
#check if the traits are labeled correctly
assert len(category_cols) + len(numeric_cols) == merchant.shape[1]

# #2.check the discrete data
# # check values(show the maximum)
# print(merchant[category_cols].nunique())
# # check types
# print(merchant[category_cols].dtypes)
# # check losts
# print(merchant[category_cols].isnull().sum())

##The results showed that category_2 lost 11887 data
##get the values in this variable
# print(merchant['category_2'].unique())
##Replace missing values ​​with -1
merchant['category_2'] = merchant['category_2'].fillna(-1)
## Dictionary encoding function -> python can't process object directly, so we can convert them to integers(by sort order)
def change_object_cols(se):
    value = se.unique().tolist()
    value.sort()
    return se.map(pd.Series(range(len(value)), index=value)).values
# print(merchant['category_1'])

for col in ['category_1', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']:
    merchant[col] = change_object_cols(merchant[col])
    # print(merchant[col])

# #3.check the numeric data
# print(merchant[numeric_cols].dtypes)
# print(merchant[numeric_cols].isnull().sum())
# print(merchant[numeric_cols].describe())
# #Found inf, process function 1: fill it by the biggest data in this list. 2: delete it
inf_cols = ['avg_purchases_lag3', 'avg_purchases_lag6', 'avg_purchases_lag12']
merchant[inf_cols] = merchant[inf_cols].replace(np.inf, merchant[inf_cols].replace(np.inf, -99).max().max())
# print(merchant[numeric_cols].describe())
# #There are just 13 data in 330 thousand data lost, so we choose to fill them by average value
for col in numeric_cols:
    merchant[col] = merchant[col].fillna(merchant[col].mean())
# print(merchant[numeric_cols].describe())

##Pre-process transaction data
##1.Labeling discrete and numeric fields
numeric_cols = ['installments', 'month_lag', 'purchase_amount']
category_cols = ['authorized_flag', 'card_id', 'city_id', 'category_1',
       'category_3', 'merchant_category_id', 'merchant_id', 'category_2', 'state_id',
       'subsector_id']
time_cols = ['purchase_date']

assert len(numeric_cols) + len(category_cols) + len(time_cols) == new_transaction.shape[1] 
# #2. check the discrete data
# print(new_transaction[category_cols].dtypes)
# print(new_transaction[category_cols].isnull().sum())

#The results showed that 'authorized_flag', 'category_1', 'category_3' lost numerious data
#Dictionary encoding function
for col in ['authorized_flag', 'category_1', 'category_3']:
    new_transaction[col] = change_object_cols(new_transaction[col].fillna(-1).astype(str))    
new_transaction[category_cols] = new_transaction[category_cols].fillna(-1)
# print(new_transaction[category_cols].dtypes)


