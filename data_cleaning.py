import gc
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Load data
train = pd.read_csv('Kaggle-Elo-Merchant-Category-Recommendation/Data/train.csv')
test = pd.read_csv('Kaggle-Elo-Merchant-Category-Recommendation/Data/test.csv')
merchant = pd.read_csv('Kaggle-Elo-Merchant-Category-Recommendation/Data/merchants.csv')
new_transaction = pd.read_csv('Kaggle-Elo-Merchant-Category-Recommendation/Data/new_merchant_transactions.csv')
history_transaction = pd.read_csv('Kaggle-Elo-Merchant-Category-Recommendation/Data/historical_transactions.csv')

# Define encoding function for categorical features
def change_object_cols(se):
    value = se.unique().tolist()
    value.sort()
    return se.map(pd.Series(range(len(value)), index=value)).values

# Encode 'first_active_month' feature in train and test datasets
se_map = change_object_cols(pd.concat([train['first_active_month'], test['first_active_month']]).astype(str))
train['first_active_month'] = se_map[:train.shape[0]]
test['first_active_month'] = se_map[train.shape[0]:]

train.to_csv('Kaggle-Elo-Merchant-Category-Recommendation/Data/train_pre.csv', index=False)
test.to_csv('Kaggle-Elo-Merchant-Category-Recommendation/Data/test_pre.csv', index=False)

del train
        
del test
gc.collect()

# Process merchant data
category_cols = ['merchant_id', 'merchant_group_id', 'merchant_category_id',
                 'subsector_id', 'category_1',
                 'most_recent_sales_range', 'most_recent_purchases_range',
                 'category_4', 'city_id', 'state_id', 'category_2']
numeric_cols = ['numerical_1', 'numerical_2',
                'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
                'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6',
                'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12']

# Encode selected categorical columns
for col in ['category_1', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']:
    merchant[col] = change_object_cols(merchant[col])

# Fill missing values in categorical columns
merchant[category_cols] = merchant[category_cols].fillna(-1)

# Replace inf values in numeric columns
inf_cols = ['avg_purchases_lag3', 'avg_purchases_lag6', 'avg_purchases_lag12']
merchant[inf_cols] = merchant[inf_cols].replace(np.inf, merchant[inf_cols].replace(np.inf, -99).max().max())

# Fill missing values in numeric columns with mean
for col in numeric_cols:
    merchant[col] = merchant[col].fillna(merchant[col].mean())

# Drop duplicate columns except 'merchant_id'
duplicate_cols = ['merchant_id', 'merchant_category_id', 'subsector_id', 'category_1', 'city_id', 'state_id', 'category_2']
merchant = merchant.drop(columns=duplicate_cols[1:])
merchant = merchant.loc[merchant['merchant_id'].drop_duplicates().index.tolist()].reset_index(drop=True)

# Merge new and history transactions
transaction = pd.concat([new_transaction, history_transaction], axis=0, ignore_index=True)
del new_transaction
        
del history_transaction
gc.collect()

# Define column types
numeric_cols = ['installments', 'month_lag', 'purchase_amount']
category_cols = ['authorized_flag', 'card_id', 'city_id', 'category_1',
                 'category_3', 'merchant_category_id', 'merchant_id', 'category_2', 'state_id',
                 'subsector_id']
time_cols = ['purchase_date']

# Encode selected categorical columns and fill missing values
for col in ['authorized_flag', 'category_1', 'category_3']:
    transaction[col] = change_object_cols(transaction[col].fillna(-1).astype(str))
transaction[category_cols] = transaction[category_cols].fillna(-1)
transaction['category_2'] = transaction['category_2'].astype(int)

# Generate new time features
transaction['purchase_month'] = transaction['purchase_date'].apply(lambda x: '-'.join(x.split(' ')[0].split('-')[:2]))
transaction['purchase_hour_section'] = transaction['purchase_date'].apply(lambda x: int(x.split(' ')[1].split(':')[0]) // 6)
transaction['purchase_day'] = transaction['purchase_date'].apply(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d').weekday() // 5)

# Drop original purchase_date column
transaction.drop(columns=['purchase_date'], inplace=True)

# Select columns for merging
cols = ['merchant_id', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']

# Important: Fill missing and cast types to int8 before merge to avoid memory issues
merchant[cols[1:]] = merchant[cols[1:]].fillna(-1).astype('int8')

# Merge merchant selected columns into transaction
transaction = pd.merge(transaction, merchant[cols], how='left', on='merchant_id')

# Update numeric and categorical fields
numeric_cols = ['purchase_amount', 'installments']

category_cols = ['authorized_flag', 'city_id', 'category_1',
                 'category_3', 'merchant_category_id', 'month_lag',
                 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4',
                 'purchase_month', 'purchase_hour_section', 'purchase_day']

id_cols = ['card_id', 'merchant_id']

# Fill missing values
transaction[cols[1:]] = transaction[cols[1:]].fillna(-1).astype(int)
transaction[category_cols] = transaction[category_cols].fillna(-1).astype(str)

# Save processed transaction
transaction.to_csv('Kaggle-Elo-Merchant-Category-Recommendation/Data/transaction_d_pre.csv', index=False)

# Free memory
del transaction
gc.collect()
