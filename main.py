import os
import numpy as np
import pandas as pd

ReadDictionary = pd.read_excel('Data/Data Dictionary.xlsx', header=2, sheet_name='train')
print(ReadDictionary)

ReadSample = pd.read_csv('Data/sample_submission.csv', header=0).head(5)
print(ReadSample)