import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv('data.csv')
print(df.head(10))
print(df.select_dtypes(exclude=[np.number]).columns)

# ['MSZoning', 'Neighborhood', 'BsmtQual', 'BsmtFinType1', 'CentralAir','GarageType']
print(df['Neighborhood'].value_counts())
