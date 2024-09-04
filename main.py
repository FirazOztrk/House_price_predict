import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load data
df = pd.read_csv('Last_train.csv')

# Convert float64 columns to categorical and then to numerical codes
float64_categorical_columns = ['MSZoning', 'Neighborhood', 'BsmtQual', 'BsmtFinType1', 'CentralAir', 'GarageType']
for col in float64_categorical_columns:
    if col in df.columns:
        df[col] = pd.Categorical(df[col]).codes

# Ensure numeric columns are correctly typed
numeric_columns = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea']
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Define features and target
X = df.drop(columns=['SalePrice'])  # Features
y = df['SalePrice']                 # Target variable

# Define the threshold for feature importance
threshold = 0.005

# Train a RandomForestRegressor to determine feature importances
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Get feature importances from the trained RandomForestRegressor
importances = rf_model.feature_importances_
feature_names = X.columns

# Sort feature importances in descending order
sorted_importances = sorted(zip(importances, feature_names), reverse=True)

# List of features to drop based on the threshold
features_to_drop = [name for importance, name in sorted_importances if importance < threshold]

# Drop these features from the DataFrame
X_reduced = X.drop(columns=features_to_drop)  # Features (excluding low-importance features)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model with the best parameters
best_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    subsample=0.6,
    n_estimators=200,
    min_child_weight=1,
    max_depth=7,
    learning_rate=0.05,
    colsample_bytree=1.0,
    random_state=42
)
best_model.fit(X, y)

# Evaluate the model on the validation set
y_pred = best_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Mean Squared Error on Validation Set: {mse:.2f}")

# Optionally save the model for later use
joblib.dump(best_model, 'final_xgboost_model.pkl')

print("Final model trained with all data.")
