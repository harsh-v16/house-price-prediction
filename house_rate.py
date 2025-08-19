# ===================================================================
# HOUSE PRICE PREDICTION - A COMPLETE DATA SCIENCE PROJECT
# ===================================================================

# --- PART 1: DATA LOADING AND INITIAL ANALYSIS ---
# In this section, we load the data and perform an initial analysis to
# understand its structure and identify any immediate issues like skewness.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Keep the test IDs for the submission file
test_ids = test_data['Id']

# --- PART 2: EXPLORATORY DATA ANALYSIS (EDA) & PREPROCESSING ---
# The goal here is to fix the skewness in the target variable (SalePrice),
# understand feature relationships, handle missing values, and prepare
# categorical features for the model.

# 2a. Skewness Analysis and Correction
# First, we visually inspect the distribution of SalePrice using a histogram
# and a Q-Q plot.

fig,axes = plt.subplots(1, 2, figsize = (15, 5))
sns.histplot(train_data['SalePrice'], kde=True, ax=axes[0])
axes[0].set_title('Histogram of SalePrice')
stats.probplot(train_data['SalePrice'], plot=axes[1])
axes[1].set_title('Q-Q Plot of SalePrice')
plt.show()

# We then perform a mathematical check for skewness. A value close to 0
# indicates a symmetrical distribution. A value > 1 indicates high positive skew.

print(f"Skewness : {train_data['SalePrice'].skew()}")

# To correct the skewness, we apply a log transformation. This helps the model
# make more reliable predictions as many models assume a normal distribution.

train_data['SalePrice_log'] = np.log1p(train_data['SalePrice'])
fig,axes = plt.subplots(1, 2, figsize = (15, 5))
sns.histplot(train_data['SalePrice_log'], kde=True, ax=axes[0])
axes[0].set_title('Histogram of Log-Transformed SalePrice_log')
stats.probplot(train_data['SalePrice_log'], plot=axes[1])
axes[1].set_title('Q-Q Plot of Log-Transformed SalePrice_log')
plt.show()

# We check the new skewness value to confirm our transformation was successful.

print(f"New Skewness : {train_data['SalePrice_log'].skew()}")

# 2b. Correlation Analysis for Feature Selection
# We create a correlation heatmap to understand the relationships between
# features and our target variable, 'SalePrice_log'. This helps us select
# the most powerful and relevant features for a simple baseline model and
# understand potential multicollinearity (features that are highly
# correlated with each other)

corrmat = train_data.corr(numeric_only=True)
k = 10
top_10_cols = corrmat.nlargest(k, 'SalePrice_log')['SalePrice_log'].index
top_10_corrmat = train_data[top_10_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(top_10_corrmat, annot=True, cmap="YlGnBu", fmt='.2f')
plt.title('Top 10 Feature Correlations with SalePrice_log')
plt.show()

# 2c. Handling Missing Values

train_data['LotFrontage'].fillna(train_data['LotFrontage'].median(), inplace=True)
test_data['LotFrontage'].fillna(train_data['LotFrontage'].median(), inplace=True)

train_data['Electrical'].fillna(train_data['Electrical'].mode()[0], inplace=True)
test_data['Electrical'].fillna(train_data['Electrical'].mode()[0], inplace=True)

fill_with_zero = ['MasVnrArea', 'GarageYrBlt','BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 'GarageCars', 'GarageArea']
for cols in fill_with_zero:
    train_data[cols] = train_data[cols].fillna(0)
    test_data[cols] = test_data[cols].fillna(0)

fill_with_none = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
for features in fill_with_none:
    train_data[features] = train_data[features].fillna('none')
    test_data[features] = test_data[features].fillna('none')

test_fill_with_mode = ['MSZoning', 'Utilities', 'Functional', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType']
for col in test_fill_with_mode:
    mode_val = train_data[col].mode()[0]
    test_data[col] = test_data[col].fillna(mode_val)

missing_vals = train_data.isnull().sum()
cols_with_missing = missing_vals[missing_vals > 0].index.tolist()

numerical_cols_missing = train_data[cols_with_missing].select_dtypes(include=np.number).columns.tolist()
categorical_cols_missing = train_data[cols_with_missing].select_dtypes(include='object').columns.tolist()

print("Numeric empty spots")
print(numerical_cols_missing)
print("Categorical empty spots")
print(categorical_cols_missing)

# 2d. One-Hot Encoding

categorical_cols = [cname for cname in train_data.columns if train_data[cname].dtype == 'object']
train_data = pd.get_dummies(train_data, columns=categorical_cols)
test_data = pd.get_dummies(test_data, columns=categorical_cols)

# --- PART 3: MODELING & EVALUATION ---
# Now we will build two models:
# 1. A simple Linear Regression model as a baseline.
# 2. A powerful XGBoost model to achieve high performance.
# We will compare their R-squared and RMSE scores to choose a winner.

y = train_data['SalePrice_log']
train_data, test_data = train_data.align(test_data, join='inner', axis=1)

selected_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF']
X_simple = train_data[selected_features]

X_simple_train, X_simple_val, y_simple_train, y_simple_val = train_test_split(X_simple, y, test_size=0.2, random_state=42)

# 3a. Baseline Model: Linear Regression

simple_model = LinearRegression()
simple_model.fit(X_simple_train, y_simple_train)
predictions = simple_model.predict(X_simple_val)

r2 = r2_score(y_simple_val, predictions)
print(f"R-squared (R2 Score): {r2:.4f}")

mse = mean_squared_error(y_simple_val, predictions)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

X_complex = train_data.drop(['Id',], axis=1)

X_train, X_val, y_train, y_val = train_test_split(X_complex, y, test_size=0.2, random_state=42)

# 3b. High-Performance Model: XGBoost

complex_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)      
complex_model.fit(X_train, y_train) 
complex_predictions = complex_model.predict(X_val)  

r2 = r2_score(y_val, complex_predictions)
print(f"R-squared (R2 Score): {r2:.4f}")

mse = mean_squared_error(y_val, complex_predictions)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# --- PART 4: FINAL SUBMISSION ---
# After proving that XGBoost is the superior model, we re-train it on
# 100% of the training data and use it to predict the prices for the
# official test set, creating the final submission file.

test_data_r = test_data.drop(['Id'], axis=1)

final_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)      
final_model.fit(X_complex, y) 
predictions_log = final_model.predict(test_data_r)
predictions_actual = np.expm1(predictions_log)

submission = pd.DataFrame({'Id': test_ids, 'SalePrice': predictions_actual})
submission.to_csv('submission.csv', index=False)

print("\n--- SUCCESS! ---")
print("Submission file 'submission.csv' has been created.")
