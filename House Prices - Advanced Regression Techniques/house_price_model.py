import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the datasets
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Please download train.csv and test.csv from Kaggle and place them in the same directory as this script.")
    exit()

# Separate target variable
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=np.number).columns

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data: impute + one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Create pipeline with preprocessor and regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train-test split for validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_valid)

# Evaluate the model
print("Model Performance:")
print(f"R^2 Score: {r2_score(y_valid, y_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_valid, y_pred):.2f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_valid, y_pred)):.2f}")

# Predict on Kaggle test data for submission
if 'SalePrice' in test_df.columns:
    test_df = test_df.drop('SalePrice', axis=1)  # safeguard

try:
    test_predictions = model.predict(test_df)
    submission = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_predictions})
    submission.to_csv('submission.csv', index=False)
    print("\n✅ submission.csv created successfully.")
except Exception as e:
    print(f"\n❌ Could not generate submission.csv. Error: {e}")