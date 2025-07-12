import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# Load data from ZIP file
with zipfile.ZipFile("vehicle_price_prediction.zip", "r") as z:
    with z.open("Vehicle Price Prediction/dataset.csv") as f:
        df = pd.read_csv(f)

# Drop rows with missing critical values and duplicates
df = df.dropna(subset=["price", "mileage", "make", "model", "year"])
df = df.drop_duplicates()

# Fill missing values in categorical columns
cat_cols = [
    'engine', 'cylinders', 'fuel', 'transmission', 'trim',
    'body', 'doors', 'exterior_color', 'interior_color', 'drivetrain'
]
for col in cat_cols:
    df[col] = df[col].fillna("Unknown")

# Add 'age' feature and drop unnecessary columns
df['age'] = 2025 - df['year']
df = df.drop(columns=["year", "description", "name"])

# Log-transform target variable
df['log_price'] = np.log1p(df['price'])

# Separate features and target
X = df.drop(columns=["price", "log_price"])
y = df['log_price']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# List of categorical columns
categorical_cols = [
    'make', 'model', 'engine', 'cylinders', 'fuel',
    'transmission', 'trim', 'body', 'doors',
    'exterior_color', 'interior_color', 'drivetrain'
]

# Ensure all categorical columns are string type
for col in categorical_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# Identify numerical columns
numerical_cols = X_train.columns.difference(categorical_cols)

# One-hot encode categorical variables
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat = ohe.fit_transform(X_train[categorical_cols])
X_test_cat = ohe.transform(X_test[categorical_cols])

# Convert encoded arrays to DataFrames
X_train_cat_df = pd.DataFrame(
    X_train_cat, columns=ohe.get_feature_names_out(categorical_cols), index=X_train.index
)
X_test_cat_df = pd.DataFrame(
    X_test_cat, columns=ohe.get_feature_names_out(categorical_cols), index=X_test.index
)

# Combine numerical and encoded categorical features
X_train_final = pd.concat([X_train[numerical_cols], X_train_cat_df], axis=1)
X_test_final = pd.concat([X_test[numerical_cols], X_test_cat_df], axis=1)

# Clean column names for XGBoost
X_train_final.columns = X_train_final.columns.str.replace('[\[\]<>\s]', '_', regex=True)
X_test_final.columns = X_test_final.columns.str.replace('[\[\]<>\s]', '_', regex=True)

# Define XGBoost regressor
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Set up hyperparameter grid
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1],
    'colsample_bytree': [0.7, 0.8, 1]
}

# Randomized search for best parameters
search = RandomizedSearchCV(
    xgb_reg, param_grid, n_iter=10, cv=3,
    scoring='neg_mean_absolute_error', verbose=1, random_state=42
)
search.fit(X_train_final, y_train)

# Predict on test set
y_pred_log = search.best_estimator_.predict(X_test_final)
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)

# Print evaluation metrics
print("Mean Absolute Error:", mean_absolute_error(y_test_orig, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test_orig, y_pred)))
print("R^2 Score:", r2_score(y_test_orig, y_pred))

# Plot only the top 10 most important features
plt.figure(figsize=(10, 6))
xgb.plot_importance(
    search.best_estimator_,
    max_num_features=10,
    height=0.8,
    importance_type='weight'
)
plt.title("Top 10 XGBoost Feature Importances")
plt.tight_layout()
plt.show()

# Save the trained model and encoder for future use
joblib.dump(search.best_estimator_, "vehicle_price_model.pkl")
joblib.dump(ohe, "onehotencoder.pkl")