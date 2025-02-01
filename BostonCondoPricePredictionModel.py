# Home Price Prediction Model

# Author: Krina Patel
# Date: January 29, 2025
# Description: Predicting condo sale prices in Boston using multiple regression models.
# Models Used: Random Forest, Gradient Boosting, Linear Regression, KNN.

# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

# Path to the dataset
file_path = "MLS Data 1.29.25.csv"

# Load the MLS dataset into a DataFrame
df = pd.read_csv(file_path)

# Drop rows with missing values (ensures complete dataset)
df.dropna(inplace=True)

# Exploratory Data Analysis (EDA)

# Plot the distribution of house prices
plt.figure(figsize=(10, 6))
sns.histplot(df['SALE_PRICE'], kde=True)
plt.title('Distribution of House Prices')
plt.show()

# Calculate Price Per Square Foot (PPSF) for outlier detection
df["PRICE_PER_SQFT"] = df["SALE_PRICE"] / df["SQUARE_FEET"]

# Boxplot to visualize PPSF distribution
sns.boxplot(x=df["PRICE_PER_SQFT"])
plt.xlabel("PRICE_PER_SQFT ($)")
plt.title("Box Plot of PRICE_PER_SQFT")
plt.show()

# Remove Outliers

# Define the Interquartile Range (IQR) to remove extreme outliers
Q1 = df["PRICE_PER_SQFT"].quantile(0.25)
Q3 = df["PRICE_PER_SQFT"].quantile(0.75)
IQR = Q3 - Q1

# Keep values within 1.5 * IQR range (Removes extreme outliers)
df = df[(df["PRICE_PER_SQFT"] >= (Q1 - 1.5 * IQR)) & (df["PRICE_PER_SQFT"] <= (Q3 + 1.5 * IQR))]

# Recheck PPSF distribution boxplot after outlier removal
sns.boxplot(x=df["PRICE_PER_SQFT"])
plt.xlabel("PRICE_PER_SQFT ($)")
plt.title("Box Plot of PRICE_PER_SQFT")
plt.show()

# Recheck Sales Price distribution histogram after outlier removal
plt.figure(figsize=(10, 6))
sns.histplot(df['SALE_PRICE'], kde=True)
plt.title('Distribution of House Prices')
plt.show()

# Since extreme luxury condos are still skewing the data to the right, we should categorize these
# so their extreme PPSF is accounted for with a separate variable

# Define 'Luxury' Condos as the top 10% based on PPSF
luxury_threshold = df['PRICE_PER_SQFT'].quantile(0.90)
df['CONDO_TYPE'] = np.where(df['PRICE_PER_SQFT'] >= luxury_threshold, 'Luxury', 'Standard')

# Drop PPSF column to prevent data leakage!! 
df.drop(columns=["PRICE_PER_SQFT"], inplace=True)

# Convert Bath Description into Separate Columns
def extract_baths(bath_desc):
    parts = str(bath_desc).lower().split()
    full_baths = int(parts[0][:-1]) if 'f' in parts[0] else 0
    half_baths = int(parts[1][:-1]) if len(parts) > 1 and 'h' in parts[1] else 0
    return full_baths, half_baths

df[['FULL_BATHS', 'HALF_BATHS']] = df['BTH_DESC'].apply(lambda x: pd.Series(extract_baths(x)))
df.drop(columns=['BTH_DESC'], inplace=True)

# Convert ZIP Code to string format (preserve leading zeros)
df["ZIP_CODE"] = df["ZIP_CODE"].astype(str).str.zfill(5)

# One-Hot Encode Categorical Variables (Drop first to prevent collinearity)
df = pd.get_dummies(df, columns=["TOWN", "CONDO_TYPE", "ZIP_CODE"], drop_first=True)

# Convert Year Built into Home Age (More useful for prediction)
df["HOME_AGE"] = 2025 - df["YEAR_BUILT"]
df.drop(columns=["YEAR_BUILT"], inplace=True)  # Remove redundant feature

# Only scale numerical features (not categorical)
scaled_features = ["NO_BEDROOMS", "SQUARE_FEET", "HOME_AGE", "GARAGE_SPACES_CC", "FULL_BATHS", "HALF_BATHS"]
scaler = StandardScaler()
df[scaled_features] = scaler.fit_transform(df[scaled_features])

print("Final Column Names by Type:")
print("\nCategorical Variables:", [col for col in df.columns if df[col].dtype in ['uint8', 'bool']])
print("\nNumeric Features:", [col for col in df.columns if df[col].dtype in ['int64', 'float64', 'uint8']])

# Define Features (X) and Target (y)
X = df.drop(columns=["SALE_PRICE"])
y = df["SALE_PRICE"]

print("Final Column Names in Dataset:")
print(df.columns.tolist())  # Print all column names as a list

# Split Data into Training & Testing Sets (75% Train, 25% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"Training Data: {X_train.shape}, Testing Data: {X_test.shape}")

# Function to Compute Model Evaluation Metrics
def compute_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²": r2_score(y_true, y_pred)
    }

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# K-Nearest Neighbors Model (Hyperparameter Tuning)
knn_params = {"n_neighbors": range(3, 11), "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]}
grid_search_knn = GridSearchCV(KNeighborsRegressor(), knn_params, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
grid_search_knn.fit(X_train, y_train)
knn_model = grid_search_knn.best_estimator_
y_pred_knn = knn_model.predict(X_test)

# Gradient Boosting Model
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# Compute Metrics for All Models
metrics_rf = compute_metrics(y_test, y_pred_rf)
metrics_lr = compute_metrics(y_test, y_pred_lr)
metrics_knn = compute_metrics(y_test, y_pred_knn)
metrics_gb = compute_metrics(y_test, y_pred_gb)

# Store Results in DataFrame
results = pd.DataFrame({
    "Model": ["Random Forest", "Gradient Boosting", "Linear Regression", "KNN"],
    "MAE": [metrics_rf["MAE"], metrics_gb["MAE"], metrics_lr["MAE"], metrics_knn["MAE"]],
    "RMSE": [metrics_rf["RMSE"], metrics_gb["RMSE"], metrics_lr["RMSE"], metrics_knn["RMSE"]],
    "R²": [metrics_rf["R²"], metrics_gb["R²"], metrics_lr["R²"], metrics_knn["R²"]]
}).sort_values(by="MAE")


# Plot 
plt.figure(figsize=(12, 5))

# MAE Bar Plot
plt.subplot(1, 2, 1)
sns.barplot(x="Model", y="MAE", hue="Model", data=results, palette="viridis", dodge=False)
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("MAE Comparison")

# R² Score Bar Plot
plt.subplot(1, 2, 2)
sns.barplot(x="Model", y="R²", hue="Model", data=results, palette="coolwarm", dodge=False)
plt.ylabel("R² Score")
plt.title("R² Score Comparison")

plt.tight_layout()
plt.show()

# Print Model Performance
print("\n Model Performance Comparison:\n")
print(results.to_string(index=False))
