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
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Automatically find dataset in the same folder as the script
file_path = os.path.join(os.path.dirname(__file__), "MLSData_1.29.25.csv")

# Load the dataset
df = pd.read_csv(file_path)
print(df.info())

df.dropna(inplace=True)  # Remove missing values

# Exploratory Data Analysis (EDA) of Sale Price & PPSF
plt.figure(figsize=(10, 6))
sns.histplot(df['SALE_PRICE'], kde=True)
plt.title('Distribution of House Prices')
plt.show()

# Calculate Price Per Square Foot (PPSF)
df["PRICE_PER_SQFT"] = df["SALE_PRICE"] / df["SQUARE_FEET"]

sns.boxplot(x=df["PRICE_PER_SQFT"])
plt.xlabel("PRICE_PER_SQFT ($)")
plt.title("Box Plot of PRICE_PER_SQFT")
plt.show()

# Remove outliers using the Interquartile Range (IQR)
Q1, Q3 = df["PRICE_PER_SQFT"].quantile([0.25, 0.75])
IQR = Q3 - Q1
df = df[(df["PRICE_PER_SQFT"] >= (Q1 - 1.5 * IQR)) & (df["PRICE_PER_SQFT"] <= (Q3 + 1.5 * IQR))]

# Review distribution of Sale Price & PPSF
plt.figure(figsize=(10, 6))
sns.histplot(df['SALE_PRICE'], kde=True)
plt.title('Distribution of House Prices (After Outlier Removal)')
plt.show()

sns.boxplot(x=df["PRICE_PER_SQFT"])
plt.xlabel("PRICE_PER_SQFT ($)")
plt.title("Box Plot of PRICE_PER_SQFT (After Outlier Removal)")
plt.show()

# Define 'Luxury' Condos as the top 10% based on PPSF
luxury_threshold = df["PRICE_PER_SQFT"].quantile(0.90)
df["CONDO_TYPE"] = np.where(df["PRICE_PER_SQFT"] >= luxury_threshold, "Luxury", "Standard")
df.drop(columns=["PRICE_PER_SQFT"], inplace=True)  # Prevent data leakage

# Convert Bathroom Description into Separate Columns
def extract_baths(bath_desc):
    parts = str(bath_desc).lower().split()
    full_baths = int(parts[0][:-1]) if "f" in parts[0] else 0
    half_baths = int(parts[1][:-1]) if len(parts) > 1 and "h" in parts[1] else 0
    return full_baths, half_baths

df[['FULL_BATHS', 'HALF_BATHS']] = df['BTH_DESC'].apply(lambda x: pd.Series(extract_baths(x)))
df.drop(columns=['BTH_DESC'], inplace=True)

# Convert ZIP Code to string format (preserve leading zeros)
df["ZIP_CODE"] = df["ZIP_CODE"].astype(str).str.zfill(5)

# One-Hot Encode all categorical variables at once
df = pd.get_dummies(df, columns=["TOWN", "CONDO_TYPE", "ZIP_CODE"], drop_first=True)

# Convert Year Built into Home Age
df["HOME_AGE"] = 2025 - df["YEAR_BUILT"]
df.drop(columns=["YEAR_BUILT"], inplace=True)

# Scale numerical features
scaled_features = ["NO_BEDROOMS", "SQUARE_FEET", "HOME_AGE", "GARAGE_SPACES_CC", "FULL_BATHS", "HALF_BATHS"]
scaler = StandardScaler()
df[scaled_features] = scaler.fit_transform(df[scaled_features])

# Define Features (X) and Target (y)
X = df.drop(columns=["SALE_PRICE"])
y = df["SALE_PRICE"]

# Split Data into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Function to Train & Evaluate Models
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """Trains a model and returns performance metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "Model": model.__class__.__name__,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²": r2_score(y_test, y_pred),
        "Trained_Model": model
    }

# Define models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "KNN": GridSearchCV(KNeighborsRegressor(), {"n_neighbors": range(3, 11)}, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
}

# Train & Evaluate Models
results = []
for name, model in models.items():
    model_metrics = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
    results.append([name, model_metrics["MAE"], model_metrics["RMSE"], model_metrics["R²"], model_metrics["Trained_Model"]])

# Convert to DataFrame and sort correctly
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R²", "Trained_Model"]).sort_values(by="MAE", ascending=True)

# Remove "Trained_Model" before plotting
plot_df = results_df.drop(columns=["Trained_Model"])


# Select Best Model Automatically (Drop "Trained_Model" before sorting)
best_model = results_df.sort_values(by="MAE").iloc[0]["Trained_Model"]

# Compute Mean Absolute Error (MAE) on the test set
y_test_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_test_pred)


# Save everything 
joblib.dump({
    "model": best_model,
    "scaler": scaler,
    "mae": mae,  # Use mae instead of best_mae
    "feature_order": X.columns.tolist()
}, "model_data.pkl")

print(f"\n✅ Model, Scaler, and MAE Saved Successfully! Best Model: {best_model.__class__.__name__}, MAE: ${mae:,.2f}")

# Print Final Performance Table
print("\n✅ Final Model Performance Table:\n")
print(results_df.drop(columns=["Trained_Model"]).to_string(index=False))

# Plot Model Performance
plt.figure(figsize=(12, 5))

# MAE Bar Plot
plt.subplot(1, 2, 1)
sns.barplot(x="Model", y="MAE", hue="Model", data=plot_df, palette="viridis")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("MAE Comparison")

# R² Score Bar Plot
plt.subplot(1, 2, 2)
sns.barplot(x="Model", y="R²", hue="Model", data=results_df, palette="coolwarm", dodge=False)
plt.ylabel("R² Score")
plt.title("R² Score Comparison")

plt.tight_layout()
plt.show()
