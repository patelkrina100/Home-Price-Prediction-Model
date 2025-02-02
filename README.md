# 🏠 Home Price Prediction Model  

**Author**: Krina Patel  
**Description**: Predicting condo sale prices in Boston using multiple regression models.  

## 🚀 Overview  
This project leverages **machine learning** to predict condo sale prices in **Boston** using real estate data sourced from an **MA MLS search**. The dataset includes key features such as square footage, number of bedrooms, location, and property age. The models used for prediction include:  

- **Random Forest** 🌲  
- **Gradient Boosting** 🚀  
- **Linear Regression** 📈  
- **K-Nearest Neighbors (KNN)** 📊  

The trained model is deployed using **Flask** and can be accessed via a **web app** for real-time price predictions.

---

## 🎯 Why This Project Matters  
Real estate pricing is **complex**, influenced by factors like **location, square footage, and market trends**. This model provides **data-driven insights** for:  

✔ **Real Estate Investors** → Helps estimate property values for better investment decisions.  
✔ **Home Buyers & Sellers** → Assists in determining **fair market value**.  
✔ **Real Estate Agents & Analysts** → Supports **pricing strategies** and trend forecasting.  

This project enhances traditional **comparative market analysis (CMA)** by leveraging **machine learning** for more accurate price predictions.

---

## 📂 Dataset  

- **Source**: Massachusetts MLS Search  
- **File**: `MLSData_1.29.25.csv` *(not included in GitHub for size/privacy reasons)*  
- **Search Criteria**:
  - **Property Type**: Condos  
  - **Timeframe**: Last 2 years  
  - **Towns Included**: All major **Boston** neighborhoods  
  - **Bedrooms**: 0-3  
  - **Total Baths**: 1+  
- **Key Features**:
  - `SALE_PRICE` → Condo sale price  
  - `SQUARE_FEET` → Size of the property  
  - `NO_BEDROOMS` → Number of bedrooms  
  - `YEAR_BUILT` → Year the condo was built  
  - `ZIP_CODE` → Location of the condo  
  - `GARAGE_SPACES_CC` → Number of garage spaces  

---

## ⚙️ Technologies & Libraries Used  

| Category | Libraries |
|----------|-----------|
| **Data Processing** | pandas, numpy |
| **Visualization** | seaborn, matplotlib |
| **Machine Learning** | scikit-learn (Random Forest, Gradient Boosting, Linear Regression, KNN) |
| **Model Evaluation** | GridSearchCV, mean absolute error (MAE), R² |
| **Web App Framework** | Flask |
| **Deployment** | Render, Gunicorn |

---

## 🏗️ Methodology  

### 🔹 **Data Preprocessing**  
✔ **Outlier Removal**: Using **Interquartile Range (IQR)** on **price per square foot (PPSF)**  
✔ **Feature Engineering**:  
   - **Home Age Calculation**: `HOME_AGE = 2025 - YEAR_BUILT`  
   - **Bathroom Extraction**: Full/Half baths from `BTH_DESC`  
✔ **Categorical Encoding**: One-hot encoding for town, zip code, and condo type  
✔ **Feature Scaling**: StandardScaler applied to numerical features  

---

## 📊 Machine Learning Models  

| Model | Description |
|-------|------------|
| **Random Forest Regressor** 🌲 | Ensemble learning method combining multiple decision trees for better accuracy |
| **Gradient Boosting Regressor** 🚀 | Sequential boosting model that corrects previous model errors |
| **Linear Regression** 📈 | Simple, interpretable model predicting target based on linear features |
| **K-Nearest Neighbors (KNN) 📊** | Distance-based model predicting values by averaging nearest neighbors |

---

## 📈 Results  

- **House Price Distribution** → Histograms before and after outlier removal  
- **PPSF Boxplots** → Data cleaning impact  
- **Model Comparison** → Bar plots showing MAE and R² scores  

### 🎯 **Best Model:** `RandomForestRegressor`  
- **Mean Absolute Error (MAE)**: `$127,692.14`  
- **R² Score**: `0.91`  

---

## 🔧 How to Run the Model  

### **1️⃣ Install Dependencies**  
Ensure you have **Python 3.9+** installed. Then, install the required dependencies.  

📌 **Refer to**: `requirements.txt` for the necessary packages.  

---

### **2️⃣ Train the Model & Save Artifacts**  
Run the script to train the model and save necessary artifacts.  

📌 **Refer to**: `HomePricePredictionModel.py`  

This will generate:  
✔ `model_data.pkl` → Trained model, scaler, MAE, and feature order  

---

### **3️⃣ Run the Flask Web App**  
Start the Flask server to use the price prediction tool.  

📌 **Refer to**: `app.py`  

Then, open **[`http://127.0.0.1:5000/`](http://127.0.0.1:5000/)** in your browser to test the prediction tool.

---
