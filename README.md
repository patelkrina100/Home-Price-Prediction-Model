# Home-Price-Prediction-Model
A machine learning model that predicts condo sale prices in Boston using multiple regression models!

## Description
Predicting condo sale prices in Boston using multiple regression models.

## Overview

This project leverages machine learning to predict the sale price of condos in Boston based on real estate data sourced from an MA MLS search. The dataset includes features like square footage, number of bedrooms, location, and age of the property. The models used for prediction include Random Forest, Gradient Boosting, Linear Regression, and K-Nearest Neighbors (KNN).

## Dataset: MLS Data 1.29.25.csv

**Source**: Massachusetts MLS Search  

### Search Criteria Used:
- **Property Type**: Condos (CC)  
- **Timeframe**: Last 2 years  
- **Towns Included**:  
  - Boston: Allston, Back Bay, Beacon Hill, Brighton, Charlestown, Chinatown, East Boston, Jamaica Plain, Leather District, Midtown, North End, Seaport District, South Boston, South End, The Fenway, Waterfront, West End  
- **Advanced Criteria**:  
  - Number of Bedrooms: 0-3  
  - Number of Total Baths: 1+  

## Key Features
- **SALE_PRICE**: Final sale price of the condo.  
- **SQUARE_FEET**: Total square footage.  
- **NO_BEDROOMS**: Number of bedrooms.  
- **YEAR_BUILT**: Year the condo was constructed.  
- **ZIP_CODE**: Location of the condo.  
- **BTH_DESC**: Description of full and half bathrooms.  
- **GARAGE_SPACES_CC**: Number of garage spaces.  

## Methodology

### Data Preprocessing
1. **Outlier Detection & Removal**: Identified outliers based on price per square foot (PPSF) using the Interquartile Range (IQR) method.  
2. **Feature Engineering**: Converted year built into home age, extracted full and half baths from `BTH_DESC`, and applied one-hot encoding to categorical features.  
3. **Data Scaling**: Standardized numerical features like square footage and home age.  
4. **Luxury Classification**: Defined top 10% of condos by PPSF as "Luxury" for better model differentiation.  

## Machine Learning Models
1. **Random Forest Regressor**  
2. **Gradient Boosting Regressor**  
3. **Linear Regression**  
4. **K-Nearest Neighbors (KNN) with Hyperparameter Tuning**  

## Results & Visualizations
- **Model Comparison**: Table & Bar plots visualizing MAE and R² scores.  
