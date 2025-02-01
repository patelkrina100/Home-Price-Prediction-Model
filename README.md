# Home Price Prediction Model

**Author**: Krina Patel  
**Description**: Predicting condo sale prices in Boston using multiple regression models.  

## Overview  
This project leverages machine learning to predict the sale price of condos in Boston based on real estate data sourced from an **MA MLS search**. The dataset includes features like square footage, number of bedrooms, location, and age of the property. The models used for prediction include Random Forest, Gradient Boosting, Linear Regression, and K-Nearest Neighbors (KNN).

## 🏠 About This Model  
This **Home Price Prediction Model** is designed to estimate real estate prices based on historical sales data and property features. While this specific implementation was trained using **condo sales data from Boston**, the model structure is flexible and can be easily **adapted for predicting house prices** by incorporating relevant variables like lot size, number of floors, and detached vs. attached homes.

If applied to single-family homes or other property types, **adjustments to feature selection** and **data preprocessing** may be necessary for optimal accuracy.

## 📌 Why This Project Matters  
Real estate pricing is complex, with prices influenced by multiple factors such as location, square footage, and market trends. This model provides **data-driven insights** for:

- **Real Estate Investors**: Helps estimate property values for better investment decisions.  
- **Home Buyers & Sellers**: Assists in determining fair market value based on historical sales data.  
- **Real Estate Agents & Analysts**: Supports pricing strategies and trend forecasting.  

By leveraging **machine learning**, this project enhances the accuracy of price predictions beyond traditional comparative market analysis.

## Dataset  
- **Source**: Massachusetts MLS Search  
- **File**: `MLS Data 1.29.25.csv`  
- **Search Criteria Used**:
  - **Property Type**: Condos (CC)
  - **Timeframe**: Last 2 years
  - **Towns Included**:  
    - Boston: Allston, Back Bay, Beacon Hill, Brighton, Charlestown, Chinatown, East Boston, Jamaica Plain, Leather District, Midtown, North End, Seaport District, South Boston, South End, The Fenway, Waterfront, West End
  - **Advanced Criteria**: Number of Bedrooms (0-3), Number of Total Baths (1+)
- **Key Features**:  
  - `SALE_PRICE`: Final sale price of the condo.  
  - `SQUARE_FEET`: Total square footage.  
  - `NO_BEDROOMS`: Number of bedrooms.  
  - `YEAR_BUILT`: Year the condo was constructed.  
  - `ZIP_CODE`: Location of the condo.  
  - `BTH_DESC`: Description of full and half bathrooms.  
  - `GARAGE_SPACES_CC`: Number of garage spaces.  

## Libraries Used  
This project utilizes the following libraries:  
- **pandas**: Data manipulation and analysis.  
- **numpy**: Efficient numerical computations.  
- **seaborn & matplotlib**: Data visualization and exploratory analysis.  
- **scikit-learn**: Machine learning models, feature scaling, and model evaluation.  
- **GridSearchCV**: Hyperparameter tuning for model optimization.  

## Methodology  
### Data Preprocessing  
- **Outlier Detection & Removal**: Identified outliers based on **price per square foot (PPSF)** using the **Interquartile Range (IQR)** method.  
- **Feature Engineering**: Converted year built into home age, extracted full and half baths from `BTH_DESC`, and applied **one-hot encoding** to categorical features.  
- **Data Scaling**: Standardized numerical features like square footage and home age.  
- **Luxury Classification**: Defined top 10% of condos by PPSF as "Luxury" for better model differentiation.  

## Machine Learning Models  
- **Random Forest Regressor**: An ensemble learning method that combines multiple decision trees to improve accuracy and prevent overfitting.  
- **Gradient Boosting Regressor**: A boosting algorithm that builds models sequentially, optimizing performance by correcting errors of previous models. 
- **Linear Regression**: A simple and interpretable model that predicts target values based on a linear combination of input features.  
- **K-Nearest Neighbors (KNN) with Hyperparameter Tuning**: A distance-based algorithm that predicts values by averaging the outcomes of the nearest neighbors.  

## Results & Visualizations  
- **House Price Distribution**: Histograms before and after outlier removal.  
- **PPSF Boxplots**: Comparison before and after data cleaning.  
- **Model Comparison**: Bar plots visualizing MAE and R² scores.  

## 🔮 Future Improvements  
To improve performance and usability, future iterations of this project could include:
- **Real-Time Data Integration**: Pulling in live MLS listings for up-to-date predictions.  
- **More Features**: Including interest rates, HOA fees, proximity to attractions/schools/shopping centers, and neighborhood crime statistics.  
- **Web App Deployment**: Creating a user-friendly web interface where users can input condo details and get instant price predictions.  
