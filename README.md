# 🚗 Used Car Price Prediction

A machine learning project that predicts the estimated price of a used car based on its name, company, manufacturing year, kilometers driven, and fuel type.

## 📌 Project Overview

This project uses the Quikr Car dataset to build a supervised regression model for used-car price prediction.

The complete workflow includes:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Categorical encoding using One-Hot Encoding
- Train-test split
- Regression model training
- Cross-validation
- Feature importance analysis
- Streamlit web application for prediction

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- Google Colab

## 📊 Dataset

The dataset contains information about used cars, including:

- Car Name
- Company
- Manufacturing Year
- Price
- Kilometers Driven
- Fuel Type

After preprocessing and filtering, the working dataset contained 815 records.

## 🔎 Exploratory Data Analysis

EDA was performed to understand the dataset and identify relationships between features and car prices.

Key observations included:

- Manufacturing year showed a positive relationship with price.
- Kilometers driven showed a weak negative relationship with price.
- Car prices were strongly right-skewed.
- Several high-price and high-mileage observations were identified as potential outliers.
- Company and fuel type showed differences in average prices.

## ⚙️ Feature Engineering

The `Price` column was used as the target variable.

Categorical features:

- `name`
- `company`
- `fuel_type`

Numerical features:

- `year`
- `kms_driven`

Low-frequency car names were grouped into an `other` category to reduce the number of rare categories before encoding.

Categorical features were converted using One-Hot Encoding.

## 🤖 Machine Learning Models

The following regression models were evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

### Test Set Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | ₹169,225 | ₹313,882 | 0.516 |
| Decision Tree | ₹163,313 | ₹300,383 | 0.557 |
| Random Forest | ₹159,286 | ₹311,445 | 0.524 |

### Cross-Validation

Random Forest was further evaluated using 5-fold cross-validation.

- Mean CV R²: **0.538**
- Standard deviation of CV R²: **0.112**
- Mean CV MAE: **₹138,525**
- Mean CV RMSE: **₹242,320**

Based on the cross-validation results, Random Forest was selected as the strongest generalization candidate among the evaluated models.

## 📈 Feature Importance

Feature importance analysis from the Random Forest model showed that:

- Manufacturing year was the most important individual feature.
- Several company and car-name categories also contributed strongly.
- Kilometers driven provided additional predictive information.

## 🌐 Streamlit Application

A Streamlit web application was developed to allow users to enter:

- Company
- Car Name
- Manufacturing Year
- Kilometers Driven
- Fuel Type

The application then returns an estimated used-car price.

### Example

**Car:** Maruti Suzuki Swift  
**Company:** Maruti  
**Year:** 2017  
**Kilometers Driven:** 30,000  
**Fuel Type:** Petrol

**Estimated Price:** ₹397,779

> Note: The prediction is an ML-based estimate and may differ from the actual market price.

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone <your-github-repository-url>