# Housing Price Prediction Linear Regression Machine Learning

This project contains the implementation of a **Linear Regression** model built completely from scratch (without utilizing machine learning libraries) to predict housing prices.

---

## Project Overview
The main objective is to predict housing prices based on a set of features including square footage, number of bedrooms and bathrooms, offers, brick status, and neighborhood. The core implementation focuses on building the learning algorithm and preprocessing pipeline.

The dataset is initially split into training and testing sets (90\% train, 10\% test).

### Dataset Features:
- `SqFt`: Total square footage of the house (Continuous)
- `Bedrooms`: Number of bedrooms (Discrete)
- `Bathrooms`: Number of bathrooms (Discrete)
- `Offers`: Number of home offers made (Discrete)
- `Brick`: Categorical status (`Yes` / `No`)
- `Neighborhood`: Categorical location (`North`, `East`, `South`, `West`)

---

## Pipeline & Features

### 1. Data Cleaning & Preprocessing
- **Missing Value Imputation:** Detecting `NULL` cells using `isna()`, `info()`, and `describe()`.
- **Outlier Analysis (IQR Method):** Investigating the dataset for potential outliers.
- **Feature Encoding:**
  - The binary feature `Brick` is mapped to `0` and `1`.
  - The categorical feature `Neighborhood` is transformed using One-Hot Encoding into separate indicator variables to suit the mathematical regression model.

### 2. Data Visualization
Exploratory Data Analysis (EDA) is performed using the `seaborn` library to understand the relationships between features and the target price:
- **Distribution & Outliers:** Monitored via Boxplots before and after the clipping procedure.
- **Feature-Target Relationships:** Analyzed using Scatterplots and Linear Regression Plots (`regplot`).
- **Feature Correlation:** Synthesized through an annotated Correlation Matrix Heatmap.

### 3. Model Implementation (From Scratch)
- **Feature Scaling:** Inputs are normalized using Standardization (Z-score normalization).
- **Linear Regression:** Built fundamentally without higher-level ML libraries.
- **Optimization:** Parameters are updated iteratively using **Gradient Descent**.
- **Loss Function:** The model is optimized using **Root Mean Squared Error (RMSE)** as the objective function.

---

## Project Structure

```
├── src/
│   ├── data/
│   │   └── house-prices.csv
│   └── src.py
├── .gitignore
└── README.md
```
