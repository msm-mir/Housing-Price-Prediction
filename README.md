# Housing Price Prediction Linear Regression Machine Learning

This project contains the implementation of a **Linear Regression** model built completely from scratch (without utilizing machine learning libraries) to predict housing prices.

---

## Project Overview
The main objective is to predict housing prices based on a set of features including square footage, number of bedrooms and bathrooms, offers, brick status, and neighborhood. The core implementation focuses on building the learning algorithm and preprocessing pipeline.

The dataset is initially split into training and testing sets (90\% train, 10\% test).

---

## Pipeline & Features

### 1. Data Cleaning & Preprocessing
- **Missing Value Imputation:** Detecting `NULL` cells using `isna()`, `info()`, and `describe()`.
- **Outlier Analysis:** Investigating the dataset for potential outliers.
- **Feature Scaling:** Applying data normalization to the features before running the training pipeline to ensure stable gradient descent updates.

### 2. Data Visualization
Exploratory Data Analysis (EDA) is performed using the `seaborn` library to understand the relationships between features and the target price:
- Correlation Analysis using **Heatmaps**.
- Feature trends and distributions using **Scatterplots** and **Regplots**.

### 3. Model Implementation (From Scratch)
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
